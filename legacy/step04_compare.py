"""Сравнение SIFT, ORB, AKAZE, BRISK на Pothole-600.

Требует step03_batch.py рядом с этим файлом (для одинаковой подготовки данных).
Запуск: python step04_compare.py --split validation --limit 5
Все метрики IoU рассчитываются только ПОСЛЕ принятия решения о детекции.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import step03_batch as base

METHODS = ('SIFT', 'ORB', 'AKAZE', 'BRISK')
DEFAULT_SCENARIOS = ('identity','rotate_20','rotate_60','scale_0.75','scale_1.25',
                     'perspective','darker','blur_sigma_0.8','blur_sigma_1.5',
                     'blur_sigma_2.5','blur_sigma_4.0')


def make_method(name: str, nfeatures: int):
    if name == 'SIFT':
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2
    if name == 'ORB':
        return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING
    if name == 'AKAZE':
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if name == 'BRISK':
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    raise ValueError(f'Неизвестный метод: {name}')


def make_scene(image, mask, scenario):
    if scenario.startswith('blur_sigma_'):
        sigma = float(scenario.removeprefix('blur_sigma_'))
        if sigma <= 0:
            raise ValueError('Sigma must be > 0')
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma), mask.copy()
    return base.synthetic_scene(image, mask, scenario)


def compute_features(detector, image):
    return detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)


def detect(template_features, template_mask, scene, detector, norm, *, ratio,
           ransac_threshold, min_inliers, min_inlier_ratio):
    """Не использует истинную маску сцены. Возвращает решение, маску и метрики."""
    start = time.perf_counter()
    kp1, des1 = template_features
    pred = np.zeros(scene.shape[:2], dtype=np.uint8)
    metrics = dict(found=False, reason='', kp_template=len(kp1), kp_scene=0,
                   good_matches=0, inliers=0, inlier_ratio=0.,
                   reprojection_px=None, runtime_ms=None)
    try:
        kp2, des2 = compute_features(detector, scene)
        metrics['kp_scene'] = len(kp2)
        if des1 is None or des2 is None or len(des2) < 2:
            metrics['reason'] = 'no_descriptors'
        else:
            matches = cv2.BFMatcher(norm, crossCheck=False).knnMatch(des1, des2, k=2)
            good = [m for pair in matches if len(pair)==2 for m,n in [pair]
                    if m.distance < ratio*n.distance]
            metrics['good_matches'] = len(good)
            if len(good) < 4:
                metrics['reason'] = 'few_matches'
            else:
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, flags = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
                n = int(flags.sum()) if flags is not None else 0
                metrics['inliers'] = n
                metrics['inlier_ratio'] = n/len(good)
                if H is None or not np.isfinite(H).all():
                    metrics['reason'] = 'no_homography'
                elif n < min_inliers or metrics['inlier_ratio'] < min_inlier_ratio:
                    metrics['reason'] = 'few_inliers'
                else:
                    src2 = cv2.perspectiveTransform(src, H).reshape(-1,2)
                    errors = np.linalg.norm(src2-dst.reshape(-1,2),axis=1)
                    metrics['reprojection_px'] = float(np.mean(errors[flags.ravel().astype(bool)]))
                    h,w = scene.shape[:2]
                    th,tw = template_mask.shape[:2]
                    corners = np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]]).reshape(-1,1,2)
                    transformed = cv2.perspectiveTransform(corners, H).reshape(-1,2)
                    if not np.isfinite(transformed).all():
                        metrics['reason'] = 'invalid_geometry'
                    else:
                        area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
                        if not (20 <= area <= 4*w*h):
                            metrics['reason'] = 'invalid_geometry'
                        else:
                            pred = cv2.warpPerspective(template_mask,H,(w,h),
                                        flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                            if not pred.any():
                                metrics['reason'] = 'empty_prediction'
                            else:
                                metrics['found'] = True
                                metrics['reason'] = 'ok'
    except cv2.error as exc:
        metrics['reason'] = 'opencv_error:' + str(exc).splitlines()[0][:130]
    metrics['runtime_ms'] = round((time.perf_counter()-start)*1000, 2)
    return metrics, pred


def summarize(rows, iou_threshold):
    pos = [r for r in rows if r['pair_type']=='positive']
    neg = [r for r in rows if r['pair_type']=='negative']
    tp = int(sum(bool(r['found']) and r['iou'] >= iou_threshold for r in pos))
    fn = len(pos)-tp
    wrong_positive_localization = int(sum(bool(r['found']) and r['iou'] < iou_threshold for r in pos))
    neg_fp = int(sum(bool(r['found']) for r in neg))
    fp = wrong_positive_localization + neg_fp
    tn = len(neg)-neg_fp
    p = tp/(tp+fp) if tp+fp else 0.
    rec = tp/(tp+fn) if tp+fn else 0.
    f1 = 2*p*rec/(p+rec) if p+rec else 0.
    return dict(samples=len(rows), positives=len(pos), negatives=len(neg),
                tp=tp, fn=fn, fp=fp, tn=tn,
                false_positive_negative_pairs=neg_fp,
                false_positive_wrong_localization=wrong_positive_localization,
                precision=round(p,5),recall=round(rec,5),f1=round(f1,5),
                mean_iou_including_failures=round(float(np.mean([r['iou'] for r in pos])),5) if pos else None,
                mean_runtime_ms=round(float(np.mean([r['runtime_ms'] for r in rows])),2) if rows else None,
                mean_total_ms=round(float(np.mean([r['total_ms'] for r in rows])),2) if rows else None)


def plots(rows, summaries, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(summaries)
    for field, filename, ylabel in (
        ('f1','f1_comparison.png','F1'),
        ('mean_total_ms','runtime_comparison.png','Time per pair, ms')):
        fig,ax=plt.subplots(figsize=(8,4.7))
        ax.bar(methods,[summaries[m][field] or 0 for m in methods])
        ax.set_ylabel(ylabel)
        if field=='f1': ax.set_ylim(0,1.05)
        fig.tight_layout(); fig.savefig(out/filename,dpi=160); plt.close(fig)
    scenarios = list(dict.fromkeys(r['scenario'] for r in rows if r['pair_type']=='positive'))
    fig,ax = plt.subplots(figsize=(max(10,len(scenarios)*1.2),5))
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    for j,m in enumerate(methods):
        vals=[]
        for s in scenarios:
            rr=[r for r in rows if r['method']==m and r['pair_type']=='positive' and r['scenario']==s]
            vals.append(np.mean([r['success'] for r in rr]) if rr else 0)
        ax.bar(x - .4 + width*(j+.5),vals,width,label=m)
    ax.set_xticks(x,scenarios,rotation=40,ha='right')
    ax.set_ylabel('Localization success (IoU threshold)')
    ax.set_ylim(0,1.05)
    ax.legend(); fig.tight_layout();fig.savefig(out/'success_by_scenario.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='Pothole-600: SIFT vs ORB vs AKAZE vs BRISK')
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--split',choices=('training','validation','testing'),default='validation')
    parser.add_argument('--limit',type=int,default=5,help='0 = вся выборка')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--scenarios',nargs='+',default=list(DEFAULT_SCENARIOS))
    parser.add_argument('--negatives',type=int,default=1)
    parser.add_argument('--padding',type=int,default=45)
    parser.add_argument('--nfeatures',type=int,default=2000,
                        help='Лимит SIFT/ORB; AKAZE/BRISK используют свои параметры')
    parser.add_argument('--ratio',type=float,default=.75)
    parser.add_argument('--ransac-threshold',type=float,default=3.0)
    parser.add_argument('--min-inliers',type=int,default=10)
    parser.add_argument('--min-inlier-ratio',type=float,default=.35)
    parser.add_argument('--iou-threshold',type=float,default=.5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--out',default=None)
    a=parser.parse_args()
    if a.limit<0 or a.negatives<0 or a.padding<0 or a.nfeatures<1 or a.min_inliers<4 or not 0<a.ratio<1 or not 0<a.iou_threshold<=1 or not 0<=a.min_inlier_ratio<=1 or a.ransac_threshold<=0:
        parser.error('Недопустимые параметры')
    if len(set(a.methods))!=len(a.methods): parser.error('Повторяющиеся методы')
    pairs=base.load_pairs(Path(a.data_root),a.split)
    selected=pairs[:a.limit] if a.limit else pairs
    rng=np.random.default_rng(a.seed)
    # Набор отрицательных пар один и тот же у всех методов.
    negatives={}
    for stem,_,_ in selected:
        others=[p for p in pairs if p[0]!=stem]
        if not others or a.negatives==0: negatives[stem]=[]
        else: negatives[stem]=[others[int(i)] for i in rng.choice(len(others),size=min(a.negatives,len(others)),replace=False)]
    out=Path(a.out) if a.out else Path('results')/'step04'/a.split
    out.mkdir(parents=True,exist_ok=True)
    rows=[];skipped=[]
    # Для честного сравнения прогрев и измерение выполняются отдельно на каждом методе.
    for method in a.methods:
        detector,norm=make_method(method,a.nfeatures)
        print(f'Метод {method}: {len(selected)} эталонов',flush=True)
        for idx,(stem,rgb_path,mask_path) in enumerate(selected,1):
            try:
                image,mask=base.read_data(rgb_path,mask_path)
                template,tm,_=base.make_template(image,mask,a.padding)
                t0=time.perf_counter()
                tf=compute_features(detector,template)
                template_ms=(time.perf_counter()-t0)*1000
                for scenario in a.scenarios:
                    scene,truth=make_scene(image,mask,scenario)
                    if not truth.any():
                        skipped.append(dict(method=method,image=stem,scenario=scenario,reason='object_out_of_frame'))
                        continue
                    metrics,pred=detect(tf,tm,scene,detector,norm,ratio=a.ratio,
                           ransac_threshold=a.ransac_threshold,min_inliers=a.min_inliers,
                           min_inlier_ratio=a.min_inlier_ratio)
                    iou=base.iou_score(pred,truth) if metrics['found'] else 0.
                    rows.append(dict(method=method,image_id=stem,scene_id=stem,
                         pair_type='positive',scenario=scenario,iou=round(iou,6),
                         success=bool(metrics['found'] and iou>=a.iou_threshold),
                         template_ms=round(template_ms,2),total_ms=round(template_ms+metrics['runtime_ms'],2),**metrics))
                for other_stem,other_rgb,_ in negatives[stem]:
                    scene=cv2.imread(str(other_rgb),cv2.IMREAD_COLOR)
                    if scene is None:
                        skipped.append(dict(method=method,image=other_stem,reason='unreadable_negative'))
                        continue
                    metrics,_=detect(tf,tm,scene,detector,norm,ratio=a.ratio,
                           ransac_threshold=a.ransac_threshold,min_inliers=a.min_inliers,
                           min_inlier_ratio=a.min_inlier_ratio)
                    rows.append(dict(method=method,image_id=stem,scene_id=other_stem,
                         pair_type='negative',scenario='other_pothole',iou=None,
                         success=not metrics['found'],template_ms=round(template_ms,2),
                         total_ms=round(template_ms+metrics['runtime_ms'],2),**metrics))
            except (ValueError,OSError,cv2.error) as e:
                skipped.append(dict(method=method,image=stem,reason=str(e)))
            if idx%20==0 or idx==len(selected): print(f'  {idx}/{len(selected)}',flush=True)
    if not rows: raise RuntimeError('Нет результатов; проверь пути и данные')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(rows[0].keys())
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summaries={method:summarize([r for r in rows if r['method']==method],a.iou_threshold) for method in a.methods}
    report=dict(split=a.split,settings=vars(a),results=summaries,skipped_count=len(skipped),
                caution=('Положительные пары синтетические; разные файлы могут содержать одну физическую яму. '
                         'FP при неверной локализации положительной пары учитывается также как FN. '
                         'Mean total includes repeated extraction of template; detector runtime excludes it. '
                         'Параметры одинаковы численно, но не обязательно оптимальны для каждого дескриптора.'))
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'skipped.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2),encoding='utf-8')
    plots(rows,summaries,out)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('Результаты:',out.resolve())

if __name__=='__main__': main()
