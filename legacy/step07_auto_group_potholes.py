"""Discover likely repeated *physical* potholes without assuming file IDs are identities.

Run from project root:
    python step07_auto_group_potholes.py --data-root data

This is a conservative *identity annotation aid*, not an unbiased identity
benchmark. A singleton is 'unknown', not proven to depict a different pothole.

Needs: opencv-contrib-python (4.11.0.86 recommended), numpy.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ('training', 'validation', 'testing')
EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_image(path: Path, flags: int):
    # imdecode works with Cyrillic Windows paths, unlike some OpenCV builds.
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def write_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f'Image encoding failed: {path}')
    buf.tofile(str(path))


def phash(gray: np.ndarray) -> np.ndarray:
    values = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8]
    part = values.flatten()[1:]
    return part > np.median(part)


def rotation_phashes(gray: np.ndarray):
    return [phash(np.rot90(gray, k)) for k in range(4)]


def rot_distance(a, b) -> int:
    return min(int(np.count_nonzero(ha != hb)) for ha in a for hb in b)


def mask_for(rgb_file: Path, label_dir: Path):
    for suffix in ('.png', '.jpg', '.bmp', '.jpeg', '.tif'):
        p = label_dir / (rgb_file.stem + suffix)
        if p.exists():
            img = read_image(p, cv2.IMREAD_GRAYSCALE)
            return (img > 127).astype(np.uint8) if img is not None else None
    return None


@dataclass
class Entry:
    key: str
    split: str
    stem: str
    image: np.ndarray
    gray: np.ndarray
    mask: np.ndarray | None
    hashes: list
    kp: list
    desc: np.ndarray | None


def load_entries(root: Path, nfeatures: int) -> list[Entry]:
    if not hasattr(cv2, 'SIFT_create'):
        raise RuntimeError('OpenCV SIFT_create unavailable. Install opencv-contrib-python==4.11.0.86')
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    entries = []
    for split in SPLITS:
        folder = root / split / 'rgb'
        if not folder.exists():
            print(f'No split RGB directory: {folder}')
            continue
        files = sorted((p for p in folder.iterdir() if p.suffix.lower() in EXTS), key=lambda p: p.name)
        for p in files:
            im = read_image(p, cv2.IMREAD_COLOR)
            if im is None:
                print(f'WARNING: cannot read {p}')
                continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mask = mask_for(p, root / split / 'label')
            # Full frame rather than crop: enough shared asphalt context for
            # estimating homography under viewpoint and orientation changes.
            kp, desc = sift.detectAndCompute(gray, None)
            entries.append(Entry(f'{split}/{p.stem}', split, p.stem, im, gray,
                                 mask, rotation_phashes(gray), kp, desc))
        print(f'Loaded {split}: {len(files)} RGB images')
    return entries


def candidates(entries: list[Entry], nearest: int, neighbors: int):
    """Rotation-aware shortlist + filename adjacency, across all splits."""
    pairs = set()
    N = len(entries)
    for i in range(N):
        scores = [(rot_distance(entries[i].hashes, entries[j].hashes), j)
                  for j in range(N) if i != j]
        scores.sort()
        for _, j in scores[:nearest]:
            pairs.add((min(i, j), max(i, j)))
    by_split = defaultdict(list)
    for i, e in enumerate(entries):
        by_split[e.split].append(i)
    for ids in by_split.values():
        # Proximity in sorted names helps for sequences with changed viewpoint.
        ids.sort(key=lambda i: entries[i].stem)
        for n, i in enumerate(ids):
            for j in ids[n+1:n+1+neighbors]:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None, H: np.ndarray):
    if a is None or b is None:
        return None
    if int(a.sum()) == 0 or int(b.sum()) == 0:
        return None
    warped = cv2.warpPerspective(a, H, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST)
    intersection = np.count_nonzero((warped > 0) & (b > 0))
    union = np.count_nonzero((warped > 0) | (b > 0))
    return float(intersection / union) if union else None


def verify(a: Entry, b: Entry, ratio: float, ransac: float):
    result = dict(good=0, inliers=0, inlier_ratio=0.0, mask_iou=None)
    if a.desc is None or b.desc is None or len(a.desc) < 4 or len(b.desc) < 4:
        return result
    # BF L2 matches previously computed SIFT descriptors; rotation-invariant.
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(a.desc, b.desc, k=2)
    good = [m for row in knn if len(row) == 2 for m, n in [row] if m.distance < ratio * n.distance]
    result['good'] = len(good)
    if len(good) < 4:
        return result
    src = np.float32([a.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([b.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None or inlier_mask is None or not np.isfinite(H).all():
        return result
    inliers = int(inlier_mask.sum())
    result['inliers'] = inliers
    result['inlier_ratio'] = round(inliers / len(good), 4)
    # Prevent pathological/degenerate projective transformations.
    try:
        corners = np.float32([[0, 0], [a.gray.shape[1], 0],
                              [a.gray.shape[1], a.gray.shape[0]],
                              [0, a.gray.shape[0]]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H)
        area = abs(cv2.contourArea(warped))
        src_area = a.gray.shape[0] * a.gray.shape[1]
        if not (0.05 * src_area < area < 20 * src_area):
            return result
    except cv2.error:
        return result
    result['mask_iou'] = mask_iou(a.mask, b.mask, H)
    return result


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def root(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, i, j):
        a, b = self.root(i), self.root(j)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def contact_sheet(entries: list[Entry], ids: list[int], destination: Path):
    width, height, cols = 160, 185, 5
    rows = (len(ids) + cols - 1) // cols
    canvas = np.full((height * rows, width * cols, 3), 250, dtype=np.uint8)
    for k, i in enumerate(ids):
        im = cv2.resize(entries[i].image, (150, 150))
        y, x = divmod(k, cols)
        canvas[y*height:y*height+150, x*width:x*width+150] = im
        cv2.putText(canvas, entries[i].key[:24], (x*width+1, y*height+167),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 0), 1, cv2.LINE_AA)
    write_image(destination, canvas)


def main():
    ap = argparse.ArgumentParser(description='Conservative physical pothole identity candidate clustering')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--output', default='results/step07/identity_audit')
    ap.add_argument('--nearest', type=int, default=16, help='Rotation-aware nearest neighbors per image')
    ap.add_argument('--neighbors', type=int, default=5, help='Adjacent filenames per split')
    ap.add_argument('--nfeatures', type=int, default=1200)
    ap.add_argument('--ratio', type=float, default=0.75)
    ap.add_argument('--ransac', type=float, default=3.0)
    ap.add_argument('--min-inliers', type=int, default=35)
    ap.add_argument('--min-inlier-ratio', type=float, default=0.45)
    ap.add_argument('--min-mask-iou', type=float, default=0.5)
    args = ap.parse_args()
    entries = load_entries(Path(args.data_root), args.nfeatures)
    if len(entries) < 2:
        ap.error('Need at least 2 images from data/<split>/rgb')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pairs = candidates(entries, args.nearest, args.neighbors)
    print(f'{len(entries)} frames; {len(pairs)} candidate pairs for SIFT/RANSAC.')
    uf = UnionFind(len(entries))
    results = []
    for count, (i, j) in enumerate(pairs, 1):
        a, b = entries[i], entries[j]
        result = verify(a, b, args.ratio, args.ransac)
        mask_score = result['mask_iou']
        strong = (result['inliers'] >= args.min_inliers and
                  result['inlier_ratio'] >= args.min_inlier_ratio and
                  mask_score is not None and mask_score >= args.min_mask_iou)
        # Candidate requiring review, not automatically merged.
        possible = (not strong and result['inliers'] >= 12 and
                    result['inlier_ratio'] >= 0.3 and
                    (mask_score is None or mask_score >= 0.15))
        status = 'strong_candidate' if strong else ('possible' if possible else 'unconfirmed')
        if strong:
            uf.union(i, j)
        results.append({'image_a': a.key, 'image_b': b.key, **result, 'status': status})
        if count % 500 == 0:
            print(f'Pairs checked: {count}/{len(pairs)}', flush=True)
    results.sort(key=lambda r: (0 if r['status'] == 'strong_candidate' else
                                1 if r['status'] == 'possible' else 2,
                                -r['inliers']))
    fields = ['image_a','image_b','good','inliers','inlier_ratio','mask_iou','status']
    with (out / 'pair_candidates.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    groups = defaultdict(list)
    for i in range(len(entries)):
        groups[uf.root(i)].append(i)
    ordered = sorted(groups.values(), key=lambda ids: (-len(ids), entries[ids[0]].key))
    gid_for = {i: f'auto_group_{g:04d}' for g, ids in enumerate(ordered, 1) for i in ids}
    with (out / 'group_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id','provisional_group_id','group_size','status'])
        writer.writeheader()
        for ids in ordered:
            for i in ids:
                writer.writerow(dict(image_id=entries[i].key, provisional_group_id=gid_for[i],
                                     group_size=len(ids), status='provisional_group' if len(ids)>1
                                     else 'unknown_singleton'))
    contact_dir = out / 'group_previews'
    for number, ids in enumerate(ordered, 1):
        if len(ids) > 1:
            contact_sheet(entries, ids, contact_dir / f'auto_group_{number:04d}.jpg')
    different_splits = [ids for ids in ordered if len({entries[i].split for i in ids}) > 1]
    report = {'images':len(entries), 'candidate_pairs_checked':len(pairs),
              'strong_pairs':sum(r['status']=='strong_candidate' for r in results),
              'possible_pairs':sum(r['status']=='possible' for r in results),
              'provisional_groups':len(ordered),
              'groups_with_2_or_more_frames':sum(len(ids)>1 for ids in ordered),
              'unknown_singletons':sum(len(ids)==1 for ids in ordered),
              'groups_spanning_splits':len(different_splits),
              'cross_split_examples':[[entries[i].key for i in ids[:20]] for ids in different_splits[:10]],
              'caveats':[
                  'Automatic edges are candidates, not certified physical pothole identities.',
                  'A singleton is UNKNOWN, not a confirmed different pothole.',
                  'Missing edges do not establish different identities.',
                  'Transitive groups can contain false merges; inspect previews of suspicious groups.',
                  'Grouping inspects labels from testing as metadata and cannot restore a pristine holdout.',
                  'No algorithm thresholds should be tuned using these testing results.'
              ]}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('\n' + json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Written: {out.resolve()}')


if __name__ == '__main__':
    main()
