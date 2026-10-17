"""Восстанавливает best_configs.json из уже рассчитанного matching_grid.csv.
Запуск из корня проекта: python restore_step05_config.py
"""
import csv
import json
from pathlib import Path

folder = Path('results/step05/validation')
source = folder / 'matching_grid.csv'
if not source.is_file():
    raise SystemExit(f'Не найден файл: {source}. Запустите скрипт из корня проекта.')

with source.open(encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))

methods = ('SIFT', 'ORB', 'AKAZE', 'BRISK')
configs = {}
for method in methods:
    candidates = [r for r in rows if r['method'].upper() == method]
    if not candidates:
        raise SystemExit(f'В CSV отсутствуют результаты метода {method}')
    best = max(candidates, key=lambda r: (float(r['f1']), -float(r['mean_runtime_ms'])))
    threshold = float(best['akaze_threshold']) if method == 'AKAZE' and best['akaze_threshold'] else None
    configs[method] = {
        'ratio': float(best['ratio']),
        'min_inliers': int(best['min_inliers']),
        'ransac_threshold': 3.0,
        'min_inlier_ratio': 0.35,
        'nfeatures': 2000,
        'akaze_threshold': threshold,
        'validation_f1': float(best['f1']),
    }

report_path = folder / 'best_configs.json'
if report_path.exists():
    try:
        report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    except (ValueError, UnicodeError):
        report = {}
else:
    report = {}
report['split'] = 'validation'
report['phase'] = 'matching'
report['selected_akaze_threshold'] = configs['AKAZE']['akaze_threshold']
report['best_configs'] = configs
report['note'] = 'Восстановлено из matching_grid.csv после полного перебора параметров.'
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print('Готово:', report_path)
for name, params in configs.items():
    print(f"{name}: F1={params['validation_f1']:.4f}, ratio={params['ratio']}, min_inliers={params['min_inliers']}")
