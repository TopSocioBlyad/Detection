from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reid.audit_dataset import audit, write_outputs
from reid.folds import build_identity_folds, partition_records
from reid.matching import ImageRecord, load_manifest
from reid.metrics import binary_metrics, select_f1_threshold
from reid.pairs import build_pairs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def record(path: str, identity: str, capture: str = "unknown") -> ImageRecord:
    return ImageRecord(path, identity, "unknown", Path(path).name, capture, "ok")


class DatasetAuditTests(unittest.TestCase):
    def test_repository_dataset_audit_and_manifest_round_trip(self):
        manifest, report = audit(REPOSITORY_ROOT / "data_reid", REPOSITORY_ROOT)
        self.assertEqual(report["total_files"], 312)
        self.assertEqual(report["unreadable_files"], [])
        self.assertEqual(report["quality_status_counts"]["exact_duplicate"], 6)
        self.assertEqual(report["quality_status_counts"]["near_duplicate"], 6)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_outputs(manifest, report, output)
            usable = load_manifest(output / "manifest.csv")
        self.assertEqual(len(usable), 300)
        self.assertEqual(len({item.pothole_id for item in usable}), 12)


class SplitAndPairTests(unittest.TestCase):
    def test_grouped_folds_have_no_identity_leakage(self):
        records = load_manifest(REPOSITORY_ROOT / "results/reid_audit/manifest.csv")
        folds = build_identity_folds(records, n_splits=4, seed=42)
        self.assertEqual([len(fold) for fold in folds], [3, 3, 3, 3])
        self.assertEqual(len(set().union(*map(set, folds))), 12)
        for fold in folds:
            training, testing = partition_records(records, fold)
            self.assertFalse(
                {item.pothole_id for item in training}
                & {item.pothole_id for item in testing}
            )

    def test_known_same_capture_is_not_a_positive_pair(self):
        records = [
            record("a1.png", "a", "capture_a"),
            record("a2.png", "a", "capture_a"),
            record("a3.png", "a", "capture_b"),
            record("b1.png", "b", "capture_c"),
            record("b2.png", "b", "capture_d"),
        ]
        pairs = build_pairs(records, max_positive_per_identity=10, seed=1)
        positives = [pair for pair in pairs if pair.label]
        self.assertTrue(positives)
        self.assertTrue(all(pair.capture_relation == "different" for pair in positives))
        self.assertFalse(
            any({pair.image_a, pair.image_b} == {"a1.png", "a2.png"} for pair in positives)
        )


class MetricTests(unittest.TestCase):
    def test_threshold_is_selected_from_training_scores(self):
        labels = [1, 1, 0, 0]
        scores = [9.0, 8.0, 2.0, 0.0]
        threshold, metrics = select_f1_threshold(labels, scores)
        self.assertEqual(threshold, 8.0)
        self.assertEqual(metrics["f1"], 1.0)
        evaluated = binary_metrics(labels, scores, threshold)
        self.assertEqual((evaluated["tp"], evaluated["fp"]), (2, 0))


if __name__ == "__main__":
    unittest.main()
