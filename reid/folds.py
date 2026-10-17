"""Identity-disjoint grouped cross-validation splits."""

from __future__ import annotations

import random
import math
from collections import Counter

from reid.matching import ImageRecord


def build_identity_folds(
    records: list[ImageRecord], *, n_splits: int = 4, seed: int = 42
) -> list[tuple[str, ...]]:
    counts = Counter(record.pothole_id for record in records)
    if n_splits < 2 or n_splits > len(counts):
        raise ValueError("n_splits must be between 2 and the identity count")

    rng = random.Random(seed)
    identities = list(counts)
    rng.shuffle(identities)
    identities.sort(key=lambda identity: counts[identity], reverse=True)

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    identity_capacity = math.ceil(len(identities) / n_splits)
    for identity in identities:
        available = [
            index
            for index in range(n_splits)
            if len(folds[index]) < identity_capacity
        ]
        target = min(
            available,
            key=lambda index: (fold_sizes[index], len(folds[index]), index),
        )
        folds[target].append(identity)
        fold_sizes[target] += counts[identity]

    result = [tuple(sorted(fold)) for fold in folds]
    flattened = [identity for fold in result for identity in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(counts):
        raise RuntimeError("Invalid grouped folds: identities overlap or are missing")
    return result


def partition_records(
    records: list[ImageRecord], test_identities: tuple[str, ...]
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    test_set = set(test_identities)
    training = [record for record in records if record.pothole_id not in test_set]
    testing = [record for record in records if record.pothole_id in test_set]
    if not training or not testing:
        raise ValueError("Both training and testing partitions must be non-empty")
    if {record.pothole_id for record in training} & {
        record.pothole_id for record in testing
    }:
        raise RuntimeError("Identity leakage between training and testing")
    return training, testing
