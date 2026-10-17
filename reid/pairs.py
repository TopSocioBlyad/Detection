"""Deterministic positive and negative pair construction."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, product

from reid.matching import ImageRecord


@dataclass(frozen=True)
class Pair:
    image_a: str
    image_b: str
    label: int
    identity_a: str
    identity_b: str
    capture_relation: str

    @property
    def pair_type(self) -> str:
        return "positive" if self.label else "negative"


def capture_relation(first: ImageRecord, second: ImageRecord) -> str:
    if first.capture_id == "unknown" or second.capture_id == "unknown":
        return "unknown"
    return "same" if first.capture_id == second.capture_id else "different"


def _balanced_negative_sample(
    grouped_records: dict[str, list[ImageRecord]], target: int, rng: random.Random
) -> list[Pair]:
    queues = []
    for first_identity, second_identity in combinations(sorted(grouped_records), 2):
        candidates = [
            Pair(
                first.image_path,
                second.image_path,
                0,
                first_identity,
                second_identity,
                capture_relation(first, second),
            )
            for first, second in product(
                grouped_records[first_identity], grouped_records[second_identity]
            )
        ]
        rng.shuffle(candidates)
        queues.append(deque(candidates))
    rng.shuffle(queues)

    selected = []
    while queues and len(selected) < target:
        next_round = []
        for queue in queues:
            if queue and len(selected) < target:
                selected.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    return selected


def build_pairs(
    records: list[ImageRecord],
    *,
    max_positive_per_identity: int = 30,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> list[Pair]:
    if max_positive_per_identity < 1:
        raise ValueError("max_positive_per_identity must be positive")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.pothole_id].append(record)
    if len(grouped) < 2:
        raise ValueError("At least two identities are required for negative pairs")

    rng = random.Random(seed)
    positives = []
    for identity in sorted(grouped):
        candidates = []
        for first, second in combinations(sorted(grouped[identity], key=lambda r: r.image_path), 2):
            relation = capture_relation(first, second)
            # Known frames from one recording are not independent positives.
            if relation == "same":
                continue
            candidates.append(
                Pair(
                    first.image_path,
                    second.image_path,
                    1,
                    identity,
                    identity,
                    relation,
                )
            )
        rng.shuffle(candidates)
        positives.extend(candidates[:max_positive_per_identity])
    if not positives:
        raise ValueError("No valid positive pairs could be constructed")

    negative_target = round(len(positives) * negative_ratio)
    negatives = _balanced_negative_sample(grouped, negative_target, rng)
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs
