"""Seeded subsets of VLMEvalKit datasets that its official evaluators still accept.

MMBench is scored with circular evaluation: each question appears several
times with its options rotated (indices i, i + 1e6, i + 2e6, ...), and counts
as correct only if every rotation is answered correctly. A subset must keep
whole groups, so MMBench is sampled by question group, not by row.
"""

import random


def group_index(dataset_name: str, index: int) -> int:
    if "mmbench" in dataset_name.lower():
        return int(index % 1_000_000)
    return int(index)


def sample_indices(dataset_name: str, indices, n: int, seed: int = 0) -> list:
    """Row indices of a seeded sample of n question groups (all groups if n <= 0)."""
    indices = [int(i) for i in indices]
    groups = sorted({group_index(dataset_name, i) for i in indices})
    if 0 < n < len(groups):
        groups = set(random.Random(seed).sample(groups, n))
    else:
        groups = set(groups)
    return [i for i in indices if group_index(dataset_name, i) in groups]
