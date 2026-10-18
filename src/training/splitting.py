"""Leakage-safe splitting by physical part.

Migration: call split_by_part(samples, seed=...), not split_dataset(inventory).
Every capture of a physical part stays in one partition, including across views.
"""
from src.data.dataset import DatasetIntegrityError, DatasetSplit, Sample, split_by_part

__all__ = ["DatasetIntegrityError", "DatasetSplit", "Sample", "split_by_part"]
