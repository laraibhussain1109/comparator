"""Physical-part provenance and validated dataset storage.

Migration: use DatasetStore with explicit view, label, and physical part ID.
The old flat-folder scan_dataset interface is retired; it could not guarantee
independent parts or prevent leakage between captures of the same housing.
"""
from src.data.dataset import DatasetIntegrityError, DatasetSplit, DatasetStore, Sample, VIEWS, split_by_part

__all__ = ["DatasetIntegrityError", "DatasetSplit", "DatasetStore", "Sample", "VIEWS", "split_by_part"]
