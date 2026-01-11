"""
FreeTime Datasets Package

Provides data loading utilities for FreeTimeGS training:
- FreeTimeParser: Loads COLMAP camera poses and image paths
- FreeTimeDataset: PyTorch dataset for training/validation
- load_multiframe_colmap_points: 4D initialization from multi-frame COLMAP
- load_single_frame_with_velocity: Single-frame initialization with velocity
"""

from .FreeTime_dataset import (
    FreeTimeParser,
    FreeTimeDataset,
    load_multiframe_colmap_points,
    load_single_frame_with_velocity,
    find_available_colmap_frames,
)

from .normalize import (
    similarity_from_cameras,
    transform_cameras,
    transform_points,
    align_principle_axes,
    normalize_scene,
    compute_scene_extent,
)

__all__ = [
    # Dataset classes
    "FreeTimeParser",
    "FreeTimeDataset",
    # Initialization functions
    "load_multiframe_colmap_points",
    "load_single_frame_with_velocity",
    "find_available_colmap_frames",
    # Normalization functions
    "similarity_from_cameras",
    "transform_cameras",
    "transform_points",
    "align_principle_axes",
    "normalize_scene",
    "compute_scene_extent",
]
