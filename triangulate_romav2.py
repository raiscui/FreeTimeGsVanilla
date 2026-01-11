#!/usr/bin/env python3
"""
RoMaV2 Multi-View Triangulation Script

This script:
1. Reads COLMAP camera poses from bin files
2. For each camera pair at a given time frame, finds matches using RoMaV2
3. Triangulates 3D points using OpenCV
4. Writes results back to COLMAP binary format
5. Visualizes results with Plotly including camera frustums
"""

import sys
import os
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
import torch
import plotly.graph_objects as go
from tqdm import tqdm
from PIL import Image as PILImage
import open3d as o3d

# Add the read_write_model to path
sys.path.insert(0, str(Path(__file__).parent))
from read_write_model import (
    read_model, write_model, qvec2rotmat,
    Camera, Image, Point3D
)

# Add RoMaV2 to path
sys.path.insert(0, str(Path(__file__).parent / "RoMaV2" / "src"))
from romav2 import RoMaV2


def get_intrinsic_matrix(camera: Camera) -> np.ndarray:
    """
    Build 3x3 intrinsic matrix K from COLMAP camera.

    For PINHOLE model: params = [fx, fy, cx, cy]
    For SIMPLE_PINHOLE model: params = [f, cx, cy]
    """
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params[:3]
        fx = fy = f
    else:
        # Fallback: assume first params are focal length and principal point
        if len(camera.params) >= 4:
            fx, fy, cx, cy = camera.params[:4]
        else:
            fx = fy = camera.params[0]
            cx, cy = camera.params[1], camera.params[2] if len(camera.params) > 2 else camera.width / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def get_projection_matrix(camera: Camera, image: Image) -> np.ndarray:
    """
    Build 3x4 projection matrix P = K @ [R | t] from COLMAP camera and image.

    COLMAP stores: P_camera = R @ P_world + t
    So the extrinsic [R | t] can be directly used with OpenCV.
    """
    K = get_intrinsic_matrix(camera)
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)

    # Projection matrix P = K @ [R | t]
    Rt = np.hstack([R, t])
    P = K @ Rt
    return P


def get_camera_center(image: Image) -> np.ndarray:
    """
    Get camera center in world coordinates.

    From COLMAP: P_camera = R @ P_world + t
    Camera center is where P_camera = 0, so: C = -R.T @ t
    """
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    C = -R.T @ t
    return C.flatten()


def triangulate_points_pair(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    Triangulate 3D points from 2D correspondences using OpenCV.

    Args:
        pts1: Nx2 array of 2D points in image 1
        pts2: Nx2 array of 2D points in image 2
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2

    Returns:
        Nx3 array of 3D points in world coordinates
    """
    if len(pts1) == 0:
        return np.array([]).reshape(0, 3)

    # OpenCV expects 2xN arrays
    pts1_T = pts1.T.astype(np.float64)
    pts2_T = pts2.T.astype(np.float64)

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)

    # Convert from homogeneous to 3D
    points_3d = points_4d[:3, :] / points_4d[3:4, :]

    return points_3d.T


def compute_reprojection_errors(
    points_3d: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reprojection errors for triangulated points.

    Returns:
        errors1: reprojection errors in image 1
        errors2: reprojection errors in image 2
        mean_errors: mean of errors1 and errors2
    """
    if len(points_3d) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert to homogeneous
    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    # Reproject to image 1
    proj1 = (P1 @ points_3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]

    # Reproject to image 2
    proj2 = (P2 @ points_3d_h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]

    # Compute errors
    errors1 = np.linalg.norm(proj1 - pts1, axis=1)
    errors2 = np.linalg.norm(proj2 - pts2, axis=1)
    mean_errors = (errors1 + errors2) / 2

    return errors1, errors2, mean_errors


def check_points_in_front(
    points_3d: np.ndarray,
    image: Image
) -> np.ndarray:
    """Check which points are in front of the camera (positive depth)."""
    if len(points_3d) == 0:
        return np.array([], dtype=bool)

    R = qvec2rotmat(image.qvec)
    t = image.tvec

    # Transform to camera coordinates: P_cam = R @ P_world + t
    points_cam = (R @ points_3d.T).T + t

    # Points should have positive Z (in front of camera)
    return points_cam[:, 2] > 0


def sample_pixel_colors_batch(
    img_array: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    Sample RGB colors at multiple points from a pre-loaded image array.

    Args:
        img_array: HxWx3 numpy array (already loaded)
        points: Nx2 array of (x, y) coordinates

    Returns:
        Nx3 array of RGB colors
    """
    h, w = img_array.shape[:2]
    n_points = len(points)
    colors = np.zeros((n_points, 3), dtype=np.uint8)

    for i, (x, y) in enumerate(points):
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        colors[i] = img_array[yi, xi]

    return colors


def load_image_array(img_path: Path) -> Tuple[np.ndarray, int, int]:
    """Load image and return array with dimensions."""
    with PILImage.open(img_path) as img:
        img_array = np.array(img.convert('RGB'))
        h, w = img_array.shape[:2]
    return img_array, w, h


def extract_camera_folder_from_name(image_name: str) -> str:
    """
    Extract camera folder name from COLMAP image name.
    Example: "cam_01/000000.jpg" -> "cam_01"
    """
    parts = image_name.split('/')
    if len(parts) >= 2:
        return parts[0]
    return image_name


def build_camera_mapping(images: Dict[int, Image]) -> Dict[str, int]:
    """
    Build mapping from camera folder name to image ID.
    Returns: {camera_folder: image_id}
    """
    mapping = {}
    for img_id, img in images.items():
        cam_folder = extract_camera_folder_from_name(img.name)
        mapping[cam_folder] = img_id
    return mapping


def get_camera_positions(
    images: Dict[int, Image],
    camera_mapping: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """
    Get camera positions in world coordinates for all cameras.

    Returns: {camera_folder: position_xyz}
    """
    positions = {}
    for cam_folder, img_id in camera_mapping.items():
        img = images[img_id]
        # Camera center: C = -R.T @ t
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)
        C = -R.T @ t
        positions[cam_folder] = C.flatten()
    return positions


def get_k_nearest_pairs(
    camera_mapping: Dict[str, int],
    images: Dict[int, Image],
    k: int = 10
) -> List[Tuple[str, str]]:
    """
    Get camera pairs based on K-nearest neighbors by camera position.

    For each camera, finds the K nearest cameras and creates pairs.
    Pairs are deduplicated (only unique unordered pairs returned).

    Args:
        camera_mapping: Mapping from camera folder to image ID
        images: COLMAP images dict
        k: Number of nearest neighbors per camera

    Returns:
        List of (cam_folder_A, cam_folder_B) pairs
    """
    # Get camera positions
    positions = get_camera_positions(images, camera_mapping)
    camera_folders = sorted(camera_mapping.keys())
    n_cameras = len(camera_folders)

    # Build position matrix
    pos_matrix = np.array([positions[cam] for cam in camera_folders])

    # Compute pairwise distances
    # dist[i, j] = ||pos[i] - pos[j]||
    diff = pos_matrix[:, np.newaxis, :] - pos_matrix[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # For each camera, find K nearest (excluding self)
    pairs_set = set()

    for i in range(n_cameras):
        # Get indices sorted by distance (excluding self)
        sorted_indices = np.argsort(distances[i])
        neighbors = sorted_indices[1:k+1]  # Skip self (index 0 after sort if distance is 0)

        for j in neighbors:
            # Create ordered pair to avoid duplicates
            pair = tuple(sorted([camera_folders[i], camera_folders[j]]))
            pairs_set.add(pair)

    pairs = list(pairs_set)
    print(f"K-nearest pairing: {n_cameras} cameras, k={k} -> {len(pairs)} unique pairs")

    return pairs


def get_frame_path(cam_dir: Path, camera_folder: str, frame_idx: int) -> Path:
    """Get full path to a frame image."""
    frame_name = f"{frame_idx:06d}.jpg"
    return cam_dir / camera_folder / frame_name


def create_camera_frustum(
    camera_center: np.ndarray,
    R: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    scale: float = 0.1,
    color: str = 'blue'
) -> List[go.Scatter3d]:
    """Create a camera frustum visualization for Plotly."""
    K_inv = np.linalg.inv(K)

    # Image corners in pixel coordinates
    corners_px = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T

    # Convert to normalized camera coordinates (rays)
    rays = K_inv @ corners_px
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
    rays = rays * scale

    # Transform to world coordinates
    corners_world = camera_center.reshape(3, 1) + R.T @ rays

    traces = []

    # Camera center point
    traces.append(go.Scatter3d(
        x=[camera_center[0]],
        y=[camera_center[1]],
        z=[camera_center[2]],
        mode='markers',
        marker=dict(size=5, color=color),
        showlegend=False
    ))

    # Lines from center to corners
    for i in range(4):
        traces.append(go.Scatter3d(
            x=[camera_center[0], corners_world[0, i]],
            y=[camera_center[1], corners_world[1, i]],
            z=[camera_center[2], corners_world[2, i]],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))

    # Rectangle connecting corners
    corner_order = [0, 1, 2, 3, 0]
    traces.append(go.Scatter3d(
        x=[corners_world[0, i] for i in corner_order],
        y=[corners_world[1, i] for i in corner_order],
        z=[corners_world[2, i] for i in corner_order],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    ))

    return traces


def visualize_with_plotly(
    points_3d: np.ndarray,
    point_colors: np.ndarray,
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    camera_mapping: Dict[str, int],
    output_path: Optional[str] = None,
    frustum_scale: float = 0.3,
    point_size: float = 2
) -> go.Figure:
    """Create a Plotly 3D visualization with camera frustums and triangulated points."""
    fig = go.Figure()

    # Add triangulated points
    if len(points_3d) > 0:
        # Subsample if too many points
        max_points = 50000
        if len(points_3d) > max_points:
            indices = np.random.choice(len(points_3d), max_points, replace=False)
            points_display = points_3d[indices]
            colors_display = point_colors[indices] if len(point_colors) > 0 else None
        else:
            points_display = points_3d
            colors_display = point_colors if len(point_colors) > 0 else None

        # Convert colors to plotly format
        if colors_display is not None and len(colors_display) > 0:
            color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors_display]
        else:
            color_strings = 'red'

        fig.add_trace(go.Scatter3d(
            x=points_display[:, 0],
            y=points_display[:, 1],
            z=points_display[:, 2],
            mode='markers',
            marker=dict(size=point_size, color=color_strings, opacity=0.8),
            name='Triangulated Points'
        ))

    # Add camera frustums
    colors = [
        'blue', 'green', 'orange', 'purple', 'cyan', 'magenta',
        'yellow', 'pink', 'brown', 'gray'
    ]

    for idx, (cam_folder, img_id) in enumerate(camera_mapping.items()):
        img = images[img_id]
        cam = cameras[img.camera_id]

        camera_center = get_camera_center(img)
        R = qvec2rotmat(img.qvec)
        K = get_intrinsic_matrix(cam)

        color = colors[idx % len(colors)]

        frustum_traces = create_camera_frustum(
            camera_center, R, K, cam.width, cam.height,
            scale=frustum_scale, color=color
        )

        for trace in frustum_traces:
            fig.add_trace(trace)

        # Add camera label
        fig.add_trace(go.Scatter3d(
            x=[camera_center[0]],
            y=[camera_center[1]],
            z=[camera_center[2]],
            mode='text',
            text=[cam_folder],
            textposition='top center',
            showlegend=False
        ))

    fig.update_layout(
        title='Multi-View Triangulation Result',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1400,
        height=800
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved visualization to {output_path}")

    return fig


def visualize_4d_with_slider(
    frame_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    camera_mapping: Dict[str, int],
    output_path: Optional[str] = None,
    frustum_scale: float = 0.3,
    point_size: float = 2,
    max_points_per_frame: int = 30000
) -> go.Figure:
    """
    Create a 4D Plotly visualization with time slider.

    Args:
        frame_data: Dict mapping frame_idx -> (points_3d, colors)
        cameras: COLMAP cameras
        images: COLMAP images
        camera_mapping: camera folder to image ID mapping
        output_path: path to save HTML
        frustum_scale: size of camera frustums
        point_size: size of points
        max_points_per_frame: max points to display per frame
    """
    frame_indices = sorted(frame_data.keys())

    if len(frame_indices) == 0:
        print("No frame data to visualize")
        return None

    # Create figure with frames
    fig = go.Figure()

    # Get camera frustum traces (static across all frames)
    camera_traces = []
    cam_colors = [
        'blue', 'green', 'orange', 'purple', 'cyan', 'magenta',
        'yellow', 'pink', 'brown', 'gray'
    ]

    for idx, (cam_folder, img_id) in enumerate(camera_mapping.items()):
        img = images[img_id]
        cam = cameras[img.camera_id]
        camera_center = get_camera_center(img)
        R = qvec2rotmat(img.qvec)
        K = get_intrinsic_matrix(cam)
        color = cam_colors[idx % len(cam_colors)]

        frustum_traces = create_camera_frustum(
            camera_center, R, K, cam.width, cam.height,
            scale=frustum_scale, color=color
        )
        camera_traces.extend(frustum_traces)

    # Add camera traces (visible in all frames)
    for trace in camera_traces:
        fig.add_trace(trace)

    n_camera_traces = len(camera_traces)

    # Create frames for animation
    frames = []

    for frame_idx in frame_indices:
        points_3d, colors = frame_data[frame_idx]

        # Subsample if needed
        if len(points_3d) > max_points_per_frame:
            indices = np.random.choice(len(points_3d), max_points_per_frame, replace=False)
            points_display = points_3d[indices]
            colors_display = colors[indices] if len(colors) > 0 else None
        else:
            points_display = points_3d
            colors_display = colors if len(colors) > 0 else None

        # Convert colors to plotly format
        if colors_display is not None and len(colors_display) > 0:
            color_strings = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_display]
        else:
            color_strings = 'red'

        # Create point cloud trace for this frame
        point_trace = go.Scatter3d(
            x=points_display[:, 0],
            y=points_display[:, 1],
            z=points_display[:, 2],
            mode='markers',
            marker=dict(size=point_size, color=color_strings, opacity=0.8),
            name=f'Frame {frame_idx}'
        )

        frames.append(go.Frame(
            data=[point_trace],
            name=str(frame_idx),
            traces=[n_camera_traces]  # Update only the point cloud trace
        ))

    # Add initial point cloud (first frame)
    first_frame = frame_indices[0]
    points_3d, colors = frame_data[first_frame]
    if len(points_3d) > max_points_per_frame:
        indices = np.random.choice(len(points_3d), max_points_per_frame, replace=False)
        points_display = points_3d[indices]
        colors_display = colors[indices] if len(colors) > 0 else None
    else:
        points_display = points_3d
        colors_display = colors if len(colors) > 0 else None

    if colors_display is not None and len(colors_display) > 0:
        color_strings = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_display]
    else:
        color_strings = 'red'

    fig.add_trace(go.Scatter3d(
        x=points_display[:, 0],
        y=points_display[:, 1],
        z=points_display[:, 2],
        mode='markers',
        marker=dict(size=point_size, color=color_strings, opacity=0.8),
        name=f'Frame {first_frame}'
    ))

    fig.frames = frames

    # Create slider
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Frame: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.05,
        'y': 0,
        'steps': [
            {
                'args': [[str(frame_idx)], {
                    'frame': {'duration': 100, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 100}
                }],
                'label': str(frame_idx),
                'method': 'animate'
            }
            for frame_idx in frame_indices
        ]
    }]

    # Add play/pause buttons
    updatemenus = [{
        'buttons': [
            {
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 100}
                }],
                'label': '▶ Play',
                'method': 'animate'
            },
            {
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': '⏸ Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 70},
        'showactive': False,
        'type': 'buttons',
        'x': 0.05,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]

    fig.update_layout(
        title=f'4D Point Cloud Visualization ({len(frame_indices)} frames)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1400,
        height=900,
        sliders=sliders,
        updatemenus=updatemenus
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved 4D visualization to {output_path}")

    return fig


def save_point_cloud_as_ply(
    points_3d: np.ndarray,
    colors: np.ndarray,
    output_path: Path
) -> bool:
    """
    Save 3D points and colors as a PLY file using Open3D.

    Args:
        points_3d: numpy array of shape (N, 3) containing 3D points
        colors: numpy array of shape (N, 3) containing RGB colors in range [0, 255]
        output_path: path where to save the PLY file

    Returns:
        True if successful, False otherwise
    """
    print(f"Saving point cloud as PLY: {output_path}")

    if len(points_3d) == 0:
        print("No points to save")
        return False

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set points
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Set colors (normalize from [0, 255] to [0, 1])
    if colors is not None and len(colors) > 0:
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        print(f"Added {len(colors)} color values")

    # Save as PLY file
    success = o3d.io.write_point_cloud(str(output_path), pcd)

    if success:
        print(f"PLY file saved successfully: {output_path}")
        print(f"Points: {len(points_3d)}")
    else:
        print(f"Failed to save PLY file: {output_path}")

    return success


def match_and_triangulate_full(
    model: RoMaV2,
    cam_dir: Path,
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    camera_mapping: Dict[str, int],
    frame_idx: int = 0,
    num_matches: int = 2000,
    max_pairs: Optional[int] = None,
    camera_pairs: Optional[List[Tuple[str, str]]] = None,
    reprojection_threshold: float = 10.0
) -> Tuple[Dict[int, Image], Dict[int, Point3D], np.ndarray, np.ndarray]:
    """
    Main function to match images, triangulate points, and build COLMAP structures.

    Args:
        camera_pairs: Optional list of (cam_folder_A, cam_folder_B) pairs to process.
                     If None, uses all pairs or max_pairs random pairs.

    Returns:
        new_images: Updated images dict with 2D points and point3D_ids
        points3D: Dict of Point3D namedtuples
        all_points_3d: Nx3 array of 3D points
        all_colors: Nx3 array of RGB colors
    """
    if camera_pairs is not None:
        # Use provided camera pairs (e.g., from K-nearest)
        all_pairs = camera_pairs
    else:
        # Generate all possible pairs
        camera_folders = sorted(camera_mapping.keys())
        all_pairs = list(combinations(camera_folders, 2))

        if max_pairs is not None and max_pairs < len(all_pairs):
            indices = np.random.choice(len(all_pairs), max_pairs, replace=False)
            all_pairs = [all_pairs[i] for i in indices]

    print(f"Processing {len(all_pairs)} camera pairs for frame {frame_idx}")

    # Storage for 2D points per image
    # Key: image_id, Value: list of (x, y) coordinates
    image_2d_points: Dict[int, List[Tuple[float, float]]] = {
        img_id: [] for img_id in images.keys()
    }
    # Key: image_id, Value: list of point3D_ids
    image_point3d_ids: Dict[int, List[int]] = {
        img_id: [] for img_id in images.keys()
    }

    # Storage for 3D points
    all_points_3d = []
    all_colors = []
    all_errors = []
    all_tracks = []  # Each track: [(img_id, pt2d_idx), ...]

    point_id = 1  # COLMAP point IDs start from 1

    for cam_folder_A, cam_folder_B in tqdm(all_pairs, desc="Matching pairs"):
        # Get image paths
        img_path_A = get_frame_path(cam_dir, cam_folder_A, frame_idx)
        img_path_B = get_frame_path(cam_dir, cam_folder_B, frame_idx)

        if not img_path_A.exists() or not img_path_B.exists():
            print(f"Warning: Missing images for {cam_folder_A} or {cam_folder_B}")
            continue

        # Get COLMAP data
        img_id_A = camera_mapping[cam_folder_A]
        img_id_B = camera_mapping[cam_folder_B]
        img_A = images[img_id_A]
        img_B = images[img_id_B]
        cam_A = cameras[img_A.camera_id]
        cam_B = cameras[img_B.camera_id]

        try:
            # Load images ONCE (for dimensions and colors)
            img_array_A, W_A, H_A = load_image_array(img_path_A)
            img_array_B, W_B, H_B = load_image_array(img_path_B)

            # Match with RoMaV2
            preds = model.match(str(img_path_A), str(img_path_B))

            # Sample matches
            matches, overlaps, precision_AB, precision_BA = model.sample(preds, num_matches)

            # Convert to pixel coordinates
            kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

            # Convert to numpy
            pts_A = kptsA.cpu().numpy()
            pts_B = kptsB.cpu().numpy()

            if len(pts_A) < 10:
                continue

            # Get projection matrices
            P_A = get_projection_matrix(cam_A, img_A)
            P_B = get_projection_matrix(cam_B, img_B)

            # Triangulate
            pts_3d = triangulate_points_pair(pts_A, pts_B, P_A, P_B)

            if len(pts_3d) == 0:
                continue

            # Compute reprojection errors
            errors_A, errors_B, mean_errors = compute_reprojection_errors(
                pts_3d, pts_A, pts_B, P_A, P_B
            )

            # Check points in front of both cameras
            in_front_A = check_points_in_front(pts_3d, img_A)
            in_front_B = check_points_in_front(pts_3d, img_B)

            # Filter by reprojection error and visibility
            valid_mask = (
                (mean_errors < reprojection_threshold) &
                in_front_A &
                in_front_B &
                np.isfinite(pts_3d).all(axis=1)
            )

            pts_A = pts_A[valid_mask]
            pts_B = pts_B[valid_mask]
            pts_3d = pts_3d[valid_mask]
            mean_errors = mean_errors[valid_mask]

            if len(pts_3d) == 0:
                continue

            # Sample colors in BATCH (no per-point image loading!)
            colors_A = sample_pixel_colors_batch(img_array_A, pts_A)
            colors_B = sample_pixel_colors_batch(img_array_B, pts_B)
            # Average colors from both views
            point_colors = ((colors_A.astype(np.float32) + colors_B.astype(np.float32)) / 2).astype(np.uint8)

            # Build track data (vectorized)
            n_points = len(pts_3d)
            base_idx_A = len(image_2d_points[img_id_A])
            base_idx_B = len(image_2d_points[img_id_B])

            # Extend 2D points lists
            image_2d_points[img_id_A].extend([(pts_A[i, 0], pts_A[i, 1]) for i in range(n_points)])
            image_2d_points[img_id_B].extend([(pts_B[i, 0], pts_B[i, 1]) for i in range(n_points)])

            # Extend point3D_id references
            new_point_ids = list(range(point_id, point_id + n_points))
            image_point3d_ids[img_id_A].extend(new_point_ids)
            image_point3d_ids[img_id_B].extend(new_point_ids)

            # Store 3D point data
            all_points_3d.extend(pts_3d.tolist())
            all_colors.extend(point_colors.tolist())
            all_errors.extend(mean_errors.tolist())

            # Build tracks
            for i in range(n_points):
                all_tracks.append([(img_id_A, base_idx_A + i), (img_id_B, base_idx_B + i)])

            point_id += n_points

        except Exception as e:
            print(f"Error processing pair {cam_folder_A}-{cam_folder_B}: {e}")
            continue

    print(f"Total triangulated points: {len(all_points_3d)}")

    # Build new Images with updated 2D points
    new_images = {}
    for img_id, img in images.items():
        xys_list = image_2d_points[img_id]
        p3d_ids_list = image_point3d_ids[img_id]

        if xys_list:
            xys = np.array(xys_list, dtype=np.float64)
            point3D_ids = np.array(p3d_ids_list, dtype=np.int64)
        else:
            xys = np.zeros((0, 2), dtype=np.float64)
            point3D_ids = np.array([], dtype=np.int64)

        new_images[img_id] = Image(
            id=img.id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,
            name=img.name,
            xys=xys,
            point3D_ids=point3D_ids
        )

    # Build Point3D dict
    points3D = {}
    for i, (xyz, rgb, error, track) in enumerate(
        zip(all_points_3d, all_colors, all_errors, all_tracks)
    ):
        pt_id = i + 1  # IDs start from 1
        image_ids = np.array([t[0] for t in track], dtype=np.int32)
        point2D_idxs = np.array([t[1] for t in track], dtype=np.int32)

        points3D[pt_id] = Point3D(
            id=pt_id,
            xyz=np.array(xyz, dtype=np.float64),
            rgb=np.array(rgb, dtype=np.uint8),
            error=float(error),
            image_ids=image_ids,
            point2D_idxs=point2D_idxs
        )

    # Convert to numpy arrays for return
    all_points_3d = np.array(all_points_3d) if all_points_3d else np.zeros((0, 3))
    all_colors = np.array(all_colors) if all_colors else np.zeros((0, 3), dtype=np.uint8)

    return new_images, points3D, all_points_3d, all_colors


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RoMaV2 Multi-View Triangulation")
    parser.add_argument(
        "--colmap_path",
        type=str,
        default="/data/shared/elaheh/4D_demo/elaheh_tech/undistorted/sparse/0",
        help="Path to COLMAP sparse reconstruction folder"
    )
    parser.add_argument(
        "--cam_dir",
        type=str,
        default="/data/shared/elaheh/4D_demo/elaheh_tech/undistorted/images",
        help="Path to camera images directory"
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=None,
        help="Single frame index to process (use --frame_start/--frame_end for multiple)"
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="Start frame index (for processing multiple frames)"
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="End frame index (for processing multiple frames)"
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=60,
        help="Frame step size for dense map generation (default 60 for ~5 COLMAP frames per 300-frame video)"
    )
    parser.add_argument(
        "--num_matches",
        type=int,
        default=2000,
        help="Number of matches to sample per pair"
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of camera pairs to process (default: all)"
    )
    parser.add_argument(
        "--k_nearest",
        type=int,
        default=None,
        help="Use K-nearest camera neighbors instead of all pairs (e.g., 10)"
    )
    parser.add_argument(
        "--reprojection_threshold",
        type=float,
        default=10.0,
        help="Maximum reprojection error in pixels"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/shared/elaheh/test_romav2",
        help="Output directory for results"
    )
    parser.add_argument(
        "--frustum_scale",
        type=float,
        default=0.3,
        help="Scale of camera frustums in visualization"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="fast",
        choices=["turbo", "fast", "base", "precise"],
        help="RoMaV2 setting (turbo/fast/base/precise)"
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (faster startup, slightly slower per-match)"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set torch precision for RoMaV2
    torch.set_float32_matmul_precision("highest")

    # Read COLMAP model
    print(f"Reading COLMAP model from {args.colmap_path}")
    cameras, images, points3D_orig = read_model(args.colmap_path)
    print(f"Loaded {len(cameras)} cameras, {len(images)} images")
    print(f"Original points3D: {len(points3D_orig)}")

    # Build camera mapping
    camera_mapping = build_camera_mapping(images)
    print(f"Found {len(camera_mapping)} camera folders:")
    for cam in sorted(camera_mapping.keys())[:10]:
        print(f"  - {cam}")
    if len(camera_mapping) > 10:
        print(f"  ... and {len(camera_mapping) - 10} more")

    # Initialize RoMaV2
    print(f"\nInitializing RoMaV2 with setting '{args.setting}'...")
    compile_model = not args.no_compile
    cfg = RoMaV2.Cfg(compile=compile_model, setting=args.setting)
    model = RoMaV2(cfg=cfg)

    # Determine camera pairs (computed ONCE, reused for all frames)
    if args.k_nearest is not None:
        print(f"\nUsing K-nearest camera pairing (k={args.k_nearest})...")
        camera_pairs = get_k_nearest_pairs(camera_mapping, images, k=args.k_nearest)
    else:
        camera_pairs = None  # Will use all pairs or max_pairs

    # Determine frames to process
    if args.frame_idx is not None:
        # Single frame mode
        frame_indices = [args.frame_idx]
    else:
        # Multi-frame mode
        frame_end = args.frame_end if args.frame_end is not None else args.frame_start
        frame_indices = list(range(args.frame_start, frame_end + 1, args.frame_step))

    print(f"\nFrames to process: {frame_indices}")
    print(f"Camera pairs: {len(camera_pairs) if camera_pairs else 'all'} (computed once, reused for all frames)")

    # Collect frame data for 4D visualization
    all_frame_data = {}

    # Process each frame
    for frame_idx in frame_indices:
        print(f"\n{'='*60}")
        print(f"Processing frame {frame_idx}...")
        print(f"{'='*60}")

        # Create COLMAP output subdirectory for this frame
        # Output to sparse/frame_XXXXXX format for FreeTime_dataset compatibility
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        colmap_output_dir = sparse_dir / f"frame_{frame_idx:06d}"
        colmap_output_dir.mkdir(parents=True, exist_ok=True)

        new_images, points3D, points_3d_array, colors_array = match_and_triangulate_full(
            model=model,
            cam_dir=Path(args.cam_dir),
            cameras=cameras,
            images=images,
            camera_mapping=camera_mapping,
            frame_idx=frame_idx,
            num_matches=args.num_matches,
            max_pairs=args.max_pairs,
            camera_pairs=camera_pairs,
            reprojection_threshold=args.reprojection_threshold
        )

        # Store for 4D visualization
        all_frame_data[frame_idx] = (points_3d_array, colors_array)

        # Write COLMAP model
        print(f"\nWriting COLMAP model to {colmap_output_dir}")
        write_model(cameras, new_images, points3D, str(colmap_output_dir), ext=".bin")
        print(f"  cameras.bin: {len(cameras)} cameras")
        print(f"  images.bin: {len(new_images)} images")
        print(f"  points3D.bin: {len(points3D)} points")

        # Compute statistics
        if points3D:
            track_lengths = [len(pt.image_ids) for pt in points3D.values()]
            mean_track_length = np.mean(track_lengths)
            print(f"  Mean track length: {mean_track_length:.2f}")

        # Save 3D points to numpy file
        points_npy_path = output_dir / f"points3d_frame{frame_idx:06d}.npy"
        np.save(points_npy_path, points_3d_array)
        print(f"\nSaved 3D points to {points_npy_path}")

        # Save colors
        colors_npy_path = output_dir / f"colors_frame{frame_idx:06d}.npy"
        np.save(colors_npy_path, colors_array)
        print(f"Saved colors to {colors_npy_path}")

        # Save PLY file
        ply_path = output_dir / f"points_frame{frame_idx:06d}.ply"
        save_point_cloud_as_ply(points_3d_array, colors_array, ply_path)

        print(f"\nFrame {frame_idx} done! Points: {len(points3D)}")

    # Create 4D visualization with time slider (if multiple frames)
    if len(frame_indices) > 1:
        print(f"\n{'='*60}")
        print("Creating 4D visualization with time slider...")
        print(f"{'='*60}")
        html_4d_path = output_dir / "visualization_4d.html"
        visualize_4d_with_slider(
            frame_data=all_frame_data,
            cameras=cameras,
            images=images,
            camera_mapping=camera_mapping,
            output_path=str(html_4d_path),
            frustum_scale=args.frustum_scale
        )
    else:
        # Single frame - create regular visualization
        frame_idx = frame_indices[0]
        print(f"\nCreating visualization...")
        html_path = output_dir / f"triangulation_frame{frame_idx:06d}.html"
        points_3d_array, colors_array = all_frame_data[frame_idx]
        visualize_with_plotly(
            points_3d=points_3d_array,
            point_colors=colors_array,
            cameras=cameras,
            images=new_images,
            camera_mapping=camera_mapping,
            output_path=str(html_path),
            frustum_scale=args.frustum_scale
        )

    print(f"\n{'='*60}")
    print(f"All done! Processed {len(frame_indices)} frames.")
    print(f"Output directory: {output_dir}")
    if len(frame_indices) > 1:
        print(f"4D visualization: {output_dir / 'visualization_4d.html'}")
    print(f"{'='*60}")

    return cameras, new_images, points3D


if __name__ == "__main__":
    main()
