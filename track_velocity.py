#!/usr/bin/env python3
"""
RoMaV2 Temporal Tracking and Velocity Estimation

This script:
1. Tracks 3D points across multiple frames using temporal 2D matching
2. Computes velocity as linear slope (averaged over all dt)
3. Uses robust color estimation (patch averaging from both views)
4. Saves results to PLY file with x, y, z, r, g, b, vx, vy, vz
"""

import sys
import os
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image as PILImage
import open3d as o3d

# Add the read_write_model to path
sys.path.insert(0, str(Path(__file__).parent))
from read_write_model import read_model, qvec2rotmat, Camera, Image

# Add RoMaV2 to path
sys.path.insert(0, str(Path(__file__).parent / "RoMaV2" / "src"))
from romav2 import RoMaV2


def get_intrinsic_matrix(camera: Camera) -> np.ndarray:
    """Build 3x3 intrinsic matrix K from COLMAP camera."""
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params[:3]
        fx = fy = f
    else:
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
    """Build 3x4 projection matrix P = K @ [R | t]."""
    K = get_intrinsic_matrix(camera)
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    Rt = np.hstack([R, t])
    P = K @ Rt
    return P


def triangulate_points_pair(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from 2D correspondences."""
    if len(pts1) == 0:
        return np.array([]).reshape(0, 3)

    pts1_T = pts1.T.astype(np.float64)
    pts2_T = pts2.T.astype(np.float64)
    points_4d = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)
    points_3d = points_4d[:3, :] / points_4d[3:4, :]
    return points_3d.T


def compute_reprojection_error(
    points_3d: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """Compute mean reprojection error."""
    if len(points_3d) == 0:
        return np.array([])

    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    proj1 = (P1 @ points_3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]

    proj2 = (P2 @ points_3d_h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]

    errors1 = np.linalg.norm(proj1 - pts1, axis=1)
    errors2 = np.linalg.norm(proj2 - pts2, axis=1)

    return (errors1 + errors2) / 2


def check_points_in_front(points_3d: np.ndarray, image: Image) -> np.ndarray:
    """Check which points are in front of the camera."""
    if len(points_3d) == 0:
        return np.array([], dtype=bool)

    R = qvec2rotmat(image.qvec)
    t = image.tvec
    points_cam = (R @ points_3d.T).T + t
    return points_cam[:, 2] > 0


def sample_patch_color(img_array: np.ndarray, x: float, y: float, patch_size: int = 5) -> np.ndarray:
    """
    Sample color from a patch around (x, y) using averaging.
    More robust than single pixel sampling.
    """
    h, w = img_array.shape[:2]
    half = patch_size // 2

    # Get patch bounds
    x_int, y_int = int(round(x)), int(round(y))
    x_min = max(0, x_int - half)
    x_max = min(w, x_int + half + 1)
    y_min = max(0, y_int - half)
    y_max = min(h, y_int + half + 1)

    # Extract patch and compute mean
    patch = img_array[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return np.array([128, 128, 128], dtype=np.uint8)

    return patch.mean(axis=(0, 1)).astype(np.uint8)


def estimate_color_robust(
    img_path_A: Path,
    img_path_B: Path,
    pts_A: np.ndarray,
    pts_B: np.ndarray,
    patch_size: int = 5
) -> np.ndarray:
    """
    Robust color estimation using:
    1. Patch averaging (reduces noise)
    2. Both views (more reliable color estimate)
    """
    with PILImage.open(img_path_A) as img:
        img_A = np.array(img.convert('RGB'))
    with PILImage.open(img_path_B) as img:
        img_B = np.array(img.convert('RGB'))

    colors = []
    for pt_A, pt_B in zip(pts_A, pts_B):
        color_A = sample_patch_color(img_A, pt_A[0], pt_A[1], patch_size)
        color_B = sample_patch_color(img_B, pt_B[0], pt_B[1], patch_size)
        # Average colors from both views
        color = ((color_A.astype(float) + color_B.astype(float)) / 2).astype(np.uint8)
        colors.append(color)

    return np.array(colors)


def extract_camera_folder_from_name(image_name: str) -> str:
    """Extract camera folder name from COLMAP image name."""
    parts = image_name.split('/')
    if len(parts) >= 2:
        return parts[0]
    return image_name


def build_camera_mapping(images: Dict[int, Image]) -> Dict[str, int]:
    """Build mapping from camera folder name to image ID."""
    mapping = {}
    for img_id, img in images.items():
        cam_folder = extract_camera_folder_from_name(img.name)
        mapping[cam_folder] = img_id
    return mapping


def get_camera_positions(
    images: Dict[int, Image],
    camera_mapping: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """Get camera positions in world coordinates for all cameras."""
    positions = {}
    for cam_folder, img_id in camera_mapping.items():
        img = images[img_id]
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
    """
    positions = get_camera_positions(images, camera_mapping)
    camera_folders = sorted(camera_mapping.keys())
    n_cameras = len(camera_folders)

    pos_matrix = np.array([positions[cam] for cam in camera_folders])

    diff = pos_matrix[:, np.newaxis, :] - pos_matrix[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    pairs_set = set()
    for i in range(n_cameras):
        sorted_indices = np.argsort(distances[i])
        neighbors = sorted_indices[1:k+1]
        for j in neighbors:
            pair = tuple(sorted([camera_folders[i], camera_folders[j]]))
            pairs_set.add(pair)

    pairs = list(pairs_set)
    print(f"K-nearest pairing: {n_cameras} cameras, k={k} -> {len(pairs)} unique pairs")
    return pairs


def get_frame_path(cam_dir: Path, camera_folder: str, frame_idx: int) -> Path:
    """Get full path to a frame image."""
    frame_name = f"{frame_idx:06d}.jpg"
    return cam_dir / camera_folder / frame_name


def warp_points_temporal(
    model: RoMaV2,
    img_path_t: Path,
    img_path_t1: Path,
    pts_t: np.ndarray,
    H: int,
    W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp points from frame t to frame t+1 using RoMaV2's dense warp.

    Returns:
        pts_t1: Warped point positions at t+1
        valid_mask: Boolean mask for successfully warped points
    """
    if len(pts_t) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=bool)

    # Get dense warp from t to t+1
    preds = model.match(str(img_path_t), str(img_path_t1))
    warp = preds["warp_AB"][0].cpu().numpy()  # H x W x 2, in [-1, 1]
    overlap = preds["overlap_AB"][0, ..., 0].cpu().numpy()  # H x W

    warp_H, warp_W = warp.shape[:2]

    # Convert pixel coordinates to normalized [-1, 1]
    pts_norm = pts_t.copy()
    pts_norm[:, 0] = (pts_t[:, 0] / W) * 2 - 1  # x
    pts_norm[:, 1] = (pts_t[:, 1] / H) * 2 - 1  # y

    # Sample warp at point locations
    pts_t1 = []
    valid = []

    for i, (px, py) in enumerate(pts_norm):
        # Convert to warp grid coordinates
        wx = int((px + 1) / 2 * (warp_W - 1))
        wy = int((py + 1) / 2 * (warp_H - 1))

        wx = np.clip(wx, 0, warp_W - 1)
        wy = np.clip(wy, 0, warp_H - 1)

        # Get warped position (in [-1, 1])
        warped_norm = warp[wy, wx]
        conf = overlap[wy, wx]

        # Convert back to pixel coordinates
        warped_px = (warped_norm[0] + 1) / 2 * W
        warped_py = (warped_norm[1] + 1) / 2 * H

        pts_t1.append([warped_px, warped_py])
        valid.append(conf > 0.3 and 0 <= warped_px < W and 0 <= warped_py < H)

    return np.array(pts_t1), np.array(valid, dtype=bool)


def track_points_temporal(
    model: RoMaV2,
    cam_dir: Path,
    cam_folder_A: str,
    cam_folder_B: str,
    frame_start: int,
    frame_end: int,
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    camera_mapping: Dict[str, int],
    num_matches: int = 1000,
    reprojection_threshold: float = 10.0,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Track 3D points for a camera pair across multiple frames with stepping.

    Args:
        step: Frame step size (e.g., 5 means process frames 0, 5, 10, ...)

    Returns:
        trajectories_3d: N x T x 3 array of 3D positions (NaN for invalid)
        colors: N x 3 array of RGB colors
        valid_frames: N x T boolean array indicating valid observations
        frame_indices: List of actual frame indices processed
    """
    img_id_A = camera_mapping[cam_folder_A]
    img_id_B = camera_mapping[cam_folder_B]
    img_A = images[img_id_A]
    img_B = images[img_id_B]
    cam_A = cameras[img_A.camera_id]
    cam_B = cameras[img_B.camera_id]

    P_A = get_projection_matrix(cam_A, img_A)
    P_B = get_projection_matrix(cam_B, img_B)

    # Generate list of frame indices to process (with step)
    frame_indices = list(range(frame_start, frame_end + 1, step))
    num_stepped_frames = len(frame_indices)

    # Get initial matches at frame_start
    img_path_A_0 = get_frame_path(cam_dir, cam_folder_A, frame_start)
    img_path_B_0 = get_frame_path(cam_dir, cam_folder_B, frame_start)

    if not img_path_A_0.exists() or not img_path_B_0.exists():
        return np.array([]).reshape(0, num_stepped_frames, 3), np.array([]).reshape(0, 3), np.array([]).reshape(0, num_stepped_frames), frame_indices

    with PILImage.open(img_path_A_0) as pil_img:
        W_A, H_A = pil_img.size
    with PILImage.open(img_path_B_0) as pil_img:
        W_B, H_B = pil_img.size

    # Get spatial matches at frame_start
    preds = model.match(str(img_path_A_0), str(img_path_B_0))
    matches, _, _, _ = model.sample(preds, num_matches)
    kpts_A, kpts_B = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    pts_A_0 = kpts_A.cpu().numpy()
    pts_B_0 = kpts_B.cpu().numpy()

    # Triangulate initial points
    pts_3d_0 = triangulate_points_pair(pts_A_0, pts_B_0, P_A, P_B)

    # Filter initial points
    errors = compute_reprojection_error(pts_3d_0, pts_A_0, pts_B_0, P_A, P_B)
    in_front_A = check_points_in_front(pts_3d_0, img_A)
    in_front_B = check_points_in_front(pts_3d_0, img_B)

    valid_init = (
        (errors < reprojection_threshold) &
        in_front_A &
        in_front_B &
        np.isfinite(pts_3d_0).all(axis=1)
    )

    pts_A_0 = pts_A_0[valid_init]
    pts_B_0 = pts_B_0[valid_init]
    pts_3d_0 = pts_3d_0[valid_init]

    N = len(pts_3d_0)
    if N == 0:
        return np.array([]).reshape(0, num_stepped_frames, 3), np.array([]).reshape(0, 3), np.array([]).reshape(0, num_stepped_frames), frame_indices

    # Get robust colors at frame_start
    colors = estimate_color_robust(img_path_A_0, img_path_B_0, pts_A_0, pts_B_0, patch_size=5)

    # Initialize trajectories for stepped frames only
    trajectories_3d = np.full((N, num_stepped_frames, 3), np.nan)
    valid_frames = np.zeros((N, num_stepped_frames), dtype=bool)

    trajectories_3d[:, 0, :] = pts_3d_0
    valid_frames[:, 0] = True

    # Track current positions
    current_pts_A = pts_A_0.copy()
    current_pts_B = pts_B_0.copy()
    current_valid = np.ones(N, dtype=bool)

    # Track through stepped frames
    for step_idx in range(1, num_stepped_frames):
        frame_curr = frame_indices[step_idx]
        frame_prev = frame_indices[step_idx - 1]

        img_path_A_prev = get_frame_path(cam_dir, cam_folder_A, frame_prev)
        img_path_A_curr = get_frame_path(cam_dir, cam_folder_A, frame_curr)
        img_path_B_prev = get_frame_path(cam_dir, cam_folder_B, frame_prev)
        img_path_B_curr = get_frame_path(cam_dir, cam_folder_B, frame_curr)

        if not all(p.exists() for p in [img_path_A_prev, img_path_A_curr, img_path_B_prev, img_path_B_curr]):
            continue

        # Warp points from prev to curr in both cameras
        warped_A, valid_A = warp_points_temporal(
            model, img_path_A_prev, img_path_A_curr,
            current_pts_A[current_valid], H_A, W_A
        )
        warped_B, valid_B = warp_points_temporal(
            model, img_path_B_prev, img_path_B_curr,
            current_pts_B[current_valid], H_B, W_B
        )

        # Both warps must be valid
        both_valid = valid_A & valid_B

        # Update current valid indices
        valid_indices = np.where(current_valid)[0]

        for local_idx, global_idx in enumerate(valid_indices):
            if both_valid[local_idx]:
                pt_A = warped_A[local_idx]
                pt_B = warped_B[local_idx]

                # Triangulate
                pt_3d = triangulate_points_pair(
                    pt_A.reshape(1, 2), pt_B.reshape(1, 2), P_A, P_B
                )[0]

                # Validate
                error = compute_reprojection_error(
                    pt_3d.reshape(1, 3), pt_A.reshape(1, 2), pt_B.reshape(1, 2), P_A, P_B
                )[0]

                if error < reprojection_threshold and np.isfinite(pt_3d).all():
                    trajectories_3d[global_idx, step_idx, :] = pt_3d
                    valid_frames[global_idx, step_idx] = True
                    current_pts_A[global_idx] = pt_A
                    current_pts_B[global_idx] = pt_B
                else:
                    current_valid[global_idx] = False
            else:
                current_valid[global_idx] = False

    return trajectories_3d, colors, valid_frames, frame_indices


def compute_velocity_from_displacements(
    trajectories_3d: np.ndarray,
    valid_frames: np.ndarray,
    frame_indices: List[int],
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity as average displacement per step interval.

    For step=5, frames 0, 5, 10, 15, ...:
    - Compute displacement from 0→5, 5→10, 10→15, ...
    - Average all displacements for each point
    - Velocity = average displacement / step (displacement per frame)

    Returns:
        velocities: N x 3 array of velocity (vx, vy, vz) per frame
        mean_positions: N x 3 array of mean 3D positions
    """
    N, T, _ = trajectories_3d.shape

    velocities = np.zeros((N, 3))
    mean_positions = np.zeros((N, 3))

    for i in range(N):
        valid_t = valid_frames[i]
        valid_indices = np.where(valid_t)[0]

        if len(valid_indices) < 2:
            # Not enough observations for velocity
            velocities[i] = [0, 0, 0]
            if len(valid_indices) >= 1:
                mean_positions[i] = trajectories_3d[i, valid_t].mean(axis=0)
            continue

        # Compute displacements between consecutive valid stepped frames
        displacements = []
        for j in range(len(valid_indices) - 1):
            idx1 = valid_indices[j]
            idx2 = valid_indices[j + 1]

            # Only use consecutive stepped frames
            if idx2 - idx1 == 1:  # consecutive in stepped index
                pos1 = trajectories_3d[i, idx1]
                pos2 = trajectories_3d[i, idx2]
                displacement = pos2 - pos1  # displacement over one step interval
                displacements.append(displacement)

        if len(displacements) > 0:
            # Average displacement per step interval, then divide by step to get per-frame velocity
            avg_displacement = np.mean(displacements, axis=0)
            velocities[i] = avg_displacement / step
        else:
            velocities[i] = [0, 0, 0]

        mean_positions[i] = trajectories_3d[i, valid_t].mean(axis=0)

    return velocities, mean_positions


def write_ply_with_velocity(
    filepath: Path,
    positions: np.ndarray,
    colors: np.ndarray,
    velocities: np.ndarray
):
    """
    Write PLY file with positions, colors, and velocity attributes.

    Format: x, y, z, red, green, blue, vx, vy, vz
    """
    N = len(positions)

    header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float vx
property float vy
property float vz
end_header
"""

    with open(filepath, 'w') as f:
        f.write(header)
        for i in range(N):
            x, y, z = positions[i]
            r, g, b = colors[i]
            vx, vy, vz = velocities[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {vx:.6f} {vy:.6f} {vz:.6f}\n")

    print(f"Saved PLY with velocity to {filepath}")


def save_point_cloud_as_ply_o3d(
    positions: np.ndarray,
    colors: np.ndarray,
    output_path: Path
) -> bool:
    """
    Save 3D points and colors as a standard PLY file using Open3D.
    (Without velocity - for visualization tools that don't support custom attributes)
    """
    print(f"Saving point cloud as PLY (Open3D): {output_path}")

    if len(positions) == 0:
        print("No points to save")
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if colors is not None and len(colors) > 0:
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    success = o3d.io.write_point_cloud(str(output_path), pcd)

    if success:
        print(f"PLY file saved: {output_path} ({len(positions)} points)")
    else:
        print(f"Failed to save PLY file: {output_path}")

    return success


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RoMaV2 Velocity Tracking")
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
        "--frame_start",
        type=int,
        default=0,
        help="Start frame index"
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=10,
        help="End frame index"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step size (e.g., 5 means process frames 0, 5, 10, ...)"
    )
    parser.add_argument(
        "--num_matches",
        type=int,
        default=500,
        help="Number of matches to track per camera pair"
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of camera pairs (default: all)"
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
        help="Output directory"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="turbo",
        choices=["turbo", "fast", "base", "precise"],
        help="RoMaV2 setting"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set torch precision
    torch.set_float32_matmul_precision("highest")

    # Read COLMAP model
    print(f"Reading COLMAP model from {args.colmap_path}")
    cameras, images, _ = read_model(args.colmap_path)
    print(f"Loaded {len(cameras)} cameras, {len(images)} images")

    # Build camera mapping
    camera_mapping = build_camera_mapping(images)
    camera_folders = sorted(camera_mapping.keys())
    print(f"Found {len(camera_folders)} cameras")

    # Generate camera pairs
    if args.k_nearest is not None:
        print(f"\nUsing K-nearest camera pairing (k={args.k_nearest})...")
        all_pairs = get_k_nearest_pairs(camera_mapping, images, k=args.k_nearest)
    else:
        all_pairs = list(combinations(camera_folders, 2))
        if args.max_pairs is not None and args.max_pairs < len(all_pairs):
            indices = np.random.choice(len(all_pairs), args.max_pairs, replace=False)
            all_pairs = [all_pairs[i] for i in indices]
        print(f"Processing {len(all_pairs)} camera pairs")

    # Initialize RoMaV2
    print(f"\nInitializing RoMaV2 with setting '{args.setting}'...")
    model = RoMaV2()
    model.apply_setting(args.setting)

    # Track all camera pairs
    all_positions = []
    all_colors = []
    all_velocities = []

    # Compute actual frame list with stepping
    frame_indices = list(range(args.frame_start, args.frame_end + 1, args.step))
    num_stepped_frames = len(frame_indices)
    print(f"\nTracking frames {args.frame_start} to {args.frame_end} with step={args.step}")
    print(f"Frames to process: {frame_indices} ({num_stepped_frames} frames)")

    for cam_A, cam_B in tqdm(all_pairs, desc="Processing camera pairs"):
        try:
            trajectories, colors, valid_frames, frame_idxs = track_points_temporal(
                model=model,
                cam_dir=Path(args.cam_dir),
                cam_folder_A=cam_A,
                cam_folder_B=cam_B,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                cameras=cameras,
                images=images,
                camera_mapping=camera_mapping,
                num_matches=args.num_matches,
                reprojection_threshold=args.reprojection_threshold,
                step=args.step
            )

            if len(trajectories) == 0:
                continue

            # Compute velocity from displacements
            velocities, mean_positions = compute_velocity_from_displacements(
                trajectories, valid_frames, frame_idxs, step=args.step
            )

            # Filter points with valid velocity (at least 2 observations)
            valid_vel = valid_frames.sum(axis=1) >= 2

            if valid_vel.sum() > 0:
                all_positions.append(mean_positions[valid_vel])
                all_colors.append(colors[valid_vel])
                all_velocities.append(velocities[valid_vel])

        except Exception as e:
            print(f"Error processing pair {cam_A}-{cam_B}: {e}")
            continue

    if not all_positions:
        print("No valid points found!")
        return

    # Concatenate all results
    positions = np.vstack(all_positions)
    colors = np.vstack(all_colors)
    velocities = np.vstack(all_velocities)

    print(f"\nTotal tracked points: {len(positions)}")
    print(f"Velocity statistics:")
    print(f"  Mean |v|: {np.linalg.norm(velocities, axis=1).mean():.4f}")
    print(f"  Max |v|: {np.linalg.norm(velocities, axis=1).max():.4f}")
    print(f"  Mean vx: {velocities[:, 0].mean():.4f}, vy: {velocities[:, 1].mean():.4f}, vz: {velocities[:, 2].mean():.4f}")

    # Create filename suffix with frame info and step
    file_suffix = f"f{args.frame_start:06d}_f{args.frame_end:06d}_s{args.step}"

    # Save PLY with position, color, and velocity
    ply_path = output_dir / f"points_velocity_{file_suffix}.ply"
    write_ply_with_velocity(ply_path, positions, colors, velocities)

    # Also save as numpy for further analysis
    npz_path = output_dir / f"velocity_data_{file_suffix}.npz"
    np.savez(
        npz_path,
        positions=positions,
        colors=colors,
        velocities=velocities,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        step=args.step,
        frame_indices=frame_indices
    )
    print(f"Saved numpy data to {npz_path}")

    print("\nDone!")
    return positions, colors, velocities


if __name__ == "__main__":
    main()
