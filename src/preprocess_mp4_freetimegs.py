#!/usr/bin/env python3
"""
mp4 -> FreeTimeGS 数据准备(抽帧 + 参考帧 COLMAP + RoMA 逐帧三角化)

你提供的原始输入是"每路相机一个 mp4".
本脚本把它转成 FreeTimeGsVanilla 可直接消费的数据结构:
- data_dir/images/<cam>/%06d.jpg
- data_dir/sparse/0 (参考帧跑一次 COLMAP,得到静态相机位姿)
- triangulation_dir/{points3d,colors}_frame%06d.npy (逐帧三角化点云)

设计原则:
- 相机位姿静态: 只在 reference_frame 跑一次 COLMAP.
- 帧范围语义: [start_frame, end_frame) (end exclusive).
- 任何相机缺帧: 直接报错,不要静默跳过(避免多视角错位).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from datasets.read_write_model import read_model


@dataclass(frozen=True)
class CameraInfo:
    """静态相机信息(来自 COLMAP sparse/0)."""

    name: str
    width: int
    height: int
    model: str
    params: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    fisheye: bool
    R: np.ndarray  # world -> cam
    t: np.ndarray  # world -> cam
    P: np.ndarray  # 3x4 world -> cam (normalized coords)


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """运行外部命令并在失败时直接抛异常."""
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _list_mp4_files(mp4_dir: Path) -> List[Path]:
    """枚举 mp4_dir 下的 mp4 文件(大小写不敏感)."""
    if not mp4_dir.exists():
        raise FileNotFoundError(f"--mp4-dir 不存在: {mp4_dir}")

    mp4_files: List[Path] = []
    for p in sorted(mp4_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".mp4":
            mp4_files.append(p)
    return mp4_files


def _extract_frames_one_camera(
    video_path: Path,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    frame_digits: int,
    image_ext: str,
) -> None:
    """
    用 OpenCV VideoCapture 抽帧.

    这里用顺序读取+跳过,比随机 seek 更稳定.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    try:
        # -----------------------------
        # 先跳过 start_frame 之前的帧
        # -----------------------------
        for _ in range(start_frame):
            ok = cap.grab()
            if not ok:
                raise RuntimeError(f"视频帧不足,无法跳到 start_frame={start_frame}: {video_path}")

        # -----------------------------
        # 写入 [start_frame, end_frame)
        # -----------------------------
        for frame_idx in range(start_frame, end_frame):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"读取帧失败,frame_idx={frame_idx}: {video_path}")

            out_path = out_dir / f"{frame_idx:0{frame_digits}d}.{image_ext}"
            ok_write = cv2.imwrite(str(out_path), frame_bgr)
            if not ok_write:
                raise RuntimeError(f"写入图片失败: {out_path}")
    finally:
        cap.release()


def _build_cv_intrinsics(camera_model: str, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    把 COLMAP 的 camera model/params 转成 OpenCV 的 K + dist.

    返回:
    - K: 3x3
    - dist: (N,) (OpenCV distortion coefficients)
    - fisheye: 是否使用 cv2.fisheye 的畸变模型
    """
    model = camera_model.upper()
    params = np.asarray(params, dtype=np.float64).copy()

    # -----------------------------
    # 1) 先构造 K
    # -----------------------------
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params.tolist()
        fx, fy = f, f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params.tolist()
    elif model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy, _k = params.tolist()
        fx, fy = f, f
    elif model in {"RADIAL", "RADIAL_FISHEYE"}:
        f, cx, cy, _k1, _k2 = params.tolist()
        fx, fy = f, f
    elif model in {"OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy = params[:4].tolist()
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy = params[:4].tolist()
    else:
        # 兜底: 按 pinhole 解释前 4 个参数,否则只能放弃畸变
        if len(params) >= 4:
            fx, fy, cx, cy = params[:4].tolist()
        else:
            raise ValueError(f"不支持的相机模型(且参数不足): {camera_model}, params={params.tolist()}")

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # -----------------------------
    # 2) 再构造 dist + fisheye 标记
    # -----------------------------
    fisheye = False
    dist = np.zeros((8,), dtype=np.float64)

    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        dist[:] = 0.0
    elif model == "SIMPLE_RADIAL":
        # COLMAP: f, cx, cy, k
        k = float(params[3])
        dist[:5] = np.array([k, 0.0, 0.0, 0.0, 0.0])
    elif model == "RADIAL":
        # COLMAP: f, cx, cy, k1, k2
        k1, k2 = float(params[3]), float(params[4])
        dist[:5] = np.array([k1, k2, 0.0, 0.0, 0.0])
    elif model == "OPENCV":
        # COLMAP: fx, fy, cx, cy, k1, k2, p1, p2
        k1, k2, p1, p2 = params[4:8].tolist()
        dist[:8] = np.array([k1, k2, p1, p2, 0.0, 0.0, 0.0, 0.0])
    elif model == "FULL_OPENCV":
        # COLMAP: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        # OpenCV 支持最多 8 个(或更多)畸变系数.这里直接透传前 8 个.
        k1, k2, p1, p2, k3, k4, k5, k6 = params[4:12].tolist()
        dist[:8] = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
    elif model == "OPENCV_FISHEYE":
        # COLMAP: fx, fy, cx, cy, k1, k2, k3, k4
        fisheye = True
        k1, k2, k3, k4 = params[4:8].tolist()
        dist[:4] = np.array([k1, k2, k3, k4])
    elif model == "SIMPLE_RADIAL_FISHEYE":
        fisheye = True
        k1 = float(params[3])
        dist[:4] = np.array([k1, 0.0, 0.0, 0.0])
    elif model == "RADIAL_FISHEYE":
        fisheye = True
        k1, k2 = float(params[3]), float(params[4])
        dist[:4] = np.array([k1, k2, 0.0, 0.0])
    else:
        # 兜底: 不做畸变
        dist[:] = 0.0

    return K, dist, fisheye


def _load_colmap_cameras(colmap_sparse0: Path) -> List[CameraInfo]:
    """
    读取 COLMAP sparse/0,得到静态相机参数与位姿.

    注意:
    - image.name 通常是 "<cam>.jpg",我们用 stem 作为相机名.
    - 这里返回按 name 排序后的列表,用于稳定选择 anchor camera.
    """
    cameras, images, _points3d = read_model(str(colmap_sparse0))
    if cameras is None or images is None:
        raise RuntimeError(f"读取 COLMAP 失败: {colmap_sparse0}")

    cam_infos: List[CameraInfo] = []
    for _image_id, im in images.items():
        cam = cameras[im.camera_id]

        cam_name = Path(im.name).stem
        R = im.qvec2rotmat().astype(np.float64)
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3, 1)

        K, dist, fisheye = _build_cv_intrinsics(cam.model, cam.params)

        # 归一化坐标下的投影矩阵: x ~ [R|t] X
        P = np.concatenate([R, t], axis=1).astype(np.float64)

        cam_infos.append(
            CameraInfo(
                name=cam_name,
                width=int(cam.width),
                height=int(cam.height),
                model=str(cam.model),
                params=np.asarray(cam.params, dtype=np.float64),
                K=K,
                dist=dist,
                fisheye=fisheye,
                R=R,
                t=t,
                P=P,
            )
        )

    cam_infos.sort(key=lambda x: x.name)
    return cam_infos


def _undistort_to_normalized(
    cam: CameraInfo,
    kpts_px: np.ndarray,
) -> np.ndarray:
    """把像素坐标去畸变并转成 normalized camera coords."""
    if kpts_px.ndim != 2 or kpts_px.shape[1] != 2:
        raise ValueError(f"kpts_px 形状不对,期望 Nx2,得到: {kpts_px.shape}")

    pts = kpts_px.astype(np.float64).reshape(-1, 1, 2)

    if cam.fisheye:
        # fisheye: dist 只用前 4 个系数
        undist = cv2.fisheye.undistortPoints(pts, cam.K, cam.dist[:4])
    else:
        # perspective: 支持 (k1,k2,p1,p2,k3,k4,k5,k6...)
        undist = cv2.undistortPoints(pts, cam.K, cam.dist)

    return undist.reshape(-1, 2).astype(np.float64)


def _voxel_dedup(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单 voxel 去重(保留每个 voxel 的第一个点).

    说明:
    - 这里的目标是"去掉明显重复点,控制规模".
    - 不是高质量重采样/重建,不要过度设计.
    """
    if len(points) == 0:
        return points, colors

    if voxel_size <= 0:
        return points, colors

    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    voxel_key = (
        voxel_idx[:, 0] * 73856093
        ^ voxel_idx[:, 1] * 19349663
        ^ voxel_idx[:, 2] * 83492791
    )
    _, unique_idx = np.unique(voxel_key, return_index=True)
    unique_idx = np.sort(unique_idx)

    return points[unique_idx], colors[unique_idx]


def _triangulate_one_pair(
    cam_a: CameraInfo,
    cam_b: CameraInfo,
    kpts_a_px: np.ndarray,
    kpts_b_px: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在已知静态相机位姿下,对一对相机的 inlier 匹配做三角化.

    返回:
    - Xw: [N, 3] world coords
    - valid_mask: [N] (用于同步过滤颜色采样点)
    """
    if len(kpts_a_px) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    # 1) 去畸变并转 normalized coords
    a_norm = _undistort_to_normalized(cam_a, kpts_a_px)
    b_norm = _undistort_to_normalized(cam_b, kpts_b_px)

    # 2) cv2.triangulatePoints 需要 2xN
    X_h = cv2.triangulatePoints(cam_a.P, cam_b.P, a_norm.T, b_norm.T)  # 4xN
    X = (X_h[:3] / (X_h[3:4] + 1e-12)).T  # Nx3

    # 3) cheirality: 两边深度都要 > 0
    X_a = (cam_a.R @ X.T) + cam_a.t  # 3xN
    X_b = (cam_b.R @ X.T) + cam_b.t
    valid = (X_a[2] > 0.0) & (X_b[2] > 0.0)
    valid = valid.reshape(-1)

    # 4) 过滤掉 NaN/Inf
    finite = np.isfinite(X).all(axis=1)
    valid = valid & finite

    return X, valid


def _sample_anchor_colors(anchor_rgb: np.ndarray, kpts_a_px: np.ndarray) -> np.ndarray:
    """从 anchor 图像按像素坐标采样 RGB."""
    H, W = anchor_rgb.shape[:2]
    xy = np.round(kpts_a_px).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, W - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, H - 1)
    colors = anchor_rgb[xy[:, 1], xy[:, 0], :3]
    return colors.astype(np.uint8)


def _ensure_symlink(src: Path, dst: Path) -> None:
    """创建 symlink(已存在则覆盖)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _prepare_colmap_reference(
    data_dir: Path,
    work_dir: Path,
    camera_names: List[str],
    reference_frame: int,
    frame_digits: int,
    image_ext: str,
    overwrite: bool,
) -> Tuple[Path, Path]:
    """
    把每路相机的 reference_frame 软链接到 work_dir/colmap_ref/images/<cam>.jpg.

    返回:
    - colmap_ref_dir
    - colmap_images_dir
    """
    colmap_ref_dir = work_dir / "colmap_ref"
    colmap_images_dir = colmap_ref_dir / "images"

    if colmap_ref_dir.exists():
        if overwrite:
            shutil.rmtree(colmap_ref_dir)
        else:
            raise FileExistsError(
                f"colmap_ref 已存在: {colmap_ref_dir}. 如需覆盖请加 --overwrite"
            )

    colmap_images_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in camera_names:
        src = data_dir / "images" / cam_name / f"{reference_frame:0{frame_digits}d}.{image_ext}"
        if not src.exists():
            raise FileNotFoundError(f"reference_frame 图片不存在: {src}")

        dst = colmap_images_dir / f"{cam_name}.{image_ext}"
        _ensure_symlink(src, dst)

    return colmap_ref_dir, colmap_images_dir


def _run_colmap_mapper(
    colmap_images_dir: Path,
    colmap_ref_dir: Path,
    camera_model: str,
) -> Path:
    """
    在 colmap_ref_dir 下运行:
    - feature_extractor
    - exhaustive_matcher
    - mapper

    返回 sparse/0 的路径.
    """
    database_path = colmap_ref_dir / "database.db"
    sparse_dir = colmap_ref_dir / "sparse"

    # 备注:
    # - apt 的 colmap 常见是无 CUDA,因此 use_gpu=0.
    _run_cmd(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(colmap_images_dir),
            "--ImageReader.camera_model",
            camera_model,
            "--SiftExtraction.use_gpu",
            "0",
        ],
        cwd=colmap_ref_dir,
    )
    _run_cmd(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "0",
        ],
        cwd=colmap_ref_dir,
    )
    _run_cmd(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(colmap_images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        cwd=colmap_ref_dir,
    )

    sparse0 = sparse_dir / "0"
    if not sparse0.exists():
        raise RuntimeError(f"COLMAP mapper 没有产出 sparse/0: {sparse0}")

    return sparse0


def _copy_sparse0_to_data_dir(sparse0: Path, data_dir: Path, overwrite: bool) -> Path:
    """把 work_dir 的 sparse/0 拷贝到 data_dir/sparse/0."""
    out_sparse0 = data_dir / "sparse" / "0"
    out_sparse0.parent.mkdir(parents=True, exist_ok=True)

    if out_sparse0.exists():
        if overwrite:
            shutil.rmtree(out_sparse0)
        else:
            raise FileExistsError(f"目标已存在: {out_sparse0}. 如需覆盖请加 --overwrite")

    shutil.copytree(sparse0, out_sparse0)
    return out_sparse0


def _triangulate_all_frames(
    data_dir: Path,
    triangulation_dir: Path,
    cam_infos: List[CameraInfo],
    start_frame: int,
    end_frame: int,
    frame_digits: int,
    image_ext: str,
    device: str,
    ransac_reproj_threshold: float,
    ransac_confidence: float,
    ransac_max_iters: int,
    voxel_size: float,
    overwrite: bool,
) -> None:
    """
    逐帧生成 points3d_frameXXXXXX.npy/colors_frameXXXXXX.npy.

    核心思路:
    - 选 anchor camera(按 name 排序第 0 个).
    - anchor 与每个其它相机做 RoMA 匹配,再 RANSAC 过滤,再三角化.
    - 合并所有 pair 的点云,做 voxel 去重.
    """
    try:
        from romatch import roma_outdoor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("romatch 未安装或导入失败. 请先 uv sync --locked") from exc

    triangulation_dir.mkdir(parents=True, exist_ok=True)

    cam_by_name: Dict[str, CameraInfo] = {c.name: c for c in cam_infos}
    cam_names = [c.name for c in cam_infos]
    if len(cam_names) < 2:
        raise ValueError("相机数量不足(至少 2 路才能三角化).")

    anchor_name = cam_names[0]
    anchor_cam = cam_by_name[anchor_name]
    other_names = cam_names[1:]

    # -----------------------------
    # 初始化 RoMA 模型(一次即可)
    # -----------------------------
    torch_device = torch.device(device)
    roma_model = roma_outdoor(device=torch_device)
    roma_model.symmetric = False

    print(f"[RoMA] device={device}, anchor={anchor_name}, others={other_names}")

    images_root = data_dir / "images"

    for frame_idx in range(start_frame, end_frame):
        out_points = triangulation_dir / f"points3d_frame{frame_idx:0{frame_digits}d}.npy"
        out_colors = triangulation_dir / f"colors_frame{frame_idx:0{frame_digits}d}.npy"

        if (out_points.exists() or out_colors.exists()) and not overwrite:
            print(f"[Triangulate] Skip existing frame {frame_idx}: {out_points.name}")
            continue

        anchor_img_path = images_root / anchor_name / f"{frame_idx:0{frame_digits}d}.{image_ext}"
        if not anchor_img_path.exists():
            raise FileNotFoundError(f"anchor 图像不存在: {anchor_img_path}")

        # 读取 anchor 图像用于颜色采样
        anchor_rgb = imageio.imread(anchor_img_path)[..., :3]

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []

        for other_name in other_names:
            other_cam = cam_by_name[other_name]
            other_img_path = images_root / other_name / f"{frame_idx:0{frame_digits}d}.{image_ext}"
            if not other_img_path.exists():
                raise FileNotFoundError(f"other 图像不存在: {other_img_path}")

            # -----------------------------
            # 1) RoMA 匹配
            # -----------------------------
            with torch.no_grad():
                warp, certainty = roma_model.match(
                    str(anchor_img_path),
                    str(other_img_path),
                    device=torch_device,
                )
                matches, _certainty = roma_model.sample(warp, certainty)

            # RoMA 输出是 [-1,1] 的归一化坐标,需要转换成像素坐标
            kpts_a, kpts_b = roma_model.to_pixel_coordinates(
                matches,
                anchor_cam.height,
                anchor_cam.width,
                other_cam.height,
                other_cam.width,
            )

            kpts_a_np = kpts_a.detach().cpu().numpy().astype(np.float32)
            kpts_b_np = kpts_b.detach().cpu().numpy().astype(np.float32)

            if len(kpts_a_np) < 8:
                # F 矩阵至少需要 8 点
                continue

            # -----------------------------
            # 2) RANSAC 过滤外点(像素域)
            # -----------------------------
            F, inlier_mask = cv2.findFundamentalMat(
                kpts_a_np,
                kpts_b_np,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
                confidence=ransac_confidence,
                maxIters=ransac_max_iters,
            )
            if F is None or inlier_mask is None:
                continue

            inlier = inlier_mask.reshape(-1).astype(bool)
            if inlier.sum() < 8:
                continue

            kpts_a_in = kpts_a_np[inlier]
            kpts_b_in = kpts_b_np[inlier]

            # -----------------------------
            # 3) 三角化 + cheirality 过滤
            # -----------------------------
            Xw, valid = _triangulate_one_pair(anchor_cam, other_cam, kpts_a_in, kpts_b_in)
            if valid.sum() == 0:
                continue

            Xw_valid = Xw[valid].astype(np.float64)
            kpts_a_valid = kpts_a_in[valid]

            # -----------------------------
            # 4) 从 anchor 采样颜色
            # -----------------------------
            colors = _sample_anchor_colors(anchor_rgb, kpts_a_valid)

            all_points.append(Xw_valid)
            all_colors.append(colors)

        if len(all_points) == 0:
            raise RuntimeError(f"frame={frame_idx} 没有生成任何三角化点. 请检查匹配/位姿/图像质量.")

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        # -----------------------------
        # voxel 去重,避免重复点爆炸
        # -----------------------------
        points, colors = _voxel_dedup(points, colors, voxel_size=voxel_size)

        # 落盘
        np.save(out_points, points.astype(np.float32))
        np.save(out_colors, colors.astype(np.uint8))

        print(f"[Triangulate] Frame {frame_idx}: {len(points):,} points saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="mp4 -> FreeTimeGS preprocess pipeline")
    parser.add_argument("--mp4-dir", type=str, required=True, help="包含多路 mp4 的目录(每路相机一个 mp4)")
    parser.add_argument("--data-dir", type=str, required=True, help="输出数据目录(含 images/ 与 sparse/0)")
    parser.add_argument("--triangulation-dir", type=str, required=True, help="输出逐帧点云目录")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧(包含)")
    parser.add_argument("--end-frame", type=int, default=61, help="结束帧(不包含)")
    parser.add_argument("--reference-frame", type=int, default=0, help="参考帧(用于 COLMAP 标定)")
    parser.add_argument("--image-ext", type=str, default="jpg", help="输出图片扩展名(默认 jpg)")
    parser.add_argument("--frame-digits", type=int, default=6, help="帧号位数(默认 6,即 %06d)")
    parser.add_argument("--keep-intermediate", action="store_true", help="保留中间 colmap_ref 工作目录")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出(谨慎)")

    # RoMA + 三角化参数
    parser.add_argument("--device", type=str, default=None, help="torch device,如 cuda,cuda:0,cpu(默认自动)")
    parser.add_argument("--ransac-reproj-threshold", type=float, default=0.2, help="RANSAC 像素阈值")
    parser.add_argument("--ransac-confidence", type=float, default=0.999999, help="RANSAC 置信度")
    parser.add_argument("--ransac-max-iters", type=int, default=10000, help="RANSAC 最大迭代")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="三角化点云 voxel 去重尺寸(米)")

    # COLMAP 参数
    parser.add_argument("--colmap-camera-model", type=str, default="OPENCV", help="COLMAP ImageReader.camera_model")

    args = parser.parse_args()

    if args.end_frame <= args.start_frame:
        raise ValueError("--end-frame 必须大于 --start-frame")

    if not (args.start_frame <= args.reference_frame < args.end_frame):
        raise ValueError("--reference-frame 必须落在 [start_frame, end_frame) 内")

    mp4_dir = Path(args.mp4_dir)
    data_dir = Path(args.data_dir)
    triangulation_dir = Path(args.triangulation_dir)

    # work_dir 推断:
    # - 默认用 triangulation_dir 的父目录作为 work_dir,这样 run_mp4_pipeline.sh 更好组织目录结构.
    work_dir = triangulation_dir.parent

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Step A: mp4 -> images/<cam>/%06d.jpg
    # =========================================================================
    mp4_files = _list_mp4_files(mp4_dir)
    if len(mp4_files) == 0:
        raise FileNotFoundError(f"--mp4-dir 下未找到 mp4: {mp4_dir}")

    camera_names = [p.stem for p in mp4_files]
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Extract] Cameras: {camera_names}")
    for mp4_path in mp4_files:
        cam_name = mp4_path.stem
        out_dir = images_dir / cam_name

        # 说明:
        # - overwrite: 先清空再写,避免残留旧帧.
        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)

        print(f"[Extract] {cam_name}: {mp4_path.name} -> {out_dir}")
        _extract_frames_one_camera(
            video_path=mp4_path,
            out_dir=out_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_digits=args.frame_digits,
            image_ext=args.image_ext,
        )

    # =========================================================================
    # Step B: 参考帧跑一次 COLMAP -> data_dir/sparse/0
    # =========================================================================
    colmap_ref_dir, colmap_images_dir = _prepare_colmap_reference(
        data_dir=data_dir,
        work_dir=work_dir,
        camera_names=camera_names,
        reference_frame=args.reference_frame,
        frame_digits=args.frame_digits,
        image_ext=args.image_ext,
        overwrite=args.overwrite,
    )

    sparse0 = _run_colmap_mapper(
        colmap_images_dir=colmap_images_dir,
        colmap_ref_dir=colmap_ref_dir,
        camera_model=args.colmap_camera_model,
    )

    data_sparse0 = _copy_sparse0_to_data_dir(
        sparse0=sparse0,
        data_dir=data_dir,
        overwrite=args.overwrite,
    )

    print(f"[COLMAP] sparse/0 ready: {data_sparse0}")

    if not args.keep_intermediate:
        shutil.rmtree(colmap_ref_dir)
        print(f"[COLMAP] cleaned intermediate: {colmap_ref_dir}")

    # =========================================================================
    # Step C: RoMA 逐帧匹配 + 三角化 -> points3d_frameXXXXXX.npy
    # =========================================================================
    cam_infos = _load_colmap_cameras(data_sparse0)

    _triangulate_all_frames(
        data_dir=data_dir,
        triangulation_dir=triangulation_dir,
        cam_infos=cam_infos,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_digits=args.frame_digits,
        image_ext=args.image_ext,
        device=device,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
        ransac_confidence=args.ransac_confidence,
        ransac_max_iters=args.ransac_max_iters,
        voxel_size=args.voxel_size,
        overwrite=args.overwrite,
    )

    print("\n[Done] preprocess complete.")
    print(f"  data_dir: {data_dir}")
    print(f"  triangulation_dir: {triangulation_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
