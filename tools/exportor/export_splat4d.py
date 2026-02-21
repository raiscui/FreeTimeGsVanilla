#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 FreeTimeGsVanilla 的 checkpoint(.pt)导出 `.splat4d`.

为什么这个导出更“顺手”:
- FreeTimeGS 的 checkpoint 已经包含了每个 Gaussian 的 4D 参数:
  - 位置 means(在 canonical time 上的中心)
  - 速度 velocities(单位: meters/normalized_time)
  - 时间 times(canonical time,通常是最可见的中心时刻)
  - 时间尺度 durations(存的是 log(sigma),viewer 里用 exp 后作为时间高斯核的 sigma)
- 因此我们不需要像 4DGaussians 那样先导出一堆 PLY 再做差分拟合.

`.splat4d` 与本 Unity 插件(gsplat-unity)的 importer 对齐:
- 无 header, little-endian, 64 bytes/record
- 字段: position(float3), scale(float3), rgba8, quat8(wxyz), velocity(float3), time(float), duration(float), padding(3*float)
- 颜色只写 SH0(DC),并按 baseRgb = f_dc * SH_C0 + 0.5 量化到 uint8.

时间窗映射(关键):
- FreeTimeGS 的 temporal opacity 是高斯核:
    temporal_opacity(t) = exp(-0.5 * ((t - mu_t) / s)^2)
  其中 mu_t=times, s=exp(durations).

本脚本支持两种版本(由 `--splat4d-version` 控制):

## v1: hard window 语义(保持兼容,旧 importer 常见)
- `.splat4d` 解释为“硬窗口”:
    visible iff time0 <= t <= time0 + duration
- 用 `--temporal-threshold` 把 FreeTimeGS 的高斯核近似成硬窗口:
    half_width = s * sqrt(-2 * ln(threshold))
    window = [mu_t - half_width, mu_t + half_width]
- 写入:
  - `time=time0`(窗口起点)
  - `duration=window_length`
  - `position` 平移到 time0 时刻,保证线性运动轨迹一致.

## v2: Gaussian 语义(新增,更贴近 FreeTimeGS checkpoint)
- `.splat4d` 直接表达 FreeTimeGS 的时间高斯核:
    temporal_opacity(t) = exp(-0.5 * ((t - mu_t) / sigma)^2)
- 写入:
  - `time=mu_t`(checkpoint 的 times)
  - `duration=sigma`(=exp(checkpoint 的 durations),并 clamp `min_sigma`)
  - `position` 直接使用 checkpoint 的 means(它本来就是 mu_t 时刻的位置)

此外,本脚本也支持 `.splat4d` 的两种“文件格式版本”(由 `--splat4d-format-version` 控制):

## format v1: legacy(兼容旧 importer)
- 无 header,直接写 64B/record 的 record 数组.
- 仅承载 SH0(DC)与 4D 字段,不承载 SH rest.

## format v2: header + sections(用于承载 per-band SH 与 deltaSegments)
- 文件开头包含 magic `SPL4DV02` 的 header,并在尾部包含 section table.
- v2 可以承载:
  - per-band SH rest codebooks(`sh1/sh2/sh3`)与 base labels.
  - 可配置的 delta segment length,并写入 delta-v1 blocks(一期默认 updateCount=0,为未来 SH 动态变化铺路).
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch


SH_C0: float = 0.28209479177387814

# -----------------------------
# `.splat4d` v2(binary) format constants
# -----------------------------
_SPLAT4D_V2_MAGIC: bytes = b"SPL4DV02"
_SPLAT4D_V2_VERSION: int = 2
_SPLAT4D_V2_HEADER_SIZE_BYTES: int = 64

_SECT_MAGIC: bytes = b"SECT"
_SECT_VERSION: int = 1


def _fourcc(code: str) -> int:
    """
    把 4 字符 ASCII fourcc 转成 little-endian u32,用于写入 section kind.
    """
    b = code.encode("ascii")
    if len(b) != 4:
        raise ValueError(f"fourcc must be 4 chars, got {code!r}")
    return struct.unpack("<I", b)[0]


_SECT_RECS: int = _fourcc("RECS")
_SECT_META: int = _fourcc("META")
_SECT_SHCT: int = _fourcc("SHCT")  # SH centroids
_SECT_SHLB: int = _fourcc("SHLB")  # SH base labels(u16[N])
_SECT_SHDL: int = _fourcc("SHDL")  # SH delta(labelDeltaV1)


def _centroids_type_code(centroids_type: Literal["f16", "f32"]) -> int:
    if centroids_type == "f16":
        return 1
    if centroids_type == "f32":
        return 2
    raise ValueError(f"unknown centroids_type: {centroids_type}")


def _labels_encoding_code(labels_encoding: Literal["full", "delta-v1"]) -> int:
    if labels_encoding == "full":
        return 1
    if labels_encoding == "delta-v1":
        return 2
    raise ValueError(f"unknown labels_encoding: {labels_encoding}")


def _build_delta_segments(*, frame_count: int, segment_length: int) -> List[Tuple[int, int]]:
    """
    把 `[0, frame_count)` 切成多个连续 segment.

    返回: List[(startFrame, segmentFrameCount)].
    - 第 0 段 startFrame=0
    - 相邻段严格连续
    - 所有 frameCount 之和等于 frame_count
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    if segment_length <= 0:
        raise ValueError("segment_length must be > 0")

    segments: List[Tuple[int, int]] = []
    start = 0
    while start < int(frame_count):
        seg_count = min(int(segment_length), int(frame_count) - start)
        segments.append((int(start), int(seg_count)))
        start += int(seg_count)
    return segments


def _build_label_delta_v1_static(
    *,
    segment_start_frame: int,
    segment_frame_count: int,
    splat_count: int,
    label_count: int,
    magic: bytes,
) -> bytes:
    """
    生成 delta-v1 的最小实现: segment 内所有后续帧都无更新(updateCount=0).

    备注:
    - FreeTimeGS 的 SH 通常静态,labels 跨帧一致,因此 updateCount=0 是常见情况.
    - magic 允许用不同容器复用同一 delta 格式.
    """
    if segment_start_frame < 0:
        raise ValueError("segment_start_frame must be >= 0")
    if segment_frame_count <= 0:
        raise ValueError("segment_frame_count must be > 0")
    if splat_count <= 0:
        raise ValueError("splat_count must be > 0")
    if not (1 <= label_count <= 65535):
        raise ValueError("label_count must be in [1,65535]")
    if len(magic) != 8:
        raise ValueError("delta magic must be 8 bytes")

    header = magic + struct.pack(
        "<5I",
        1,  # version
        int(segment_start_frame),
        int(segment_frame_count),
        int(splat_count),
        int(label_count),
    )
    body = struct.pack("<I", 0) * int(max(segment_frame_count - 1, 0))
    return header + body


def _flatten_sh_rest_v2_per_band(shn: np.ndarray, *, bands: int) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    把 checkpoint 的 `shN[N,K,3]` 按 band 拆分成多个低维向量,用于 per-band kmeans.

    返回:
    - dict,按顺序包含(取决于 bands):
      - "sh1": ([N, 3*3], coeffCount=3)
      - "sh2": ([N, 5*3], coeffCount=5)
      - "sh3": ([N, 7*3], coeffCount=7)
    """
    if not (1 <= bands <= 3):
        raise ValueError(f"bands must be in [1,3], got {bands}")

    shn = shn.astype(np.float32, copy=False)
    n = int(shn.shape[0])
    if shn.ndim != 3 or shn.shape[0] != n or shn.shape[2] != 3:
        raise ValueError(f"shN must be [N,K,3], got shape={shn.shape}")

    rest_coeff_total = int((bands + 1) ** 2 - 1)
    if shn.shape[1] < rest_coeff_total:
        raise ValueError(
            f"shN coeff count too small: need >= {rest_coeff_total}, got {shn.shape[1]}"
        )

    sh_rest = shn[:, :rest_coeff_total, :]  # [N,restCoeffTotal,3]

    out: Dict[str, Tuple[np.ndarray, int]] = {}
    offset = 0

    if bands >= 1:
        coeff_count = 3
        band = sh_rest[:, offset : offset + coeff_count, :]
        out["sh1"] = (band.reshape(n, coeff_count * 3), coeff_count)
        offset += coeff_count

    if bands >= 2:
        coeff_count = 5
        band = sh_rest[:, offset : offset + coeff_count, :]
        out["sh2"] = (band.reshape(n, coeff_count * 3), coeff_count)
        offset += coeff_count

    if bands >= 3:
        coeff_count = 7
        band = sh_rest[:, offset : offset + coeff_count, :]
        out["sh3"] = (band.reshape(n, coeff_count * 3), coeff_count)
        offset += coeff_count

    if offset != rest_coeff_total:
        raise RuntimeError("internal error: per-band SH rest coeff offset mismatch")

    return out


class _ShCodebookResult:
    """
    SH rest 的 codebook(centroids)与 labels.

    - centroids_f32: [K,D] float32
    - labels_u16: [N] u16
    """

    def __init__(self, *, centroids_f32: np.ndarray, labels_u16: np.ndarray) -> None:
        self.centroids_f32 = centroids_f32
        self.labels_u16 = labels_u16


def _build_sh_codebook_and_labels(
    sh_rest_flat: np.ndarray,
    *,
    name: str,
    codebook_size: int,
    sample_size: int,
    seed: int,
    assign_chunk: int,
    kmeans_iters: int,
) -> _ShCodebookResult:
    """
    用 kmeans2 拟合 SH rest codebook,并对全量 splat 分配 labels(u16).

    说明:
    - 我们复用 `.sog4d` exporter 的实现策略:
      - 捕获 empty cluster warning,最多重试 3 次.
      - 用 KDTree 为全量分配最近中心.
    """
    if not (1 <= codebook_size <= 65535):
        raise ValueError(f"--shn-count must be in [1,65535], got {codebook_size}")
    if sample_size <= 0:
        raise ValueError(f"--shn-codebook-sample must be > 0, got {sample_size}")
    if assign_chunk <= 0:
        raise ValueError(f"--shn-assign-chunk must be > 0, got {assign_chunk}")
    if kmeans_iters <= 0:
        raise ValueError(f"--shn-kmeans-iters must be > 0, got {kmeans_iters}")

    # 延迟 import,避免在不导出 SH 时引入 SciPy 依赖.
    import warnings

    from scipy.cluster.vq import kmeans2
    from scipy.spatial import cKDTree

    x = sh_rest_flat.astype(np.float32, copy=False)
    n = int(x.shape[0])
    d = int(x.shape[1])
    if x.shape != (n, d):
        raise ValueError("internal error: sh_rest_flat shape mismatch")
    if d % 3 != 0:
        raise ValueError(f"internal error: sh_rest_flat dim must be multiple of 3, got {d}")

    rng = np.random.default_rng(int(seed))
    sample_n = min(int(sample_size), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    sample = x[sample_idx]

    print(f"[splat4d] {name} kmeans: sample={sample_n:,}, K={codebook_size}, D={d}")
    centroids: Optional[np.ndarray] = None
    for attempt in range(3):
        attempt_seed = int(seed) + int(attempt)

        # kmeans2 内部依赖 numpy 的全局 RNG,这里临时设置并恢复.
        rng_state = np.random.get_state()
        try:
            np.random.seed(int(attempt_seed))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                c_try, labels_try = kmeans2(
                    sample,
                    int(codebook_size),
                    iter=int(kmeans_iters),
                    minit="++",
                )
        finally:
            np.random.set_state(rng_state)

        warned_empty = any("clusters is empty" in str(w.message) for w in caught)
        used = int(np.unique(labels_try).size)
        centroids = c_try

        if (not warned_empty) and used == int(codebook_size):
            break
        if attempt == 0:
            print(f"[splat4d] {name} kmeans: empty cluster detected, retrying...")
        print(f"[splat4d] {name} kmeans retry {attempt+1}/3: used={used}/{int(codebook_size)} seed={attempt_seed}")

    if centroids is None:
        raise RuntimeError("internal error: kmeans2 returned no centroids")

    centroids_f32 = centroids.astype(np.float32, copy=False)

    tree = cKDTree(centroids_f32)
    labels = np.empty((n,), dtype=np.uint16)
    for start in range(0, n, int(assign_chunk)):
        end = min(start + int(assign_chunk), n)
        _, idx = tree.query(x[start:end], k=1, workers=-1)
        labels[start:end] = idx.astype(np.uint16)

    return _ShCodebookResult(centroids_f32=centroids_f32, labels_u16=labels)


def _safe_f32(x: np.ndarray, *, default: float) -> np.ndarray:
    """
    把非有限值(nan/inf)替换为 default,避免导出结果出现不可读数据.
    """
    x = x.astype(np.float32, copy=False)
    return np.where(np.isfinite(x), x, np.float32(default)).astype(np.float32)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _quantize_0_1_to_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), 0.0, 1.0)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32, copy=False)
    norm = np.linalg.norm(q, axis=1, keepdims=True).astype(np.float32)  # [N,1]
    norm1 = norm.squeeze(1)  # [N]
    good = np.isfinite(norm1) & (norm1 >= 1e-8)

    out = np.empty_like(q, dtype=np.float32)
    out[good] = q[good] / norm[good]
    out[~good] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # q 与 -q 表示同一旋转. 强制 w>=0,减少量化抖动.
    flip = out[:, 0] < 0
    out[flip] *= -1.0
    return out


def _quantize_quat_to_u8(q_norm: np.ndarray) -> np.ndarray:
    q = np.clip(q_norm, -1.0, 1.0)
    q8 = np.round(q * 128.0 + 128.0).astype(np.int32)
    q8 = np.clip(q8, 0, 255).astype(np.uint8)
    return q8


def _record_dtype() -> np.dtype:
    dt = np.dtype(
        [
            ("px", "<f4"),
            ("py", "<f4"),
            ("pz", "<f4"),
            ("sx", "<f4"),
            ("sy", "<f4"),
            ("sz", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
            ("a", "u1"),
            ("rw", "u1"),
            ("rx", "u1"),
            ("ry", "u1"),
            ("rz", "u1"),
            ("vx", "<f4"),
            ("vy", "<f4"),
            ("vz", "<f4"),
            ("time", "<f4"),
            ("duration", "<f4"),
            ("pad0", "<f4"),
            ("pad1", "<f4"),
            ("pad2", "<f4"),
        ],
        align=False,
    )
    if dt.itemsize != 64:
        raise RuntimeError(f"internal error: record dtype itemsize={dt.itemsize}, expected 64")
    return dt


def _as_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().to(dtype=torch.float32).numpy()


def export_splat4d_from_ckpt(
    *,
    ckpt_path: Path,
    output_path: Path,
    splat4d_format_version: int,
    splat4d_version: int,
    temporal_threshold: float,
    min_sigma: float,
    chunk_size: int,
    base_opacity_threshold: float,
    sh_bands: int,
    shn_count: int,
    shn_centroids_type: Literal["f16", "f32"],
    shn_labels_encoding: Literal["full", "delta-v1"],
    frame_count: int,
    delta_segment_length: int,
    shn_codebook_sample: int,
    shn_assign_chunk: int,
    shn_kmeans_iters: int,
    seed: int,
) -> None:
    if splat4d_format_version not in (1, 2):
        raise ValueError("--splat4d-format-version must be 1 or 2")
    if splat4d_version not in (1, 2):
        raise ValueError("--splat4d-version must be 1 or 2")
    if splat4d_version == 1 and not (0.0 < temporal_threshold < 1.0):
        raise ValueError("--temporal-threshold must be in (0, 1) for splat4d-version=1")
    if min_sigma <= 0.0:
        raise ValueError("--min-sigma must be > 0")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if not (0.0 <= base_opacity_threshold <= 1.0):
        raise ValueError("--base-opacity-threshold must be in [0, 1]")
    if not (0 <= sh_bands <= 3):
        raise ValueError("--sh-bands must be in [0,3]")
    if sh_bands > 0 and splat4d_format_version != 2:
        raise ValueError("SH rest export requires --splat4d-format-version 2")
    if delta_segment_length < 0:
        raise ValueError("--delta-segment-length must be >= 0")
    if sh_bands > 0:
        if not (1 <= shn_count <= 65535):
            raise ValueError("--shn-count must be in [1,65535]")
        if shn_labels_encoding == "delta-v1" and frame_count <= 0:
            raise ValueError("--frame-count must be > 0 when --shn-labels-encoding=delta-v1")

    if splat4d_version == 1:
        sigma_factor = math.sqrt(-2.0 * math.log(temporal_threshold))
        print(f"[splat4d] version=1(hard-window) temporal_threshold={temporal_threshold} -> sigma_factor={sigma_factor:.6f}")
    else:
        sigma_factor = 0.0
        print(f"[splat4d] version=2(gaussian) time=mu_t, duration=sigma, temporal_cutoff={temporal_threshold}")

    print(f"[splat4d] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    required = ["means", "scales", "quats", "opacities", "sh0", "times", "durations", "velocities"]
    if sh_bands > 0:
        required.append("shN")
    missing = [k for k in required if k not in splats]
    if missing:
        raise KeyError(f"checkpoint missing splats keys: {missing}")

    means = splats["means"]  # [N,3] canonical position at mu_t
    scales_log = splats["scales"]  # [N,3] log scale
    quats = splats["quats"]  # [N,4] quaternion(wxyz)
    opacities_logit = splats["opacities"]  # [N] logit
    sh0 = splats["sh0"]  # [N,1,3]
    times = splats["times"]  # [N,1] canonical time mu_t
    durations_log = splats["durations"]  # [N,1] log sigma
    velocities = splats["velocities"]  # [N,3] meters/normalized_time
    shn: Optional[torch.Tensor] = splats.get("shN") if sh_bands > 0 else None

    n_total = int(means.shape[0])
    print(f"[splat4d] gaussians: {n_total:,}")

    # base opacity filter(可选): 先在 CPU 上做一次筛选,降低输出体积.
    # 注意: 这只是 base opacity,不会考虑 temporal opacity.
    if base_opacity_threshold > 0.0:
        base_opacity = torch.sigmoid(opacities_logit.detach().cpu())
        keep = base_opacity >= float(base_opacity_threshold)
        keep_count = int(keep.sum().item())
        print(f"[splat4d] base opacity filter: >= {base_opacity_threshold} -> keep {keep_count:,}/{n_total:,}")
        means = means[keep]
        scales_log = scales_log[keep]
        quats = quats[keep]
        opacities_logit = opacities_logit[keep]
        sh0 = sh0[keep]
        times = times[keep]
        durations_log = durations_log[keep]
        velocities = velocities[keep]
        if shn is not None:
            shn = shn[keep]
        n_total = keep_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = _record_dtype()

    # 统计信息(用于 sanity check)
    times_np = _as_numpy_f32(times).reshape(-1)
    sigmas_np = np.exp(_as_numpy_f32(durations_log).reshape(-1))
    print(f"[splat4d] times range: [{times_np.min():.6f}, {times_np.max():.6f}]")
    print(f"[splat4d] sigma(exp(duration)) range: [{sigmas_np.min():.6f}, {sigmas_np.max():.6f}]")

    # -----------------------------
    # format v1: legacy,仅写 record 数组
    # -----------------------------
    if int(splat4d_format_version) == 1:
        bytes_expected = n_total * 64
        print(f"[splat4d] format=v1(legacy) output: {output_path} ({bytes_expected/1024/1024:.1f} MB)")

        with output_path.open("wb") as f:
            written = 0
            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                n = end - start

                means_np = _as_numpy_f32(means[start:end])  # [n,3]
                vel_np = _as_numpy_f32(velocities[start:end])  # [n,3]
                mu_np = _as_numpy_f32(times[start:end]).reshape(-1)  # [n]
                sigma_np = np.exp(_as_numpy_f32(durations_log[start:end]).reshape(-1))  # [n]
                sigma_np = np.clip(sigma_np, float(min_sigma), np.inf).astype(np.float32)

                mu_np = _safe_f32(mu_np, default=0.0)
                sigma_np = _safe_f32(sigma_np, default=float(min_sigma))

                if splat4d_version == 1:
                    half_width = (sigma_np * np.float32(sigma_factor)).astype(np.float32)  # [n]
                    t0 = (mu_np - half_width).astype(np.float32)
                    t1 = (mu_np + half_width).astype(np.float32)
                    t0 = np.clip(t0, 0.0, 1.0)
                    t1 = np.clip(t1, 0.0, 1.0)
                    duration = (t1 - t0).astype(np.float32)
                    duration = np.clip(duration, 0.0, 1.0)

                    # 位置: checkpoint 的 means 是 mu_t 时刻的位置,我们要平移到 time0=t0.
                    pos0 = means_np + vel_np * (t0.reshape(-1, 1) - mu_np.reshape(-1, 1))
                    out_time = t0
                    out_duration = duration
                else:
                    # legacy v1 文件格式无法表达“gaussian 时间核”的语义差异.
                    # 这里仍然写入 time=mu,duration=sigma,但注意: 旧 importer 会按 window 解释.
                    pos0 = means_np
                    out_time = mu_np.astype(np.float32, copy=False)
                    out_duration = sigma_np.astype(np.float32, copy=False)

                scales_lin = np.exp(_as_numpy_f32(scales_log[start:end]))  # [n,3]
                q_norm = _normalize_quat_wxyz(_as_numpy_f32(quats[start:end]))  # [n,4]
                q8 = _quantize_quat_to_u8(q_norm)  # [n,4]

                f_dc = _as_numpy_f32(sh0[start:end]).reshape(n, 3)
                base_rgb = 0.5 + SH_C0 * f_dc
                rgb8 = _quantize_0_1_to_u8(base_rgb)  # [n,3]

                opa_logit = _as_numpy_f32(opacities_logit[start:end]).reshape(-1)
                alpha = _stable_sigmoid(opa_logit)
                a8 = _quantize_0_1_to_u8(alpha)  # [n]

                rec = np.empty(n, dtype=dt)
                rec["px"] = pos0[:, 0]
                rec["py"] = pos0[:, 1]
                rec["pz"] = pos0[:, 2]
                rec["sx"] = scales_lin[:, 0]
                rec["sy"] = scales_lin[:, 1]
                rec["sz"] = scales_lin[:, 2]
                rec["r"] = rgb8[:, 0]
                rec["g"] = rgb8[:, 1]
                rec["b"] = rgb8[:, 2]
                rec["a"] = a8
                rec["rw"] = q8[:, 0]
                rec["rx"] = q8[:, 1]
                rec["ry"] = q8[:, 2]
                rec["rz"] = q8[:, 3]
                rec["vx"] = vel_np[:, 0]
                rec["vy"] = vel_np[:, 1]
                rec["vz"] = vel_np[:, 2]
                rec["time"] = out_time
                rec["duration"] = out_duration
                rec["pad0"] = np.float32(0.0)
                rec["pad1"] = np.float32(0.0)
                rec["pad2"] = np.float32(0.0)

                f.write(rec.tobytes(order="C"))
                written += n

                if written == n or (written // chunk_size) % 10 == 0 or written == n_total:
                    print(f"[splat4d] wrote {written:,}/{n_total:,}")

        print("[splat4d] done")
        return

    # -----------------------------
    # format v2: header + sections
    # -----------------------------
    print(f"[splat4d] format=v2(header+sections) output: {output_path}")

    # 1) 预计算 SH per-band codebook + labels(静态)
    sh_band_centroids_bytes: Dict[int, bytes] = {}
    sh_band_labels_u16: Dict[int, np.ndarray] = {}
    sh_band_codebook_count: Dict[int, int] = {}

    if sh_bands > 0:
        if shn is None:
            raise RuntimeError("internal error: shn is None but sh_bands > 0")
        shn_np = _as_numpy_f32(shn)  # [N,K,3]
        per_band = _flatten_sh_rest_v2_per_band(shn_np, bands=int(sh_bands))

        for band_idx, (band_name, (band_flat, _coeff_count)) in enumerate(per_band.items()):
            # band: 1,2,3
            band = 1 + int(band_idx)
            band_seed = int(seed) + int(band_idx)
            res = _build_sh_codebook_and_labels(
                band_flat,
                name=str(band_name),
                codebook_size=int(shn_count),
                sample_size=int(shn_codebook_sample),
                seed=int(band_seed),
                assign_chunk=int(shn_assign_chunk),
                kmeans_iters=int(shn_kmeans_iters),
            )

            codebook_count = int(res.centroids_f32.shape[0])
            sh_band_codebook_count[int(band)] = int(codebook_count)
            sh_band_labels_u16[int(band)] = res.labels_u16.astype(np.uint16, copy=False)

            if shn_centroids_type == "f16":
                sh_band_centroids_bytes[int(band)] = res.centroids_f32.astype("<f2", copy=False).tobytes(order="C")
            elif shn_centroids_type == "f32":
                sh_band_centroids_bytes[int(band)] = res.centroids_f32.astype("<f4", copy=False).tobytes(order="C")
            else:  # pragma: no cover
                raise ValueError(f"unknown shn_centroids_type: {shn_centroids_type}")

        # 小 sanity: labels 必须在 [0,codebookCount) 内(理论上 KDTree 分配不会越界).
        for band, labels_u16 in sh_band_labels_u16.items():
            max_label = int(labels_u16.max(initial=0))
            if max_label >= int(sh_band_codebook_count[band]):
                raise RuntimeError(
                    f"internal error: band={band} labels has out-of-range value {max_label} >= {sh_band_codebook_count[band]}"
                )

    # 2) delta segments(仅用于写 section entry + delta bytes)
    delta_segments: Optional[List[Tuple[int, int]]] = None
    if sh_bands > 0 and shn_labels_encoding == "delta-v1":
        seg_len = int(frame_count) if int(delta_segment_length) == 0 else int(delta_segment_length)
        seg_len = min(int(seg_len), int(frame_count))
        delta_segments = _build_delta_segments(frame_count=int(frame_count), segment_length=int(seg_len))
        print(f"[splat4d] sh delta segments: count={len(delta_segments)}, segment_length={seg_len}")

    # 3) 写文件: header(占位) -> sections -> section table -> 回填 header
    sections: List[Tuple[int, int, int, int, int, int]] = []
    # sections entry tuple:
    # (kindFourCC, band, startFrame, frameCount, offset, length)

    temporal_cutoff = float(temporal_threshold)
    time_model = 1 if int(splat4d_version) == 1 else 2

    with output_path.open("wb") as f:
        # header placeholder
        f.write(b"\x00" * int(_SPLAT4D_V2_HEADER_SIZE_BYTES))

        # ---- RECS ----
        recs_offset = int(f.tell())
        written = 0
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            n = end - start

            means_np = _as_numpy_f32(means[start:end])  # [n,3]
            vel_np = _as_numpy_f32(velocities[start:end])  # [n,3]
            mu_np = _as_numpy_f32(times[start:end]).reshape(-1)  # [n]
            sigma_np = np.exp(_as_numpy_f32(durations_log[start:end]).reshape(-1))  # [n]
            sigma_np = np.clip(sigma_np, float(min_sigma), np.inf).astype(np.float32)

            mu_np = _safe_f32(mu_np, default=0.0)
            sigma_np = _safe_f32(sigma_np, default=float(min_sigma))

            if splat4d_version == 1:
                half_width = (sigma_np * np.float32(sigma_factor)).astype(np.float32)  # [n]
                t0 = (mu_np - half_width).astype(np.float32)
                t1 = (mu_np + half_width).astype(np.float32)
                t0 = np.clip(t0, 0.0, 1.0)
                t1 = np.clip(t1, 0.0, 1.0)
                duration = (t1 - t0).astype(np.float32)
                duration = np.clip(duration, 0.0, 1.0)

                pos0 = means_np + vel_np * (t0.reshape(-1, 1) - mu_np.reshape(-1, 1))
                out_time = t0
                out_duration = duration
            else:
                pos0 = means_np
                out_time = mu_np.astype(np.float32, copy=False)
                out_duration = sigma_np.astype(np.float32, copy=False)

            scales_lin = np.exp(_as_numpy_f32(scales_log[start:end]))  # [n,3]
            q_norm = _normalize_quat_wxyz(_as_numpy_f32(quats[start:end]))  # [n,4]
            q8 = _quantize_quat_to_u8(q_norm)  # [n,4]

            f_dc = _as_numpy_f32(sh0[start:end]).reshape(n, 3)
            base_rgb = 0.5 + SH_C0 * f_dc
            rgb8 = _quantize_0_1_to_u8(base_rgb)  # [n,3]

            opa_logit = _as_numpy_f32(opacities_logit[start:end]).reshape(-1)
            alpha = _stable_sigmoid(opa_logit)
            a8 = _quantize_0_1_to_u8(alpha)  # [n]

            rec = np.empty(n, dtype=dt)
            rec["px"] = pos0[:, 0]
            rec["py"] = pos0[:, 1]
            rec["pz"] = pos0[:, 2]
            rec["sx"] = scales_lin[:, 0]
            rec["sy"] = scales_lin[:, 1]
            rec["sz"] = scales_lin[:, 2]
            rec["r"] = rgb8[:, 0]
            rec["g"] = rgb8[:, 1]
            rec["b"] = rgb8[:, 2]
            rec["a"] = a8
            rec["rw"] = q8[:, 0]
            rec["rx"] = q8[:, 1]
            rec["ry"] = q8[:, 2]
            rec["rz"] = q8[:, 3]
            rec["vx"] = vel_np[:, 0]
            rec["vy"] = vel_np[:, 1]
            rec["vz"] = vel_np[:, 2]
            rec["time"] = out_time
            rec["duration"] = out_duration
            rec["pad0"] = np.float32(0.0)
            rec["pad1"] = np.float32(0.0)
            rec["pad2"] = np.float32(0.0)

            f.write(rec.tobytes(order="C"))
            written += n
            if written == n or (written // chunk_size) % 10 == 0 or written == n_total:
                print(f"[splat4d] wrote recs {written:,}/{n_total:,}")

        recs_length = int(f.tell()) - int(recs_offset)
        sections.append((_SECT_RECS, 0, 0, 0, int(recs_offset), int(recs_length)))

        # ---- META ----
        meta_offset = int(f.tell())
        band_infos: List[Tuple[int, int, int, int]] = []
        for band in (1, 2, 3):
            if band <= int(sh_bands):
                band_infos.append(
                    (
                        int(sh_band_codebook_count.get(band, 0)),
                        int(_centroids_type_code(shn_centroids_type)),
                        int(_labels_encoding_code(shn_labels_encoding)),
                        0,
                    )
                )
            else:
                band_infos.append((0, 0, 0, 0))

        meta_bytes = struct.pack(
            "<IfII" + "IIII" * 3,
            1,  # metaVersion
            float(temporal_cutoff),
            int(delta_segment_length),
            0,
            *[x for tup in band_infos for x in tup],
        )
        if len(meta_bytes) != 64:
            raise RuntimeError(f"internal error: meta_bytes size={len(meta_bytes)}, expected 64")
        f.write(meta_bytes)
        sections.append((_SECT_META, 0, 0, 0, int(meta_offset), int(len(meta_bytes))))

        # ---- SH sections ----
        if sh_bands > 0:
            # centroids
            for band in range(1, int(sh_bands) + 1):
                centroids_offset = int(f.tell())
                centroids_bytes = sh_band_centroids_bytes[int(band)]
                f.write(centroids_bytes)
                sections.append((_SECT_SHCT, int(band), 0, 0, int(centroids_offset), int(len(centroids_bytes))))

            # base labels: 为避免静态 labels 在每个 segment 重复占用空间,我们每个 band 只写 1 份.
            labels_offset_by_band: Dict[int, int] = {}
            labels_length_by_band: Dict[int, int] = {}
            for band in range(1, int(sh_bands) + 1):
                labels_offset = int(f.tell())
                labels_u16 = sh_band_labels_u16[int(band)].astype("<u2", copy=False)
                labels_bytes = labels_u16.tobytes(order="C")
                f.write(labels_bytes)
                labels_offset_by_band[int(band)] = int(labels_offset)
                labels_length_by_band[int(band)] = int(len(labels_bytes))

            if shn_labels_encoding == "full":
                # 1 个 labels blob 足够.
                for band in range(1, int(sh_bands) + 1):
                    sections.append(
                        (
                            _SECT_SHLB,
                            int(band),
                            0,
                            0,
                            int(labels_offset_by_band[int(band)]),
                            int(labels_length_by_band[int(band)]),
                        )
                    )
            else:
                if delta_segments is None:
                    raise RuntimeError("internal error: delta_segments is None but delta-v1 requested")
                for band in range(1, int(sh_bands) + 1):
                    label_count = int(sh_band_codebook_count[int(band)])
                    for seg_start, seg_count in delta_segments:
                        # base labels entry(指向同一份 labels blob)
                        sections.append(
                            (
                                _SECT_SHLB,
                                int(band),
                                int(seg_start),
                                int(seg_count),
                                int(labels_offset_by_band[int(band)]),
                                int(labels_length_by_band[int(band)]),
                            )
                        )

                        # delta bytes(每个 segment 单独写,header 中包含 segmentStartFrame/FrameCount)
                        delta_offset = int(f.tell())
                        delta_bytes = _build_label_delta_v1_static(
                            segment_start_frame=int(seg_start),
                            segment_frame_count=int(seg_count),
                            splat_count=int(n_total),
                            label_count=int(label_count),
                            magic=b"SPL4DLB1",
                        )
                        f.write(delta_bytes)
                        sections.append(
                            (
                                _SECT_SHDL,
                                int(band),
                                int(seg_start),
                                int(seg_count),
                                int(delta_offset),
                                int(len(delta_bytes)),
                            )
                        )

        # ---- SectionTable ----
        section_table_offset = int(f.tell())
        section_count = int(len(sections))

        f.write(_SECT_MAGIC)
        f.write(struct.pack("<III", int(_SECT_VERSION), int(section_count), 0))
        for kind, band, start_frame, seg_frame_count, offset, length in sections:
            f.write(
                struct.pack(
                    "<4I2Q",
                    int(kind),
                    int(band),
                    int(start_frame),
                    int(seg_frame_count),
                    int(offset),
                    int(length),
                )
            )

        # 回填 header
        header_bytes = _SPLAT4D_V2_MAGIC + struct.pack(
            "<8I3Q",
            int(_SPLAT4D_V2_VERSION),
            int(_SPLAT4D_V2_HEADER_SIZE_BYTES),
            int(section_count),
            int(64),  # recordSizeBytes
            int(n_total),  # splatCount
            int(sh_bands),  # shBands
            int(time_model),  # timeModel
            int(frame_count if (sh_bands > 0 and shn_labels_encoding == "delta-v1") else 0),
            int(section_table_offset),
            0,
            0,
        )
        if len(header_bytes) != int(_SPLAT4D_V2_HEADER_SIZE_BYTES):
            raise RuntimeError(f"internal error: header size={len(header_bytes)}, expected {_SPLAT4D_V2_HEADER_SIZE_BYTES}")
        f.seek(0)
        f.write(header_bytes)

    print("[splat4d] done")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Export .splat4d from FreeTimeGsVanilla checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to ckpt_*.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output .splat4d path")
    parser.add_argument(
        "--splat4d-format-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="文件格式版本: 1=legacy无header(仅SH0); 2=header+sections(支持SH rest与deltaSegments)",
    )

    parser.add_argument(
        "--splat4d-version",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "导出版本: 1=hard-window(旧语义,通过 temporal-threshold 近似); "
            "2=gaussian(新语义,time=mu_t,duration=sigma,更贴近 FreeTimeGS)"
        ),
    )
    parser.add_argument(
        "--temporal-threshold",
        type=float,
        default=0.01,
        help=(
            "v1(window)用于把 sigma 近似为硬窗口; "
            "v2(gaussian)作为 runtime 的 cutoff(小于该权重视为不可见). (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=0.02,
        help="Clamp sigma(exp(duration)) to at least this value (default: 0.02, matches viewer)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Records per chunk to limit peak memory (default: 1,000,000)",
    )
    parser.add_argument(
        "--base-opacity-threshold",
        type=float,
        default=0.0,
        help="Optional base opacity filter in [0,1] to reduce file size (default: 0, keep all)",
    )
    parser.add_argument("--sh-bands", type=int, default=0, choices=[0, 1, 2, 3], help="SH bands for rest coefficients (0..3)")
    parser.add_argument("--shn-count", type=int, default=512, help="Codebook size for per-band SH(rest) (default: 512)")
    parser.add_argument(
        "--shn-centroids-type",
        type=str,
        default="f16",
        choices=["f16", "f32"],
        help="Centroids scalar type: f16 or f32 (default: f16)",
    )
    parser.add_argument(
        "--shn-labels-encoding",
        type=str,
        default="delta-v1",
        choices=["full", "delta-v1"],
        help="Labels encoding: full | delta-v1 (default: delta-v1)",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Frame count for delta-v1 segments. Required when --shn-labels-encoding=delta-v1 (default: 0)",
    )
    parser.add_argument(
        "--delta-segment-length",
        type=int,
        default=0,
        help="Delta segment length. 0 means single segment covering all frames (default: 0)",
    )
    parser.add_argument(
        "--shn-codebook-sample",
        type=int,
        default=100_000,
        help="Sample size for SH kmeans codebook fitting (default: 100000)",
    )
    parser.add_argument(
        "--shn-assign-chunk",
        type=int,
        default=200_000,
        help="Chunk size when assigning labels via KDTree (default: 200000)",
    )
    parser.add_argument(
        "--shn-kmeans-iters",
        type=int,
        default=10,
        help="kmeans iterations for SH codebook fitting (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for codebook fitting (default: 12345)")

    args = parser.parse_args(argv)

    export_splat4d_from_ckpt(
        ckpt_path=args.ckpt,
        output_path=args.output,
        splat4d_format_version=int(args.splat4d_format_version),
        splat4d_version=int(args.splat4d_version),
        temporal_threshold=args.temporal_threshold,
        min_sigma=args.min_sigma,
        chunk_size=args.chunk_size,
        base_opacity_threshold=args.base_opacity_threshold,
        sh_bands=int(args.sh_bands),
        shn_count=int(args.shn_count),
        shn_centroids_type=str(args.shn_centroids_type),  # type: ignore[arg-type]
        shn_labels_encoding=str(args.shn_labels_encoding),  # type: ignore[arg-type]
        frame_count=int(args.frame_count),
        delta_segment_length=int(args.delta_segment_length),
        shn_codebook_sample=int(args.shn_codebook_sample),
        shn_assign_chunk=int(args.shn_assign_chunk),
        shn_kmeans_iters=int(args.shn_kmeans_iters),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
