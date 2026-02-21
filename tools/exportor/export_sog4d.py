#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 FreeTimeGsVanilla 的 checkpoint(.pt)导出 `.sog4d`(ZIP bundle).

实现依据(请先读这 3 份,它们是“规格与施工图”):
- `tools/exportor/spec.md`(sog4d-sequence-encoding,权威约束: layout/timeMapping/streams)
- `tools/exportor/FreeTimeGsCheckpointToSog4D.md`(从 FreeTimeGS checkpoint 到 `.sog4d` 的映射清单)
- `tools/exportor/export_splat4d.py`(本仓库已有的 4D 参数读取/量化实现,可作为对齐参考)

导出策略(先跑通,再扩展):
- 先实现 `bands=0`:
  - 写 per-frame position(u16 hi/lo WebP)
  - 写 per-frame scale(codebook + indices WebP,内容可复用)
  - 写 per-frame rotation(quat u8 WebP,内容可复用)
  - 写 per-frame sh0.webp(RGB=DC SH 系数索引,A=opacity(t))
  - 不写 SH rest(shN),以降低复杂度与导出体积

注意:
- `.sog4d` 要求 splat identity 在所有帧稳定,因此任何裁剪都必须是“全局裁剪”(对所有帧一致).
- 本脚本提供 `--base-opacity-threshold` 与 `--max-splats` 作为可选全局裁剪手段.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import warnings
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree


ZipCompression = Literal["stored", "deflated"]
TimeMappingType = Literal["uniform", "explicit"]


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    数值稳定的 sigmoid,避免极端 logit 下溢/上溢.
    """
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """
    归一化四元数(w,x,y,z),并做半球规范化(w>=0),减少插值抖动.
    """
    q = q.astype(np.float32, copy=False)
    norm = np.linalg.norm(q, axis=1, keepdims=True).astype(np.float32)  # [N,1]
    norm1 = norm.squeeze(1)  # [N]
    good = np.isfinite(norm1) & (norm1 >= 1e-8)

    out = np.empty_like(q, dtype=np.float32)
    out[good] = q[good] / norm[good]
    out[~good] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    flip = out[:, 0] < 0
    out[flip] *= -1.0
    return out


def _quantize_quat_to_u8(q_norm: np.ndarray) -> np.ndarray:
    """
    把归一化 quaternion 量化到 RGBA8,解码规则与 `.splat4d` 保持一致.
    """
    q = np.clip(q_norm, -1.0, 1.0)
    q8 = np.round(q * 128.0 + 128.0).astype(np.int32)
    q8 = np.clip(q8, 0, 255).astype(np.uint8)
    return q8


def _format_frame(frame_idx: int) -> str:
    """
    `{frame}` 替换规则: 十进制,左侧补零到至少 5 位.
    """
    return f"{int(frame_idx):05d}"


def _as_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().to(dtype=torch.float32).numpy()


def _encode_webp_bytes_rgba(
    img_rgba_u8: np.ndarray,
    *,
    webp_method: int,
    webp_quality: int,
) -> bytes:
    """
    把 RGBA8 numpy 图编码为 lossless WebP bytes.

    备注:
    - `method` 越大压缩越好但越慢.
    - `.sog4d` 约定为数据图,因此这里固定 lossless=True.
    """
    if img_rgba_u8.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got dtype={img_rgba_u8.dtype}")
    if img_rgba_u8.ndim != 3 or img_rgba_u8.shape[2] != 4:
        raise ValueError(f"expected HxWx4 RGBA, got shape={img_rgba_u8.shape}")

    img = Image.fromarray(img_rgba_u8, mode="RGBA")
    buf = BytesIO()
    img.save(
        buf,
        format="WEBP",
        lossless=True,
        quality=int(webp_quality),
        method=int(webp_method),
    )
    return buf.getvalue()


def _write_webp_rgba_to_zip(
    zf: zipfile.ZipFile,
    *,
    path_in_zip: str,
    img_rgba_u8: np.ndarray,
    webp_method: int,
    webp_quality: int,
) -> None:
    """
    直接把 WebP 写入 zip,避免落地中间文件占空间.
    """
    img = Image.fromarray(img_rgba_u8, mode="RGBA")
    with zf.open(path_in_zip, "w") as fp:
        img.save(
            fp,
            format="WEBP",
            lossless=True,
            quality=int(webp_quality),
            method=int(webp_method),
        )


def _build_layout(*, splat_count: int, width: int) -> Tuple[int, int, int]:
    """
    返回:
    - width
    - height
    - pixel_count(=width*height)
    """
    if splat_count <= 0:
        raise ValueError(f"splat_count must be > 0, got {splat_count}")
    if width <= 0:
        raise ValueError(f"layout width must be > 0, got {width}")

    height = int(math.ceil(splat_count / float(width)))
    pixel_count = int(width * height)
    if pixel_count < splat_count:
        raise RuntimeError("internal error: pixel_count < splat_count")
    return width, height, pixel_count


def _quantize_u16_hi_lo(
    x: np.ndarray,
    *,
    range_min: np.ndarray,
    range_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 float32 的 [N,3] 量化为 u16,并拆成 hi/lo 的 uint8 [N,3].

    量化规则(与 spec 对齐):
    - t = clamp01((x - min)/(max - min))
    - q_u16 = round(t * 65535)
    - hi = q >> 8, lo = q & 255
    """
    x = x.astype(np.float32, copy=False)
    range_min = range_min.astype(np.float32, copy=False).reshape(1, 3)
    range_max = range_max.astype(np.float32, copy=False).reshape(1, 3)

    denom = (range_max - range_min).astype(np.float32, copy=False)  # [1,3]
    # 避免除 0: 若某维度 max==min,则该维度全部量化为 0.
    safe = np.where(np.abs(denom) >= 1e-12, denom, 1.0).astype(np.float32)
    t = (x - range_min) / safe
    # 对于 denom==0 的维度,强制 t=0.
    t = np.where(np.abs(denom) >= 1e-12, t, 0.0).astype(np.float32)
    t = np.clip(t, 0.0, 1.0)

    q = np.round(t * 65535.0).astype(np.int64)
    q = np.clip(q, 0, 65535).astype(np.uint16)

    hi = (q >> 8).astype(np.uint8)
    lo = (q & np.uint16(255)).astype(np.uint8)
    return hi, lo


def _make_rgba_image_from_rgb_and_a(
    *,
    rgb_u8: np.ndarray,
    a_u8: np.ndarray,
    pixel_count: int,
    width: int,
    height: int,
    default_a: int,
) -> np.ndarray:
    """
    把 splat 的 RGB(u8)与 A(u8)按 row-major layout 写入 RGBA8 图像.
    """
    n = int(rgb_u8.shape[0])
    if rgb_u8.shape != (n, 3):
        raise ValueError(f"rgb_u8 must be [N,3], got {rgb_u8.shape}")
    if a_u8.shape != (n,):
        raise ValueError(f"a_u8 must be [N], got {a_u8.shape}")

    flat = np.zeros((pixel_count, 4), dtype=np.uint8)
    flat[:, 3] = np.uint8(int(default_a))
    flat[:n, 0:3] = rgb_u8
    flat[:n, 3] = a_u8
    return flat.reshape(height, width, 4)


def _make_rgba_image_from_rgba_flat(
    rgba_u8: np.ndarray,
    *,
    pixel_count: int,
    width: int,
    height: int,
    default_a: int,
) -> np.ndarray:
    """
    把 splat 的 RGBA(u8)按 row-major layout 写入 RGBA8 图像.
    """
    n = int(rgba_u8.shape[0])
    if rgba_u8.shape != (n, 4):
        raise ValueError(f"rgba_u8 must be [N,4], got {rgba_u8.shape}")

    flat = np.zeros((pixel_count, 4), dtype=np.uint8)
    flat[:, 3] = np.uint8(int(default_a))
    flat[:n] = rgba_u8
    return flat.reshape(height, width, 4)


def _build_sh0_codebook(
    f_dc: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """
    构建 `sh0Codebook[256]`.

    这里用 quantile 做 codebook:
    - 速度快,结果稳定.
    - 对重尾分布也更稳(比均匀分桶更不容易浪费码字).
    """
    values = f_dc.astype(np.float32, copy=False).reshape(-1)  # 3*N
    if sample_size > 0 and values.size > sample_size:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(values.size, size=int(sample_size), replace=False)
        values = values[idx]

    qs = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    codebook = np.quantile(values.astype(np.float64), qs).astype(np.float32)
    # 保证单调非递减,避免 searchsorted 出现异常抖动.
    codebook = np.maximum.accumulate(codebook).astype(np.float32)
    if codebook.shape != (256,):
        raise RuntimeError("internal error: sh0 codebook shape mismatch")
    return codebook


def _nearest_codebook_index(codebook: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    对单通道 float32 数组 `x`,返回最近的 codebook index(uint8).

    约束:
    - codebook 必须是长度 256 的非递减数组.
    """
    if codebook.shape != (256,):
        raise ValueError(f"codebook must be (256,), got {codebook.shape}")

    cb = codebook.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    idx = np.searchsorted(cb, x, side="left").astype(np.int32)

    idx_hi = np.clip(idx, 0, 255)
    idx_lo = np.clip(idx - 1, 0, 255)

    cb_hi = cb[idx_hi]
    cb_lo = cb[idx_lo]

    choose_lo = np.abs(x - cb_lo) <= np.abs(x - cb_hi)
    out = np.where(choose_lo, idx_lo, idx_hi).astype(np.uint8)
    return out


@dataclass(frozen=True)
class ScaleCodebookResult:
    codebook_lin: np.ndarray  # [K,3] float32
    indices_u16: np.ndarray  # [N] u16


@dataclass(frozen=True)
class ShNCodebookResult:
    """
    SH rest(v1) 的 palette + labels 结果.

    约束:
    - `labels_u16` 的取值范围必须满足 `[0, shn_count)`.
    - `centroids_f32` 的维度必须等于 `rest_coeff_count * 3`.
    """

    rest_coeff_count: int
    centroids_f32: np.ndarray  # [K,D] float32, D=rest_coeff_count*3
    labels_u16: np.ndarray  # [N] u16


def _build_scale_codebook(
    scale_log: np.ndarray,
    *,
    codebook_size: int,
    sample_size: int,
    seed: int,
    assign_chunk: int,
) -> ScaleCodebookResult:
    """
    用 kmeans2 在 log-domain 拟合 scale codebook,并对全量 splat 分配 u16 index.

    重要:
    - `.sog4d` 的 scale codebook 是线性值,但聚类建议在 log-domain(更稳).
    """
    if not (1 <= codebook_size <= 65535):
        raise ValueError(f"--scale-codebook-size must be in [1,65535], got {codebook_size}")
    if sample_size <= 0:
        raise ValueError(f"--scale-codebook-sample must be > 0, got {sample_size}")
    if assign_chunk <= 0:
        raise ValueError(f"--assign-chunk must be > 0, got {assign_chunk}")

    x = scale_log.astype(np.float32, copy=False)
    n = int(x.shape[0])
    if x.shape != (n, 3):
        raise ValueError(f"scale_log must be [N,3], got {x.shape}")

    rng = np.random.default_rng(int(seed))
    sample_n = min(int(sample_size), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    sample = x[sample_idx]

    print(f"[sog4d] scale kmeans: sample={sample_n:,}, K={codebook_size}")
    centroids, _ = kmeans2(sample, int(codebook_size), iter=20, minit="points")
    centroids = centroids.astype(np.float32)

    # 3 维空间用 KDTree 分配最近中心,非常快.
    tree = cKDTree(centroids)
    indices = np.empty((n,), dtype=np.uint16)
    for start in range(0, n, int(assign_chunk)):
        end = min(start + int(assign_chunk), n)
        _, idx = tree.query(x[start:end], k=1, workers=-1)
        indices[start:end] = idx.astype(np.uint16)

    codebook_lin = np.exp(centroids).astype(np.float32)
    return ScaleCodebookResult(codebook_lin=codebook_lin, indices_u16=indices)


def _flatten_sh_rest_v1(
    shn: np.ndarray,
    *,
    bands: int,
) -> Tuple[np.ndarray, int]:
    """
    把 checkpoint 的 `shN[N,K,3]` 按 v1 规则展平成 `[N, restCoeffCount*3]`.

    备注:
    - `bands` 决定 restCoeffCount:
      restCoeffCount = (bands + 1)^2 - 1
    - 这里假设 `shN` 的 coeff 顺序已经与 Unity importer 对齐(通常是 l 从小到大).
    """
    if not (1 <= bands <= 3):
        raise ValueError(f"bands must be in [1,3], got {bands}")

    shn = shn.astype(np.float32, copy=False)
    n = int(shn.shape[0])
    if shn.ndim != 3 or shn.shape[0] != n or shn.shape[2] != 3:
        raise ValueError(f"shN must be [N,K,3], got shape={shn.shape}")

    rest_coeff_count = int((bands + 1) ** 2 - 1)
    if shn.shape[1] < rest_coeff_count:
        raise ValueError(
            f"shN coeff count too small: need >= {rest_coeff_count}, got {shn.shape[1]}"
        )

    # 注意: [:, :rest_coeff_count, :] 从 0 开始取连续前缀,通常不会触发额外拷贝.
    sh_rest = shn[:, :rest_coeff_count, :]
    sh_rest_flat = sh_rest.reshape(n, rest_coeff_count * 3)
    return sh_rest_flat, rest_coeff_count


def _flatten_sh_rest_v2_per_band(
    shn: np.ndarray,
    *,
    bands: int,
) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    v2: 把 checkpoint 的 `shN[N,K,3]` 按 band 拆分成多个低维向量,用于 per-band kmeans.

    返回:
    - dict,按顺序包含(取决于 bands):
      - "sh1": ([N, 3*3], coeffCount=3)
      - "sh2": ([N, 5*3], coeffCount=5)
      - "sh3": ([N, 7*3], coeffCount=7)

    说明:
    - 这里假设 `shN` 的 coeff 顺序与 Unity importer 对齐:
      l 从小到大,并且 rest coeff 的顺序为:
        sh1(3) + sh2(5) + sh3(7) = 15(当 bands=3)
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

    # sh1: l=1, coeffCount=3
    if bands >= 1:
        coeff_count = 3
        band = sh_rest[:, offset : offset + coeff_count, :]
        out["sh1"] = (band.reshape(n, coeff_count * 3), coeff_count)
        offset += coeff_count

    # sh2: l=2, coeffCount=5
    if bands >= 2:
        coeff_count = 5
        band = sh_rest[:, offset : offset + coeff_count, :]
        out["sh2"] = (band.reshape(n, coeff_count * 3), coeff_count)
        offset += coeff_count

    # sh3: l=3, coeffCount=7
    if bands >= 3:
        coeff_count = 7
        band = sh_rest[:, offset : offset + coeff_count, :]
        out["sh3"] = (band.reshape(n, coeff_count * 3), coeff_count)
        offset += coeff_count

    if offset != rest_coeff_total:
        raise RuntimeError("internal error: per-band SH rest coeff offset mismatch")

    return out


def _build_shn_v1_codebook_and_labels(
    sh_rest_flat: np.ndarray,
    *,
    name: str,
    shn_count: int,
    sample_size: int,
    seed: int,
    assign_chunk: int,
    kmeans_iters: int,
) -> ShNCodebookResult:
    """
    v1: 用单一 palette 量化 SH rest.

    实现要点:
    - 先对 `sh_rest_flat` 做 kmeans2 得到 centroids.
    - 再用 KDTree 为全量 splat 分配 labels(u16).
    """
    if not (1 <= shn_count <= 65535):
        raise ValueError(f"--shn-count must be in [1,65535], got {shn_count}")
    if sample_size <= 0:
        raise ValueError(f"--shn-codebook-sample must be > 0, got {sample_size}")
    if assign_chunk <= 0:
        raise ValueError(f"--shn-assign-chunk must be > 0, got {assign_chunk}")
    if kmeans_iters <= 0:
        raise ValueError(f"--shn-kmeans-iters must be > 0, got {kmeans_iters}")

    x = sh_rest_flat.astype(np.float32, copy=False)
    n = int(x.shape[0])
    d = int(x.shape[1])
    if x.shape != (n, d):
        raise ValueError("internal error: sh_rest_flat shape mismatch")
    if d % 3 != 0:
        raise ValueError(f"internal error: sh_rest_flat dim must be multiple of 3, got {d}")

    rest_coeff_count = d // 3

    rng = np.random.default_rng(int(seed))
    sample_n = min(int(sample_size), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    sample = x[sample_idx]

    # scipy 的 kmeans2 可能出现 empty cluster,会导致码字浪费/质量变差.
    # 这里做一个小重试,并且用可复现的 seed 控制初始化.
    print(f"[sog4d] {name} kmeans: sample={sample_n:,}, K={shn_count}, D={d}")
    centroids: Optional[np.ndarray] = None
    for attempt in range(3):
        attempt_seed = int(seed) + int(attempt)

        # kmeans2 内部依赖 numpy 的全局 RNG,我们临时设置,并在 finally 恢复,避免污染外部随机状态.
        rng_state = np.random.get_state()
        try:
            np.random.seed(int(attempt_seed))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                c_try, labels_try = kmeans2(
                    sample,
                    int(shn_count),
                    iter=int(kmeans_iters),
                    minit="++",
                )
        finally:
            np.random.set_state(rng_state)

        # 经验上: warning + unique(label)<K 基本等价于 empty cluster.
        warned_empty = any("clusters is empty" in str(w.message) for w in caught)
        used = int(np.unique(labels_try).size)
        centroids = c_try

        if (not warned_empty) and used == int(shn_count):
            break
        if attempt == 0:
            print(f"[sog4d] {name} kmeans: empty cluster detected, retrying...")
        print(f"[sog4d] {name} kmeans retry {attempt+1}/3: used={used}/{int(shn_count)} seed={attempt_seed}")

    if centroids is None:
        raise RuntimeError("internal error: kmeans2 returned no centroids")

    centroids_f32 = centroids.astype(np.float32, copy=False)

    # 高维数据 KDTree 可能会退化,但在 K 不太大时仍然是最省内存的稳妥做法.
    tree = cKDTree(centroids_f32)
    labels = np.empty((n,), dtype=np.uint16)
    for start in range(0, n, int(assign_chunk)):
        end = min(start + int(assign_chunk), n)
        _, idx = tree.query(x[start:end], k=1, workers=-1)
        labels[start:end] = idx.astype(np.uint16)

    return ShNCodebookResult(
        rest_coeff_count=int(rest_coeff_count),
        centroids_f32=centroids_f32,
        labels_u16=labels,
    )


def _make_rgba_image_from_u16_labels(
    labels_u16: np.ndarray,
    *,
    pixel_count: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    把 u16 labels 写成 RGBA8 WebP 数据图:
    - RG: label 的小端(u16)
    - B: 0
    - A: 255
    """
    labels_u16 = labels_u16.astype(np.uint16, copy=False)
    n = int(labels_u16.shape[0])
    if labels_u16.shape != (n,):
        raise ValueError(f"labels_u16 must be [N], got {labels_u16.shape}")

    flat = np.zeros((pixel_count, 4), dtype=np.uint8)
    flat[:, 3] = np.uint8(255)
    flat[:n, 0] = (labels_u16 & np.uint16(255)).astype(np.uint8)
    flat[:n, 1] = (labels_u16 >> np.uint16(8)).astype(np.uint8)
    return flat.reshape(height, width, 4)


def _build_label_delta_v1_static(
    *,
    segment_start_frame: int,
    segment_frame_count: int,
    splat_count: int,
    label_count: int,
) -> bytes:
    """
    生成 delta-v1 的最小实现: segment 内所有后续帧都无更新(updateCount=0).

    这非常贴合 FreeTimeGS 的常见性质: SH 系数通常静态,labels 跨帧一致.
    """
    if segment_start_frame < 0:
        raise ValueError("segment_start_frame must be >= 0")
    if segment_frame_count <= 0:
        raise ValueError("segment_frame_count must be > 0")
    if splat_count <= 0:
        raise ValueError("splat_count must be > 0")
    if not (1 <= label_count <= 65535):
        raise ValueError("label_count must be in [1,65535]")

    header = b"SOG4DLB1" + struct.pack(
        "<5I",
        1,  # version
        int(segment_start_frame),
        int(segment_frame_count),
        int(splat_count),
        int(label_count),
    )
    body = struct.pack("<I", 0) * int(max(segment_frame_count - 1, 0))
    return header + body


def _build_delta_segments(
    *,
    frame_count: int,
    segment_length: int,
) -> List[Tuple[int, int]]:
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


def _apply_global_filter(
    *,
    means: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    durations: np.ndarray,
    opacities_logit: np.ndarray,
    scales_log: np.ndarray,
    quats: np.ndarray,
    sh0: np.ndarray,
    base_opacity_threshold: float,
    max_splats: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    对所有帧一致的全局裁剪,以满足 splat identity 稳定要求.

    返回:
    - filtered arrays dict
    - keep mask(bool),用于记录
    """
    base_opacity = _stable_sigmoid(opacities_logit)
    keep = np.ones((base_opacity.shape[0],), dtype=bool)

    if base_opacity_threshold > 0.0:
        keep &= base_opacity >= float(base_opacity_threshold)

    if max_splats > 0:
        max_splats = int(max_splats)
        idx_keep = np.nonzero(keep)[0]
        if idx_keep.size > max_splats:
            # 只在 keep 内取 top-k,避免“先阈值再 topk”时被阈值干扰.
            scores = base_opacity[idx_keep]
            topk_local = np.argpartition(-scores, max_splats - 1)[:max_splats]
            chosen = np.zeros_like(keep)
            chosen[idx_keep[topk_local]] = True
            keep = chosen

    kept = int(keep.sum())
    print(f"[sog4d] global filter: keep {kept:,}/{keep.size:,}")

    arrays: Dict[str, np.ndarray] = {
        "means": means[keep],
        "velocities": velocities[keep],
        "times": times[keep],
        "durations": durations[keep],
        "opacities_logit": opacities_logit[keep],
        "scales_log": scales_log[keep],
        "quats": quats[keep],
        "sh0": sh0[keep],
    }
    return arrays, keep


def export_sog4d_from_ckpt(
    *,
    ckpt_path: Path,
    output_path: Path,
    frame_count: int,
    time_mapping: TimeMappingType,
    explicit_times: Optional[List[float]],
    layout_width: int,
    min_sigma: float,
    base_opacity_threshold: float,
    max_splats: int,
    alpha_zero_threshold: float,
    webp_method: int,
    webp_quality: int,
    zip_compression: ZipCompression,
    scale_codebook_size: int,
    scale_codebook_sample: int,
    sh0_codebook_sample: int,
    sh_bands: int,
    sh_version: int,
    shn_count: int,
    shn_centroids_type: Literal["f16", "f32"],
    shn_labels_encoding: Literal["full", "delta-v1"],
    delta_segment_length: int,
    shn_codebook_sample: int,
    shn_assign_chunk: int,
    shn_kmeans_iters: int,
    seed: int,
    assign_chunk: int,
    overwrite: bool,
) -> None:
    if frame_count <= 0:
        raise ValueError("--frame-count must be > 0")
    if not (0.0 <= base_opacity_threshold <= 1.0):
        raise ValueError("--base-opacity-threshold must be in [0,1]")
    if max_splats < 0:
        raise ValueError("--max-splats must be >= 0")
    if min_sigma <= 0.0:
        raise ValueError("--min-sigma must be > 0")
    if not (0.0 <= alpha_zero_threshold <= 1.0):
        raise ValueError("--alpha-zero-threshold must be in [0,1]")
    if not (0 <= webp_method <= 6):
        raise ValueError("--webp-method must be in [0,6]")
    if not (0 <= webp_quality <= 100):
        raise ValueError("--webp-quality must be in [0,100]")
    if not (0 <= sh_bands <= 3):
        raise ValueError("--sh-bands must be in [0,3]")
    if sh_version not in (1, 2):
        raise ValueError("--sh-version must be 1 or 2")
    if delta_segment_length < 0:
        raise ValueError("--delta-segment-length must be >= 0")

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(f"output exists: {output_path} (use --overwrite)")

    if output_path.suffix.lower() != ".sog4d":
        raise ValueError(f"output must end with .sog4d, got: {output_path}")

    # -----------------------------
    # 1) 读取 checkpoint
    # -----------------------------
    print(f"[sog4d] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    required = ["means", "scales", "quats", "opacities", "sh0", "times", "durations", "velocities"]
    if sh_bands > 0:
        required.append("shN")
    missing = [k for k in required if k not in splats]
    if missing:
        raise KeyError(f"checkpoint missing splats keys: {missing}")

    means = _as_numpy_f32(splats["means"])  # [N,3]
    scales_log = _as_numpy_f32(splats["scales"])  # [N,3] log scale
    quats = _as_numpy_f32(splats["quats"])  # [N,4] wxyz
    opacities_logit = _as_numpy_f32(splats["opacities"]).reshape(-1)  # [N]
    sh0 = _as_numpy_f32(splats["sh0"]).reshape(-1, 3)  # [N,3] f_dc
    times = _as_numpy_f32(splats["times"]).reshape(-1)  # [N]
    durations_log = _as_numpy_f32(splats["durations"]).reshape(-1)  # [N] log(sigma)
    velocities = _as_numpy_f32(splats["velocities"])  # [N,3]

    n_total = int(means.shape[0])
    print(f"[sog4d] gaussians: {n_total:,}")

    # -----------------------------
    # 2) 全局裁剪(可选,必须对所有帧一致)
    # -----------------------------
    if base_opacity_threshold > 0.0 or max_splats > 0:
        arrays, keep = _apply_global_filter(
            means=means,
            velocities=velocities,
            times=times,
            durations=durations_log,
            opacities_logit=opacities_logit,
            scales_log=scales_log,
            quats=quats,
            sh0=sh0,
            base_opacity_threshold=base_opacity_threshold,
            max_splats=max_splats,
        )
        means = arrays["means"]
        velocities = arrays["velocities"]
        times = arrays["times"]
        durations_log = arrays["durations"]
        opacities_logit = arrays["opacities_logit"]
        scales_log = arrays["scales_log"]
        quats = arrays["quats"]
        sh0 = arrays["sh0"]
        n_total = int(means.shape[0])
    else:
        keep = np.ones((n_total,), dtype=bool)

    # -----------------------------
    # 3) 预计算静态参数(scale/rotation/sh0 codebook)
    # -----------------------------
    base_opacity = _stable_sigmoid(opacities_logit)  # [N]
    sigma = np.exp(durations_log.astype(np.float32, copy=False))
    sigma = np.clip(sigma, float(min_sigma), np.inf).astype(np.float32)

    # layout
    width, height, pixel_count = _build_layout(splat_count=n_total, width=int(layout_width))
    print(f"[sog4d] layout: {width}x{height} (pixels={pixel_count:,}, splats={n_total:,})")

    # scale codebook + indices
    scale_res = _build_scale_codebook(
        scales_log,
        codebook_size=int(scale_codebook_size),
        sample_size=int(scale_codebook_sample),
        seed=int(seed),
        assign_chunk=int(assign_chunk),
    )
    if scale_res.codebook_lin.shape[0] != int(scale_codebook_size):
        raise RuntimeError("internal error: scale codebook size mismatch")

    # rotation u8 (静态)
    quat_u8 = _quantize_quat_to_u8(_normalize_quat_wxyz(quats))  # [N,4]

    # sh0 codebook + RGB 索引(静态)
    sh0_codebook = _build_sh0_codebook(sh0, sample_size=int(sh0_codebook_sample), seed=int(seed))
    sh0_r = _nearest_codebook_index(sh0_codebook, sh0[:, 0])
    sh0_g = _nearest_codebook_index(sh0_codebook, sh0[:, 1])
    sh0_b = _nearest_codebook_index(sh0_codebook, sh0[:, 2])
    sh0_rgb = np.stack([sh0_r, sh0_g, sh0_b], axis=1).astype(np.uint8)  # [N,3]

    # SH rest 的 palette + labels(静态).
    # 说明:
    # - FreeTimeGS 的 shN 通常不随时间变化,因此 labels 跨帧一致.
    # - 当你希望更贴近 DualGS 一类的“多 codebook”思路时,可以用 `--sh-version 2` 导出 per-band(sh1/sh2/sh3).
    sh_delta_segments: Optional[List[Tuple[int, int]]] = None
    if sh_bands > 0 and shn_labels_encoding == "delta-v1":
        seg_len = int(frame_count) if int(delta_segment_length) == 0 else int(delta_segment_length)
        seg_len = min(int(seg_len), int(frame_count))
        sh_delta_segments = _build_delta_segments(frame_count=int(frame_count), segment_length=int(seg_len))
        print(f"[sog4d] sh delta segments: count={len(sh_delta_segments)}, segment_length={seg_len}")

    # v1(single shN palette)
    shn_res: Optional[ShNCodebookResult] = None
    shn_centroids_bytes: Optional[bytes] = None
    shn_labels_webp_bytes: Optional[bytes] = None

    # v2(per-band palettes)
    sh_bands_v2_res: Dict[str, ShNCodebookResult] = {}
    sh_bands_v2_centroids_bytes: Dict[str, bytes] = {}
    sh_bands_v2_labels_webp_bytes: Dict[str, bytes] = {}

    if sh_bands > 0:
        shn = _as_numpy_f32(splats["shN"])  # [N,K,3]
        # 注意: 全局裁剪必须对所有属性一致,因此 shN 也必须跟随 keep 掩码裁剪.
        shn = shn[keep]

        if int(sh_version) == 1:
            sh_rest_flat, _rest_coeff_count = _flatten_sh_rest_v1(shn, bands=int(sh_bands))
            shn_res = _build_shn_v1_codebook_and_labels(
                sh_rest_flat,
                name="shN",
                shn_count=int(shn_count),
                sample_size=int(shn_codebook_sample),
                seed=int(seed),
                assign_chunk=int(shn_assign_chunk),
                kmeans_iters=int(shn_kmeans_iters),
            )

            # centroids: raw binary (little-endian,无 header)
            if shn_centroids_type == "f16":
                shn_centroids_bytes = shn_res.centroids_f32.astype("<f2", copy=False).tobytes(order="C")
            elif shn_centroids_type == "f32":
                shn_centroids_bytes = shn_res.centroids_f32.astype("<f4", copy=False).tobytes(order="C")
            else:  # pragma: no cover
                raise ValueError(f"unknown shn_centroids_type: {shn_centroids_type}")

            # labels: 只要编码一次,后续按 full/delta-v1 复用.
            shn_labels_img = _make_rgba_image_from_u16_labels(
                shn_res.labels_u16,
                pixel_count=pixel_count,
                width=width,
                height=height,
            )
            shn_labels_webp_bytes = _encode_webp_bytes_rgba(
                shn_labels_img, webp_method=int(webp_method), webp_quality=int(webp_quality)
            )
        else:
            # v2: per-band(sh1/sh2/sh3)分别拟合 codebook + labels.
            per_band = _flatten_sh_rest_v2_per_band(shn, bands=int(sh_bands))
            for band_idx, (band_name, (band_flat, _coeff_count)) in enumerate(per_band.items()):
                band_seed = int(seed) + int(band_idx)
                res = _build_shn_v1_codebook_and_labels(
                    band_flat,
                    name=str(band_name),
                    shn_count=int(shn_count),
                    sample_size=int(shn_codebook_sample),
                    seed=int(band_seed),
                    assign_chunk=int(shn_assign_chunk),
                    kmeans_iters=int(shn_kmeans_iters),
                )
                sh_bands_v2_res[str(band_name)] = res

                if shn_centroids_type == "f16":
                    sh_bands_v2_centroids_bytes[str(band_name)] = res.centroids_f32.astype(
                        "<f2", copy=False
                    ).tobytes(order="C")
                elif shn_centroids_type == "f32":
                    sh_bands_v2_centroids_bytes[str(band_name)] = res.centroids_f32.astype(
                        "<f4", copy=False
                    ).tobytes(order="C")
                else:  # pragma: no cover
                    raise ValueError(f"unknown shn_centroids_type: {shn_centroids_type}")

                band_labels_img = _make_rgba_image_from_u16_labels(
                    res.labels_u16,
                    pixel_count=pixel_count,
                    width=width,
                    height=height,
                )
                sh_bands_v2_labels_webp_bytes[str(band_name)] = _encode_webp_bytes_rgba(
                    band_labels_img, webp_method=int(webp_method), webp_quality=int(webp_quality)
                )

    # -----------------------------
    # 4) 预编码静态 WebP(避免每帧重复编码)
    # -----------------------------
    scale_rg = np.zeros((n_total, 3), dtype=np.uint8)
    idx_u16 = scale_res.indices_u16.astype(np.uint16, copy=False)
    scale_rg[:, 0] = (idx_u16 & np.uint16(255)).astype(np.uint8)
    scale_rg[:, 1] = (idx_u16 >> np.uint16(8)).astype(np.uint8)
    scale_img = _make_rgba_image_from_rgb_and_a(
        rgb_u8=scale_rg,
        a_u8=np.full((n_total,), 255, dtype=np.uint8),
        pixel_count=pixel_count,
        width=width,
        height=height,
        default_a=255,
    )
    scale_webp_bytes = _encode_webp_bytes_rgba(
        scale_img, webp_method=int(webp_method), webp_quality=int(webp_quality)
    )

    rot_img = _make_rgba_image_from_rgba_flat(
        rgba_u8=quat_u8,
        pixel_count=pixel_count,
        width=width,
        height=height,
        default_a=255,
    )
    rot_webp_bytes = _encode_webp_bytes_rgba(
        rot_img, webp_method=int(webp_method), webp_quality=int(webp_quality)
    )

    # -----------------------------
    # 5) 时间映射
    # -----------------------------
    if time_mapping == "uniform":
        if frame_count == 1:
            frame_times = [0.0]
        else:
            frame_times = [i / float(frame_count - 1) for i in range(frame_count)]
    elif time_mapping == "explicit":
        if explicit_times is None:
            raise ValueError("--time-mapping explicit requires --frame-times")
        if len(explicit_times) != frame_count:
            raise ValueError("--frame-times length must equal --frame-count")
        if any((t < 0.0 or t > 1.0) for t in explicit_times):
            raise ValueError("--frame-times must be within [0,1]")
        if any(explicit_times[i] > explicit_times[i + 1] for i in range(frame_count - 1)):
            raise ValueError("--frame-times must be non-decreasing")
        frame_times = [float(t) for t in explicit_times]
    else:  # pragma: no cover
        raise ValueError(f"unknown time_mapping: {time_mapping}")

    # -----------------------------
    # 6) 写 zip bundle
    # -----------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    if zip_compression == "stored":
        zip_comp = zipfile.ZIP_STORED
        zip_kwargs: Dict[str, object] = {}
    elif zip_compression == "deflated":
        zip_comp = zipfile.ZIP_DEFLATED
        zip_kwargs = {"compresslevel": 6}
    else:  # pragma: no cover
        raise ValueError(f"unknown zip_compression: {zip_compression}")

    range_min_list: List[List[float]] = []
    range_max_list: List[List[float]] = []

    eps = np.float32(1e-8)
    with zipfile.ZipFile(tmp_path, mode="w", compression=zip_comp, **zip_kwargs) as zf:
        # -----------------------------
        # SH rest 静态文件: centroids + labels(+delta)
        # -----------------------------
        if shn_res is not None:
            # v1: 单一 shN palette
            # centroids 文件放在 zip 根目录,路径在 meta.json 里引用.
            if shn_centroids_bytes is None or shn_labels_webp_bytes is None:
                raise RuntimeError("internal error: shN centroids/labels bytes missing")

            zf.writestr("shN_centroids.bin", shn_centroids_bytes)

            if shn_labels_encoding == "full":
                # full: 每帧都要有一张 labels WebP.
                for frame_idx in range(int(frame_count)):
                    frame_str = _format_frame(frame_idx)
                    zf.writestr(f"frames/{frame_str}/shN_labels.webp", shn_labels_webp_bytes)
            elif shn_labels_encoding == "delta-v1":
                # delta-v1: 每个 segment 写 base labels(首帧) + delta 文件.
                if sh_delta_segments is None:
                    raise RuntimeError("internal error: sh delta segments missing")
                for seg_start, seg_count in sh_delta_segments:
                    frame_str = _format_frame(int(seg_start))
                    zf.writestr(f"frames/{frame_str}/shN_labels.webp", shn_labels_webp_bytes)
                    delta_bytes = _build_label_delta_v1_static(
                        segment_start_frame=int(seg_start),
                        segment_frame_count=int(seg_count),
                        splat_count=int(n_total),
                        label_count=int(shn_count),
                    )
                    zf.writestr(f"sh/shN_delta_{frame_str}.bin", delta_bytes)
            else:  # pragma: no cover
                raise ValueError(f"unknown shn_labels_encoding: {shn_labels_encoding}")

        if sh_bands_v2_res:
            # v2: per-band palettes(sh1/sh2/sh3)
            for band_name, _res in sh_bands_v2_res.items():
                centroids_bytes = sh_bands_v2_centroids_bytes.get(str(band_name))
                labels_bytes = sh_bands_v2_labels_webp_bytes.get(str(band_name))
                if centroids_bytes is None or labels_bytes is None:
                    raise RuntimeError("internal error: sh band centroids/labels bytes missing")
                zf.writestr(f"sh/{band_name}_centroids.bin", centroids_bytes)

            if shn_labels_encoding == "full":
                for frame_idx in range(int(frame_count)):
                    frame_str = _format_frame(frame_idx)
                    for band_name in sh_bands_v2_res.keys():
                        labels_bytes = sh_bands_v2_labels_webp_bytes[str(band_name)]
                        zf.writestr(f"frames/{frame_str}/{band_name}_labels.webp", labels_bytes)
            elif shn_labels_encoding == "delta-v1":
                if sh_delta_segments is None:
                    raise RuntimeError("internal error: sh delta segments missing")
                for seg_start, seg_count in sh_delta_segments:
                    frame_str = _format_frame(int(seg_start))
                    for band_name in sh_bands_v2_res.keys():
                        labels_bytes = sh_bands_v2_labels_webp_bytes[str(band_name)]
                        zf.writestr(f"frames/{frame_str}/{band_name}_labels.webp", labels_bytes)
                        delta_bytes = _build_label_delta_v1_static(
                            segment_start_frame=int(seg_start),
                            segment_frame_count=int(seg_count),
                            splat_count=int(n_total),
                            label_count=int(shn_count),
                        )
                        zf.writestr(f"sh/{band_name}_delta_{frame_str}.bin", delta_bytes)
            else:  # pragma: no cover
                raise ValueError(f"unknown shn_labels_encoding: {shn_labels_encoding}")

        # per-frame files
        for frame_idx, t in enumerate(frame_times):
            frame_str = _format_frame(frame_idx)
            base_dir = f"frames/{frame_str}"

            t_f = np.float32(t)
            dt = (t_f - times.astype(np.float32, copy=False)).astype(np.float32)  # [N]

            # ---- position(t) ----
            pos = means + velocities * dt.reshape(-1, 1)  # [N,3]
            pos_min = pos.min(axis=0).astype(np.float32)
            pos_max = pos.max(axis=0).astype(np.float32)
            range_min_list.append([float(pos_min[0]), float(pos_min[1]), float(pos_min[2])])
            range_max_list.append([float(pos_max[0]), float(pos_max[1]), float(pos_max[2])])

            hi, lo = _quantize_u16_hi_lo(pos, range_min=pos_min, range_max=pos_max)  # [N,3] u8

            # position_hi.webp
            hi_flat = np.zeros((pixel_count, 4), dtype=np.uint8)
            hi_flat[:, 3] = np.uint8(255)
            hi_flat[:n_total, 0:3] = hi
            hi_img = hi_flat.reshape(height, width, 4)
            _write_webp_rgba_to_zip(
                zf,
                path_in_zip=f"{base_dir}/position_hi.webp",
                img_rgba_u8=hi_img,
                webp_method=int(webp_method),
                webp_quality=int(webp_quality),
            )

            # position_lo.webp
            lo_flat = np.zeros((pixel_count, 4), dtype=np.uint8)
            lo_flat[:, 3] = np.uint8(255)
            lo_flat[:n_total, 0:3] = lo
            lo_img = lo_flat.reshape(height, width, 4)
            _write_webp_rgba_to_zip(
                zf,
                path_in_zip=f"{base_dir}/position_lo.webp",
                img_rgba_u8=lo_img,
                webp_method=int(webp_method),
                webp_quality=int(webp_quality),
            )

            # ---- scale/rotation(静态,直接复用 bytes) ----
            zf.writestr(f"{base_dir}/scale_indices.webp", scale_webp_bytes)
            zf.writestr(f"{base_dir}/rotation.webp", rot_webp_bytes)

            # ---- opacity(t) 写入 sh0.webp alpha ----
            # temporal_opacity(t) = exp(-0.5 * ((t - mu_t)/(sigma+eps))^2)
            temporal = np.exp(-0.5 * np.square(dt / (sigma + eps))).astype(np.float32)
            opacity_t = (base_opacity * temporal).astype(np.float32)
            opacity_t = np.clip(opacity_t, 0.0, 1.0)
            if alpha_zero_threshold > 0.0:
                opacity_t = np.where(opacity_t < float(alpha_zero_threshold), 0.0, opacity_t)
            alpha_u8 = np.clip(np.round(opacity_t * 255.0), 0, 255).astype(np.uint8)

            sh0_img = _make_rgba_image_from_rgb_and_a(
                rgb_u8=sh0_rgb,
                a_u8=alpha_u8,
                pixel_count=pixel_count,
                width=width,
                height=height,
                default_a=0,
            )
            _write_webp_rgba_to_zip(
                zf,
                path_in_zip=f"{base_dir}/sh0.webp",
                img_rgba_u8=sh0_img,
                webp_method=int(webp_method),
                webp_quality=int(webp_quality),
            )

            print(
                f"[sog4d] frame {frame_idx+1}/{frame_count} "
                f"t={float(t_f):.6f} posRange=([{pos_min[0]:.3f},{pos_min[1]:.3f},{pos_min[2]:.3f}],"
                f"[{pos_max[0]:.3f},{pos_max[1]:.3f},{pos_max[2]:.3f}])"
            )

        # meta.json 最后写入,避免提前占位导致“需要二次重写 zip”.
        meta_version = 2 if (int(sh_bands) > 0 and int(sh_version) == 2) else 1

        sh_stream: Dict[str, object] = {
            "bands": int(sh_bands),
            "sh0Path": "frames/{frame}/sh0.webp",
            "sh0Codebook": sh0_codebook.astype(np.float32).tolist(),
        }

        if meta_version == 1 and shn_res is not None:
            # v1: 单一 shN palette
            sh_stream.update(
                {
                    "shNCount": int(shn_count),
                    "shNCentroidsType": str(shn_centroids_type),
                    "shNCentroidsPath": "shN_centroids.bin",
                    "shNLabelsEncoding": str(shn_labels_encoding),
                }
            )
            if shn_labels_encoding == "full":
                sh_stream["shNLabelsPath"] = "frames/{frame}/shN_labels.webp"
            elif shn_labels_encoding == "delta-v1":
                if sh_delta_segments is None:
                    raise RuntimeError("internal error: sh delta segments missing")
                segments: List[Dict[str, object]] = []
                for seg_start, seg_count in sh_delta_segments:
                    frame_str = _format_frame(int(seg_start))
                    segments.append(
                        {
                            "startFrame": int(seg_start),
                            "frameCount": int(seg_count),
                            "baseLabelsPath": f"frames/{frame_str}/shN_labels.webp",
                            "deltaPath": f"sh/shN_delta_{frame_str}.bin",
                        }
                    )
                sh_stream["shNDeltaSegments"] = segments
            else:  # pragma: no cover
                raise ValueError(f"unknown shn_labels_encoding: {shn_labels_encoding}")

        if meta_version == 2:
            # v2: per-band palettes(sh1/sh2/sh3)
            if not sh_bands_v2_res:
                raise RuntimeError("internal error: sh per-band results missing")

            for band_name in ("sh1", "sh2", "sh3"):
                if band_name not in sh_bands_v2_res:
                    continue

                band_stream: Dict[str, object] = {
                    "count": int(shn_count),
                    "centroidsType": str(shn_centroids_type),
                    "centroidsPath": f"sh/{band_name}_centroids.bin",
                    "labelsEncoding": str(shn_labels_encoding),
                }
                if shn_labels_encoding == "full":
                    band_stream["labelsPath"] = f"frames/{{frame}}/{band_name}_labels.webp"
                elif shn_labels_encoding == "delta-v1":
                    if sh_delta_segments is None:
                        raise RuntimeError("internal error: sh delta segments missing")
                    segments: List[Dict[str, object]] = []
                    for seg_start, seg_count in sh_delta_segments:
                        frame_str = _format_frame(int(seg_start))
                        segments.append(
                            {
                                "startFrame": int(seg_start),
                                "frameCount": int(seg_count),
                                "baseLabelsPath": f"frames/{frame_str}/{band_name}_labels.webp",
                                "deltaPath": f"sh/{band_name}_delta_{frame_str}.bin",
                            }
                        )
                    band_stream["deltaSegments"] = segments
                else:  # pragma: no cover
                    raise ValueError(f"unknown shn_labels_encoding: {shn_labels_encoding}")

                sh_stream[band_name] = band_stream

        meta: Dict[str, object] = {
            "version": int(meta_version),
            "splatCount": int(n_total),
            "frameCount": int(frame_count),
            "layout": {"type": "row-major", "width": int(width), "height": int(height)},
            "timeMapping": (
                {"type": "uniform"}
                if time_mapping == "uniform"
                else {"type": "explicit", "frameTimesNormalized": frame_times}
            ),
            "streams": {
                "position": {
                    "rangeMin": range_min_list,
                    "rangeMax": range_max_list,
                    "hiPath": "frames/{frame}/position_hi.webp",
                    "loPath": "frames/{frame}/position_lo.webp",
                },
                "scale": {
                    "codebook": scale_res.codebook_lin.astype(np.float32).tolist(),
                    "indicesPath": "frames/{frame}/scale_indices.webp",
                },
                "rotation": {"path": "frames/{frame}/rotation.webp"},
                "sh": sh_stream,
            },
            "generator": {
                "tool": "FreeTimeGsVanilla/tools/exportor/export_sog4d.py",
                "ckptPath": str(ckpt_path),
                "cwd": os.getcwd(),
                "baseOpacityThreshold": float(base_opacity_threshold),
                "maxSplats": int(max_splats),
                "minSigma": float(min_sigma),
                "alphaZeroThreshold": float(alpha_zero_threshold),
                "zipCompression": str(zip_compression),
                "shBands": int(sh_bands),
                "shVersion": int(sh_version),
                "shNCount": int(shn_count),
                "shNCentroidsType": str(shn_centroids_type),
                "shNLabelsEncoding": str(shn_labels_encoding),
                "deltaSegmentLength": int(delta_segment_length),
                "shNCodebookSample": int(shn_codebook_sample),
                "shNAssignChunk": int(shn_assign_chunk),
                "shNKmeansIters": int(shn_kmeans_iters),
            },
        }
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

    tmp_path.replace(output_path)
    print(f"[sog4d] wrote: {output_path}")


def _parse_frame_times(value: str) -> List[float]:
    """
    支持两种写法:
    - CSV: "0,0.1,0.2"
    - 文件: "/path/to/times.txt"(每行一个 float)
    """
    p = Path(value)
    if p.exists() and p.is_file():
        out: List[float] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            out.append(float(s))
        return out

    # fallback: CSV
    parts = [x.strip() for x in value.split(",") if x.strip()]
    return [float(x) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FreeTimeGS checkpoint(.pt) to .sog4d")
    parser.add_argument("--ckpt-path", type=str, required=True, help="输入 checkpoint(.pt)")
    parser.add_argument("--output-path", type=str, required=True, help="输出 .sog4d 文件路径")
    parser.add_argument("--frame-count", type=int, default=61, help="导出帧数(默认 61)")
    parser.add_argument(
        "--time-mapping",
        type=str,
        default="uniform",
        choices=["uniform", "explicit"],
        help="时间映射模式(默认 uniform)",
    )
    parser.add_argument(
        "--frame-times",
        type=str,
        default=None,
        help="仅 explicit 模式使用: CSV 或文件路径(每行一个 float)",
    )
    parser.add_argument("--layout-width", type=int, default=2048, help="row-major layout 的 width")
    parser.add_argument("--min-sigma", type=float, default=0.02, help="temporal sigma 的最小值(对齐训练侧 clamp)")

    parser.add_argument(
        "--base-opacity-threshold",
        type=float,
        default=0.0,
        help="全局裁剪: base_opacity(sigmoid(logit)) < 阈值的 splat 直接丢弃",
    )
    parser.add_argument(
        "--max-splats",
        type=int,
        default=0,
        help="全局裁剪: 保留 base_opacity 最高的 top-k splats(0=不启用)",
    )
    parser.add_argument(
        "--alpha-zero-threshold",
        type=float,
        default=1.0 / 255.0,
        help="每帧: opacity(t) 小于该阈值则写 0(默认 1/255,对齐 Unity 运行时硬阈值)",
    )

    parser.add_argument("--webp-method", type=int, default=0, help="WebP method(0=最快,6=最慢但压缩最好)")
    parser.add_argument("--webp-quality", type=int, default=100, help="WebP quality(0-100,lossless 模式仍会用到)")
    parser.add_argument(
        "--zip-compression",
        type=str,
        default="stored",
        choices=["stored", "deflated"],
        help="zip 压缩方式(默认 stored)",
    )

    parser.add_argument("--scale-codebook-size", type=int, default=256, help="scale codebook 大小(K)")
    parser.add_argument(
        "--scale-codebook-sample",
        type=int,
        default=200_000,
        help="scale kmeans 的采样点数(越大越稳,但越慢)",
    )
    parser.add_argument(
        "--sh0-codebook-sample",
        type=int,
        default=0,
        help="sh0 codebook 的采样点数(0=用全量 3*N 标量,通常也不慢)",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子(用于采样与 kmeans 初始化)")
    parser.add_argument(
        "--assign-chunk",
        type=int,
        default=200_000,
        help="KDTree 分配 indices 的 chunk 大小(防止内存峰值过高)",
    )
    parser.add_argument(
        "--sh-bands",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="导出 SH 的 band 数(0=仅 sh0+opacity; 1..3=额外导出 SH rest palette+labels)",
    )
    parser.add_argument(
        "--sh-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="SH rest 的导出版本: 1=单一 shN palette(meta.version=1); 2=per-band(sh1/sh2/sh3,meta.version=2)",
    )
    parser.add_argument(
        "--shn-count",
        type=int,
        default=512,
        help="SH palette 的码字数量(1..65535).v1=shN; v2=每个 band 各一套.越大质量越好,但 kmeans/分配会显著更慢",
    )
    parser.add_argument(
        "--shn-centroids-type",
        type=str,
        default="f16",
        choices=["f16", "f32"],
        help="SH centroids.bin 的浮点精度(默认 f16,体积更小)",
    )
    parser.add_argument(
        "--shn-labels-encoding",
        type=str,
        default="delta-v1",
        choices=["full", "delta-v1"],
        help="labels 编码方式(默认 delta-v1,SH 静态时几乎零成本)",
    )
    parser.add_argument(
        "--delta-segment-length",
        type=int,
        default=0,
        help=(
            "仅当 labels-encoding=delta-v1 时生效: 每个 segment 的帧数(0=单段覆盖全帧,兼容旧输出)."
            "推荐值可参考 DualGS: 50"
        ),
    )
    parser.add_argument(
        "--shn-codebook-sample",
        type=int,
        default=100_000,
        help="SH kmeans 的采样点数(越大越稳,但越慢).高质量可调到 200k+",
    )
    parser.add_argument(
        "--shn-assign-chunk",
        type=int,
        default=50_000,
        help="SH KDTree 分配 labels 的 chunk 大小(高维时建议更小,避免内存峰值)",
    )
    parser.add_argument(
        "--shn-kmeans-iters",
        type=int,
        default=10,
        help="SH kmeans 迭代次数(越大越稳,但越慢).高质量可调到 20+",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")

    args = parser.parse_args()

    explicit_times = None
    if args.time_mapping == "explicit":
        if args.frame_times is None:
            raise ValueError("--time-mapping explicit requires --frame-times")
        explicit_times = _parse_frame_times(args.frame_times)

    export_sog4d_from_ckpt(
        ckpt_path=Path(args.ckpt_path),
        output_path=Path(args.output_path),
        frame_count=int(args.frame_count),
        time_mapping=args.time_mapping,
        explicit_times=explicit_times,
        layout_width=int(args.layout_width),
        min_sigma=float(args.min_sigma),
        base_opacity_threshold=float(args.base_opacity_threshold),
        max_splats=int(args.max_splats),
        alpha_zero_threshold=float(args.alpha_zero_threshold),
        webp_method=int(args.webp_method),
        webp_quality=int(args.webp_quality),
        zip_compression=args.zip_compression,
        scale_codebook_size=int(args.scale_codebook_size),
        scale_codebook_sample=int(args.scale_codebook_sample),
        sh0_codebook_sample=int(args.sh0_codebook_sample),
        sh_bands=int(args.sh_bands),
        sh_version=int(args.sh_version),
        shn_count=int(args.shn_count),
        shn_centroids_type=args.shn_centroids_type,
        shn_labels_encoding=args.shn_labels_encoding,
        delta_segment_length=int(args.delta_segment_length),
        shn_codebook_sample=int(args.shn_codebook_sample),
        shn_assign_chunk=int(args.shn_assign_chunk),
        shn_kmeans_iters=int(args.shn_kmeans_iters),
        seed=int(args.seed),
        assign_chunk=int(args.assign_chunk),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
