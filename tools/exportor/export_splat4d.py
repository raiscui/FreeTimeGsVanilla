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
import hashlib
import math
import struct
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

# 让脚本以 `python tools/exportor/export_splat4d.py ...` 方式运行时,
# 也能稳定 import 本仓库根目录下的模块(例如 `datasets.*`).
# 备注: 当以模块方式运行时(例如 `python -m tools.exportor.export_splat4d`),也不会有副作用.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
_SECT_XFRM: int = _fourcc("XFRM")  # 可选: 训练归一化 transform(4x4 f32),用于离线 debug


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


@dataclass(frozen=True)
class _DeltaV1BuildResult:
    """
    构建 delta-v1 的结果与统计.
    """

    delta_bytes: bytes
    total_updates: int
    max_update_count: int


def _build_label_delta_v1_from_labels_by_frame(
    labels_by_frame: List[np.ndarray],
    *,
    segment_start_frame: int,
    segment_frame_count: int,
    splat_count: int,
    label_count: int,
    magic: bytes,
) -> _DeltaV1BuildResult:
    """
    从逐帧 labels 构建 delta-v1 bytes.

    语义:
    - base labels 是 segmentStartFrame 的绝对 labels.
    - body 从 startFrame+1 开始,每帧写“相对上一帧”的更新.
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

    end_frame_excl = int(segment_start_frame) + int(segment_frame_count)
    if len(labels_by_frame) < end_frame_excl:
        raise ValueError(
            f"labels_by_frame too short: need >= {end_frame_excl}, got {len(labels_by_frame)}"
        )

    header = magic + struct.pack(
        "<5I",
        1,  # version
        int(segment_start_frame),
        int(segment_frame_count),
        int(splat_count),
        int(label_count),
    )

    # 每条 update: (u32 splatId, u16 newLabel, u16 reserved=0)
    update_dt = np.dtype([("splatId", "<u4"), ("newLabel", "<u2"), ("reserved", "<u2")], align=False)

    parts: List[bytes] = [header]
    total_updates = 0
    max_update_count = 0

    for frame in range(int(segment_start_frame) + 1, end_frame_excl):
        prev = labels_by_frame[int(frame) - 1]
        curr = labels_by_frame[int(frame)]
        if prev.shape != (int(splat_count),) or curr.shape != (int(splat_count),):
            raise ValueError(
                f"labels shape mismatch at frame={frame}: prev={prev.shape} curr={curr.shape} expected=({int(splat_count)},)"
            )

        changed = np.nonzero(prev != curr)[0].astype(np.uint32, copy=False)
        update_count = int(changed.size)
        parts.append(struct.pack("<I", int(update_count)))

        if update_count == 0:
            continue

        # np.nonzero 返回的索引天然是递增的,这里再 assert 一次,避免未来改动引入乱序.
        if update_count >= 2 and not np.all(changed[1:] > changed[:-1]):
            raise ValueError(f"delta-v1 requires strictly increasing splatId within a frame block, frame={frame}")

        new_labels = curr[changed].astype(np.uint16, copy=False)
        max_label = int(new_labels.max(initial=0))
        if max_label >= int(label_count):
            raise ValueError(f"label out of range at frame={frame}: max={max_label} label_count={int(label_count)}")

        updates = np.empty((update_count,), dtype=update_dt)
        updates["splatId"] = changed
        updates["newLabel"] = new_labels
        updates["reserved"] = np.uint16(0)
        parts.append(updates.tobytes(order="C"))

        total_updates += int(update_count)
        max_update_count = max(int(max_update_count), int(update_count))

    delta_bytes = b"".join(parts)
    return _DeltaV1BuildResult(
        delta_bytes=delta_bytes,
        total_updates=int(total_updates),
        max_update_count=int(max_update_count),
    )


def _decode_label_delta_v1(
    delta_bytes: bytes,
    *,
    expected_magic: bytes,
    base_labels_u16: np.ndarray,
) -> Tuple[int, int, int, int, List[np.ndarray]]:
    """
    解码 delta-v1,返回该 segment 的逐帧 labels.

    返回:
    - (segmentStartFrame, segmentFrameCount, splatCount, labelCount, labelsBySegmentFrame)
      - labelsBySegmentFrame 长度为 segmentFrameCount,第 0 帧是 base labels.
    """
    if len(expected_magic) != 8:
        raise ValueError("expected_magic must be 8 bytes")
    if base_labels_u16.ndim != 1:
        raise ValueError("base_labels_u16 must be 1D")

    header_size = 8 + 5 * 4
    if len(delta_bytes) < header_size:
        raise ValueError(f"delta_bytes too short: {len(delta_bytes)} < {header_size}")

    magic = delta_bytes[:8]
    if magic != expected_magic:
        raise ValueError(f"delta magic mismatch: got={magic!r} expected={expected_magic!r}")

    version, seg_start, seg_count, splat_count, label_count = struct.unpack("<5I", delta_bytes[8:header_size])
    if int(version) != 1:
        raise ValueError(f"delta version must be 1, got {version}")
    if int(seg_count) <= 0:
        raise ValueError(f"segmentFrameCount must be > 0, got {seg_count}")
    if int(splat_count) <= 0:
        raise ValueError(f"splatCount must be > 0, got {splat_count}")
    if int(base_labels_u16.shape[0]) != int(splat_count):
        raise ValueError(f"base labels length mismatch: {int(base_labels_u16.shape[0])} vs splatCount={int(splat_count)}")

    update_dt = np.dtype([("splatId", "<u4"), ("newLabel", "<u2"), ("reserved", "<u2")], align=False)

    offset = int(header_size)
    labels = base_labels_u16.astype(np.uint16, copy=True)
    out: List[np.ndarray] = [labels.copy()]

    for _rel in range(1, int(seg_count)):
        if offset + 4 > len(delta_bytes):
            raise ValueError("delta_bytes truncated while reading updateCount")
        (update_count,) = struct.unpack("<I", delta_bytes[offset : offset + 4])
        offset += 4

        update_count = int(update_count)
        if update_count < 0:
            raise ValueError("updateCount must be >= 0")
        if update_count == 0:
            out.append(labels.copy())
            continue

        need = int(update_count) * int(update_dt.itemsize)
        if offset + need > len(delta_bytes):
            raise ValueError("delta_bytes truncated while reading updates payload")

        updates = np.frombuffer(delta_bytes, dtype=update_dt, count=int(update_count), offset=int(offset))
        offset += int(need)

        if np.any(updates["reserved"] != 0):
            raise ValueError("delta update reserved field must be 0")

        splat_ids = updates["splatId"]
        if int(splat_ids.size) >= 2 and not np.all(splat_ids[1:] > splat_ids[:-1]):
            raise ValueError("delta-v1 requires strictly increasing splatId within a frame block")
        if int(splat_ids.max(initial=0)) >= int(splat_count):
            raise ValueError("delta splatId out of range")

        new_labels = updates["newLabel"]
        if int(new_labels.max(initial=0)) >= int(label_count):
            raise ValueError("delta newLabel out of range")

        labels[splat_ids] = new_labels
        out.append(labels.copy())

    if offset != len(delta_bytes):
        raise ValueError(f"delta_bytes has trailing bytes: parsed={offset} total={len(delta_bytes)}")

    return int(seg_start), int(seg_count), int(splat_count), int(label_count), out


@dataclass(frozen=True)
class _ShnLayout:
    """
    描述 checkpoint 里 `splats["shN"]` 的布局.

    支持:
    - 静态: [N,K,3]
    - 动态: [F,N,K,3] 或 [N,F,K,3]
    """

    is_dynamic: bool
    # 仅当 is_dynamic=True 时有效.
    frame_axis: Optional[int]
    frame_count: int
    splat_count: int
    coeff_count: int


def _infer_shn_layout(
    shn: torch.Tensor,
    *,
    frame_count: int,
    shn_frame_axis: Optional[int],
) -> _ShnLayout:
    """
    推断 `shN` 的帧轴,并在歧义/不匹配时 fail-fast.

    约定:
    - `--shn-frame-axis 0` 表示 shape=[F,N,K,3]
    - `--shn-frame-axis 1` 表示 shape=[N,F,K,3]
    """
    if shn.ndim == 3:
        if int(shn.shape[2]) != 3:
            raise ValueError(f"shN must have last dim=3, got shape={tuple(shn.shape)}")
        return _ShnLayout(
            is_dynamic=False,
            frame_axis=None,
            frame_count=1,
            splat_count=int(shn.shape[0]),
            coeff_count=int(shn.shape[1]),
        )

    if shn.ndim != 4:
        raise ValueError(f"shN must be [N,K,3] or [F,N,K,3]/[N,F,K,3], got shape={tuple(shn.shape)}")
    if frame_count <= 0:
        raise ValueError("dynamic shN requires --frame-count > 0")
    if int(shn.shape[3]) != 3:
        raise ValueError(f"shN must have last dim=3, got shape={tuple(shn.shape)}")

    dim0 = int(shn.shape[0])
    dim1 = int(shn.shape[1])

    match0 = dim0 == int(frame_count)
    match1 = dim1 == int(frame_count)

    if shn_frame_axis is not None:
        axis = int(shn_frame_axis)
        if axis not in (0, 1):
            raise ValueError("--shn-frame-axis must be 0 or 1")
        if axis == 0 and not match0:
            raise ValueError(f"--shn-frame-axis=0 requires shN.shape[0]==frame_count, got {dim0} vs {int(frame_count)}")
        if axis == 1 and not match1:
            raise ValueError(f"--shn-frame-axis=1 requires shN.shape[1]==frame_count, got {dim1} vs {int(frame_count)}")
    else:
        if match0 and not match1:
            axis = 0
        elif match1 and not match0:
            axis = 1
        elif match0 and match1:
            raise ValueError(
                "ambiguous dynamic shN: both shN.shape[0] and shN.shape[1] match --frame-count. "
                "Please pass --shn-frame-axis 0|1."
            )
        else:
            raise ValueError(
                f"dynamic shN frame axis mismatch: shN.shape[:2]=({dim0},{dim1}) does not match --frame-count={int(frame_count)}"
            )

    splat_count = dim1 if int(axis) == 0 else dim0
    coeff_count = int(shn.shape[2])
    return _ShnLayout(
        is_dynamic=True,
        frame_axis=int(axis),
        frame_count=int(frame_count),
        splat_count=int(splat_count),
        coeff_count=int(coeff_count),
    )


def _shn_get_frame_view(shn: torch.Tensor, *, layout: _ShnLayout, frame_index: int) -> torch.Tensor:
    """
    统一返回某一帧的 `shN` 视图: [N,K,3].

    - 静态 shN: 忽略 frame_index,直接返回原 tensor.
    - 动态 shN: 根据 layout.frame_axis 取出对应帧.
    """
    if not layout.is_dynamic:
        return shn

    if not (0 <= int(frame_index) < int(layout.frame_count)):
        raise ValueError(f"frame_index out of range: {frame_index} (frame_count={layout.frame_count})")
    if layout.frame_axis == 0:
        return shn[int(frame_index)]
    if layout.frame_axis == 1:
        return shn[:, int(frame_index)]
    raise RuntimeError("internal error: invalid layout.frame_axis")


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


def _iter_sh_rest_band_defs(*, bands: int) -> List[Tuple[int, str, int, int]]:
    """
    返回每个 band 的定义,用于统一处理静态/动态 shN.

    返回: List[(bandIndex(1..3), bandName, coeffOffset, coeffCount)].
    - coeffOffset/Count 的单位是“rest coeff index”(不是 float3 的元素偏移).
    """
    if not (1 <= int(bands) <= 3):
        raise ValueError(f"bands must be in [1,3], got {bands}")

    out: List[Tuple[int, str, int, int]] = []
    offset = 0

    # band 1: 3 coeff
    out.append((1, "sh1", int(offset), 3))
    offset += 3
    if int(bands) >= 2:
        out.append((2, "sh2", int(offset), 5))
        offset += 5
    if int(bands) >= 3:
        out.append((3, "sh3", int(offset), 7))
        offset += 7

    rest_coeff_total = int((int(bands) + 1) ** 2 - 1)
    if int(offset) != int(rest_coeff_total):
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


def _fit_sh_codebook_from_sample(
    sample_f32: np.ndarray,
    *,
    name: str,
    codebook_size: int,
    seed: int,
    kmeans_iters: int,
) -> np.ndarray:
    """
    仅拟合 codebook(centroids),不做 labels 分配.

    用途:
    - 动态 `shN` 需要跨帧抽样拟合 codebook.
    - labels 分配会按“逐帧 + chunk”的方式进行,避免一次性 flatten 到内存里.
    """
    if not (1 <= int(codebook_size) <= 65535):
        raise ValueError(f"--shn-count must be in [1,65535], got {codebook_size}")
    if int(kmeans_iters) <= 0:
        raise ValueError(f"--shn-kmeans-iters must be > 0, got {kmeans_iters}")

    # 延迟 import,避免在不导出 SH 时引入 SciPy 依赖.
    import warnings

    from scipy.cluster.vq import kmeans2

    sample = sample_f32.astype(np.float32, copy=False)
    n = int(sample.shape[0])
    d = int(sample.shape[1])
    if sample.shape != (n, d):
        raise ValueError("internal error: sample shape mismatch")
    if d % 3 != 0:
        raise ValueError(f"internal error: sample dim must be multiple of 3, got {d}")
    if n <= 0:
        raise ValueError("sample must be non-empty")

    print(f"[splat4d] {name} kmeans(dynamic): sample={n:,}, K={int(codebook_size)}, D={d}")
    centroids: Optional[np.ndarray] = None
    for attempt in range(3):
        attempt_seed = int(seed) + int(attempt)

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
            print(f"[splat4d] {name} kmeans(dynamic): empty cluster detected, retrying...")
        print(f"[splat4d] {name} kmeans(dynamic) retry {attempt+1}/3: used={used}/{int(codebook_size)} seed={attempt_seed}")

    if centroids is None:
        raise RuntimeError("internal error: kmeans2 returned no centroids")

    return centroids.astype(np.float32, copy=False)


def _torch_sh_band_flat_f32(sh_coeff: torch.Tensor) -> np.ndarray:
    """
    把 torch 的 `[n,coeffCount,3]` 转成 numpy 的 `[n,coeffCount*3]` float32.
    """
    x = sh_coeff.detach().cpu()
    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    n = int(x.shape[0])
    coeff_count = int(x.shape[1])
    if x.ndim != 3 or int(x.shape[2]) != 3:
        raise ValueError(f"internal error: sh_coeff must be [n,coeffCount,3], got shape={tuple(x.shape)}")
    return x.reshape(n, coeff_count * 3).numpy()


def _sample_shn_dynamic_band_flat(
    *,
    shn: torch.Tensor,
    layout: _ShnLayout,
    keep_indices: Optional[torch.Tensor],
    frame_count: int,
    sample_size: int,
    seed: int,
    coeff_offset: int,
    coeff_count: int,
) -> np.ndarray:
    """
    从动态 shN 中跨帧抽样,生成 kmeans 的 sample matrix.

    采样单位是二元组 `(frameIndex, splatId)`.
    - splatId 是“导出后的索引”(已应用 keep mask).
    - 当 keep_indices 不为空时,会映射回原始 ckpt 的 splat 索引再 gather.
    """
    if not layout.is_dynamic:
        raise ValueError("internal error: sample_shn_dynamic called for static shN")
    if sample_size <= 0:
        raise ValueError(f"--shn-codebook-sample must be > 0, got {sample_size}")
    if not (0 <= int(coeff_offset) and int(coeff_count) > 0):
        raise ValueError("internal error: invalid coeff_offset/coeff_count")

    n_out = int(layout.splat_count) if keep_indices is None else int(keep_indices.numel())
    if n_out <= 0:
        raise ValueError("splatCount must be > 0")

    total_points = int(frame_count) * int(n_out)
    sample_n = min(int(sample_size), int(total_points))
    if sample_n <= 0:
        raise ValueError("sample_n must be > 0")

    # 注意: 这里使用“带放回”采样,避免 totalPoints=F*N 超大时的无放回采样成本.
    rng = np.random.default_rng(int(seed))
    frames = rng.integers(0, int(frame_count), size=sample_n, dtype=np.int64)
    splats_out = rng.integers(0, int(n_out), size=sample_n, dtype=np.int64)

    order = np.argsort(frames)
    frames_sorted = frames[order]
    splats_sorted = splats_out[order]

    out = np.empty((sample_n, int(coeff_count) * 3), dtype=np.float32)

    # 逐帧 gather,避免单点索引的 Python 开销.
    start = 0
    while start < sample_n:
        f = int(frames_sorted[start])
        end = start + 1
        while end < sample_n and int(frames_sorted[end]) == f:
            end += 1

        idx_out_np = splats_sorted[start:end]
        idx_out = torch.as_tensor(idx_out_np, dtype=torch.int64)
        if keep_indices is None:
            idx_orig = idx_out
        else:
            idx_orig = keep_indices.index_select(0, idx_out)

        shn_frame = _shn_get_frame_view(shn, layout=layout, frame_index=int(f))  # [N,K,3]
        gathered = shn_frame.index_select(0, idx_orig)  # [m,K,3]
        band = gathered[:, int(coeff_offset) : int(coeff_offset + coeff_count), :]  # [m,coeff,3]
        flat = _torch_sh_band_flat_f32(band)  # [m,D]

        out[order[start:end]] = flat
        start = end

    return out


def _assign_sh_labels_dynamic_by_frame(
    *,
    shn: torch.Tensor,
    layout: _ShnLayout,
    keep_indices: Optional[torch.Tensor],
    frame_count: int,
    coeff_offset: int,
    coeff_count: int,
    centroids_f32: np.ndarray,
    assign_chunk: int,
) -> List[np.ndarray]:
    """
    对动态 shN 逐帧分配 labels.

    返回: labelsByFrame,长度=frame_count,每帧是 `[N] u16`(N 为导出后的 splatCount).
    """
    if not layout.is_dynamic:
        raise ValueError("internal error: assign_sh_labels_dynamic_by_frame called for static shN")
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    if assign_chunk <= 0:
        raise ValueError(f"--shn-assign-chunk must be > 0, got {assign_chunk}")

    from scipy.spatial import cKDTree

    tree = cKDTree(centroids_f32.astype(np.float32, copy=False))

    n_out = int(layout.splat_count) if keep_indices is None else int(keep_indices.numel())
    labels_by_frame: List[np.ndarray] = []

    for t in range(int(frame_count)):
        shn_frame = _shn_get_frame_view(shn, layout=layout, frame_index=int(t))  # [N,K,3]
        labels = np.empty((n_out,), dtype=np.uint16)

        for start in range(0, n_out, int(assign_chunk)):
            end = min(start + int(assign_chunk), n_out)

            if keep_indices is None:
                chunk = shn_frame[start:end]
            else:
                idx = keep_indices[start:end]
                chunk = shn_frame.index_select(0, idx)

            band = chunk[:, int(coeff_offset) : int(coeff_offset + coeff_count), :]  # [n,coeff,3]
            x = _torch_sh_band_flat_f32(band)
            _, idx_nn = tree.query(x, k=1, workers=-1)
            labels[start:end] = idx_nn.astype(np.uint16)

        labels_by_frame.append(labels)

    return labels_by_frame


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


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    把 3x3 旋转矩阵转成 quaternion(w,x,y,z).

    说明:
    - 这里的 quaternion 仅用于“全局坐标系旋转”叠乘到每个 Gaussian 的 rotation 上.
    - 训练归一化 transform 是 similarity(统一缩放 + 旋转 + 平移),其中旋转部分可以安全用 quaternion 表达.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got shape={R.shape}")

    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + float(R[0, 0] - R[1, 1] - R[2, 2]))) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + float(R[1, 1] - R[0, 0] - R[2, 2]))) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + float(R[2, 2] - R[0, 0] - R[1, 1]))) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    q = _normalize_quat_wxyz(q.reshape(1, 4)).reshape(4)
    return q


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    quaternion 乘法(wxyz),用于旋转叠乘.

    约定:
    - out = a * b
    - 旋转矩阵满足: R(out) = R(a) @ R(b)
      等价于: 先应用 b,再应用 a.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError(f"quat must end with 4 dims, got a={a.shape}, b={b.shape}")

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def _decompose_similarity_transform(T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    把 4x4 的 similarity transform 分解成:
    - scale: 统一缩放 s
    - rot: 旋转矩阵 R(3x3)
    - trans: 平移 t(3,)

    约束(与训练代码一致):
    - 训练时 normalize=True 的 transform 是 (similarity_from_cameras + PCA 对齐) 的组合.
    - 线性部分应满足 A ≈ s * R,且 s>0,det(R)=+1.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got shape={T.shape}")

    A = T[:3, :3]
    detA = float(np.linalg.det(A))
    if not np.isfinite(detA) or detA <= 0.0:
        raise ValueError(f"invalid similarity transform: det(A)={detA}")

    scale = float(np.cbrt(detA))
    R_approx = A / scale
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if float(np.linalg.det(R)) < 0.0:
        # 修正 SVD 可能产生的 reflection,保持 det=+1.
        U[:, -1] *= -1.0
        R = U @ Vt

    trans = T[:3, 3]
    return float(scale), R.astype(np.float32), trans.astype(np.float32)


def _compute_colmap_to_train_transform(*, colmap_dir: Path) -> np.ndarray:
    """
    复现训练时 `FreeTimeParser(normalize=True)` 的坐标归一化 transform.

    训练侧的含义:
    - p_train = T @ p_colmap
    - 其中 T = T2 @ T1
      - T1: similarity_from_cameras(以相机中心做平移+统一缩放)
      - T2: align_principle_axes(对点云做 PCA 对齐)

    这里我们用 COLMAP sparse 模型(cameras/images/points3D)重建同一个 T,
    让 exporter 能把 ckpt 的训练坐标反变换回 COLMAP 原始空间.
    """
    # 备注: 这里用 lazy import,避免用户只导出 train 空间时引入额外依赖/耗时.
    from datasets.normalize import (
        align_principle_axes,
        similarity_from_cameras,
        transform_cameras,
        transform_points,
    )
    from datasets.read_write_model import read_model

    colmap_dir = Path(colmap_dir)
    if not colmap_dir.exists():
        raise FileNotFoundError(f"colmap_dir not found: {colmap_dir}")

    cameras, images, points3D = read_model(str(colmap_dir), ext="")
    if images is None or points3D is None:
        raise RuntimeError(f"failed to read COLMAP model from: {colmap_dir}")
    if len(images) == 0:
        raise ValueError(f"COLMAP images is empty: {colmap_dir}")
    if len(points3D) == 0:
        raise ValueError(f"COLMAP points3D is empty: {colmap_dir}")

    # 1) 构建 camtoworlds: inverse(w2c),与训练代码一致.
    w2c_mats: List[np.ndarray] = []
    bottom = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4)
    for im in images.values():
        R = im.qvec2rotmat().astype(np.float32, copy=False)
        t = np.asarray(im.tvec, dtype=np.float32).reshape(3, 1)
        w2c = np.concatenate([np.concatenate([R, t], axis=1), bottom], axis=0)
        w2c_mats.append(w2c)

    w2c = np.stack(w2c_mats, axis=0).astype(np.float32)
    camtoworlds = np.linalg.inv(w2c).astype(np.float32)

    # 2) points3D xyz
    points = np.stack([p.xyz for p in points3D.values()], axis=0).astype(np.float32)

    # 3) 复现 normalize=True 的 transform
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    points = transform_points(T1, points)

    T2 = align_principle_axes(points)
    camtoworlds = transform_cameras(T2, camtoworlds)
    points = transform_points(T2, points)

    transform = (T2 @ T1).astype(np.float32)

    # 小 sanity: 相机中心应该被移到 0 附近,避免用户拿错 colmap_dir 导致“越变越歪”.
    cam_centers = camtoworlds[:, :3, 3]
    cam_mean = cam_centers.mean(axis=0)
    if float(np.linalg.norm(cam_mean)) > 1e-3:
        print(
            f"[splat4d][warn] normalized camera center mean is not ~0: mean={cam_mean.tolist()} (colmap_dir={colmap_dir})"
        )

    return transform


@dataclass(frozen=True)
class _SpaceTransform:
    """
    把 ckpt(训练 normalized 空间)的 Gaussian 反变换回某个目标空间.

    约定:
    - 对 position: out = (in - sub_trans) @ linear_T
    - 对 velocity: out = in @ linear_T
    - 对 scale: out = in * scale_mult
    - 对 rotation(quat): out = quat_left * in
    """

    sub_trans: np.ndarray  # [3] float32,先做 in - sub_trans
    linear_T: np.ndarray  # [3,3] float32,行向量右乘矩阵
    scale_mult: float  # float,统一尺度因子(仅用于 scales)
    quat_left: np.ndarray  # [4] float32,全局旋转 quaternion(wxyz)
    colmap_to_train: np.ndarray  # [4,4] float32,训练归一化 transform(用于写 XFRM section)


def _build_train_to_colmap_transform(*, colmap_to_train: np.ndarray) -> _SpaceTransform:
    """
    从训练归一化 transform(T: colmap->train)推导出 train->colmap 的反变换参数.

    这样导出 `.splat4d` 时,可以把 ckpt 里的位置/速度/尺度/旋转统一对齐到 COLMAP 原始空间.
    """
    scale, rot, trans = _decompose_similarity_transform(colmap_to_train)
    inv_scale = 1.0 / float(scale)

    # train->colmap:
    # p_colmap = (1/s) * R^T * (p_train - t)
    # 行向量实现: (p_train - t) @ ((1/s)*R^T)^T = (p_train - t) @ ((1/s)*R)
    linear_T = (inv_scale * rot).astype(np.float32)

    # quat: R_colmap = R^T @ R_train
    quat_left = _rotmat_to_quat_wxyz(rot.T)

    print(
        "[splat4d] output_space=colmap: apply train->colmap inverse normalize transform\n"
        f"  - inv_scale={inv_scale:.9f}\n"
        f"  - sub_trans(train)={trans.tolist()}\n"
        f"  - rot_det={float(np.linalg.det(rot)):.6f}"
    )

    return _SpaceTransform(
        sub_trans=trans.astype(np.float32, copy=False),
        linear_T=linear_T,
        scale_mult=float(inv_scale),
        quat_left=quat_left.astype(np.float32, copy=False),
        colmap_to_train=np.asarray(colmap_to_train, dtype=np.float32),
    )


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
    shn_frame_axis: Optional[int],
    self_check_delta: bool,
    shn_codebook_sample: int,
    shn_assign_chunk: int,
    shn_kmeans_iters: int,
    seed: int,
    output_space: Literal["train", "colmap"] = "train",
    colmap_dir: Optional[Path] = None,
) -> None:
    # `splat4d_version` 控制 time/duration 的语义(window vs gaussian),
    # `splat4d_format_version` 控制文件是否带 header(legacy vs header+sections).
    # 这两者是正交概念,但为了避免误用,我们允许 format=0(auto) 来推导一个更安全的默认值.
    if splat4d_format_version not in (0, 1, 2):
        raise ValueError("--splat4d-format-version must be 0(auto), 1 or 2")
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

    # -----------------------------
    # format version auto-resolve
    # -----------------------------
    if int(splat4d_format_version) == 0:
        reasons: List[str] = []
        if int(splat4d_version) == 2:
            # timeModel=2(gaussian) 如果用 legacy v1(无 header)导出,Unity importer 往往只能走 window 路径,
            # 很容易出现"薄层/稀疏"的裁剪伪影.因此默认改为 format v2.
            reasons.append("timeModel=2(gaussian)")
        if int(sh_bands) > 0:
            # SH rest/deltaSegments 只能放在 header+sections 里.
            reasons.append(f"sh_bands={int(sh_bands)}")

        chosen = 2 if reasons else 1
        reason_text = ", ".join(reasons) if reasons else "default"
        print(f"[splat4d] format=auto -> v{chosen} ({reason_text})")
        splat4d_format_version = int(chosen)

    if sh_bands > 0 and int(splat4d_format_version) != 2:
        raise ValueError("SH rest export requires --splat4d-format-version 2 (or 0=auto)")

    if int(splat4d_format_version) == 1 and int(splat4d_version) == 2:
        # 这是一个“技术上可写,但语义上容易被 importer 误读”的组合.
        # 允许继续导出,但必须给一个醒目 warning,避免用户以为"v2 gaussian"就一定会走 v2 importer.
        print(
            "[splat4d][warn] 你选择了: --splat4d-version 2(timeModel=2) + --splat4d-format-version 1(legacy无header).\n"
            "[splat4d][warn] 旧 importer 可能会按 window(time0+duration)解释,导致显示变薄/稀疏.\n"
            "[splat4d][warn] 建议: 去掉 --splat4d-format-version 或改用 --splat4d-format-version 2.",
            file=sys.stderr,
        )
    if delta_segment_length < 0:
        raise ValueError("--delta-segment-length must be >= 0")
    if sh_bands > 0:
        if not (1 <= shn_count <= 65535):
            raise ValueError("--shn-count must be in [1,65535]")
        if shn_labels_encoding == "delta-v1" and frame_count <= 0:
            raise ValueError("--frame-count must be > 0 when --shn-labels-encoding=delta-v1")
        if shn_frame_axis is not None and int(shn_frame_axis) not in (0, 1):
            raise ValueError("--shn-frame-axis must be 0 or 1")
    if bool(self_check_delta) and not (sh_bands > 0 and shn_labels_encoding == "delta-v1"):
        raise ValueError("--self-check-delta requires --sh-bands>0 and --shn-labels-encoding=delta-v1")

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

    shn_layout: Optional[_ShnLayout] = None
    # 动态 shN 仅在 labelsEncoding=delta-v1 时有意义,因为 format v2 当前不支持“逐帧全量 labels blob”.
    if sh_bands > 0:
        if shn is None:
            raise RuntimeError("internal error: shn is None but sh_bands > 0")
        if shn.ndim == 4 and shn_labels_encoding != "delta-v1":
            raise ValueError("dynamic shN requires --shn-labels-encoding=delta-v1")
        shn_layout = _infer_shn_layout(shn, frame_count=int(frame_count), shn_frame_axis=shn_frame_axis)

    n_total = int(means.shape[0])
    print(f"[splat4d] gaussians: {n_total:,}")

    # base opacity filter(可选): 先在 CPU 上做一次筛选,降低输出体积.
    # 注意: 这只是 base opacity,不会考虑 temporal opacity.
    shn_keep_indices: Optional[torch.Tensor] = None
    if base_opacity_threshold > 0.0:
        base_opacity = torch.sigmoid(opacities_logit.detach().cpu())
        keep = base_opacity >= float(base_opacity_threshold)
        keep_idx = keep.nonzero(as_tuple=False).reshape(-1).to(dtype=torch.int64)
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
            if shn_layout is None:
                raise RuntimeError("internal error: shn_layout is None")
            # 静态 shN 直接裁剪即可.
            # 动态 shN 若直接裁剪会复制一个巨大的 4D tensor,这里改为保存 indices,后续按 chunk gather.
            if not shn_layout.is_dynamic:
                shn = shn[keep]
            else:
                shn_keep_indices = keep_idx
        n_total = keep_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = _record_dtype()

    # 统计信息(用于 sanity check)
    times_np = _as_numpy_f32(times).reshape(-1)
    sigmas_np = np.exp(_as_numpy_f32(durations_log).reshape(-1))
    print(f"[splat4d] times range: [{times_np.min():.6f}, {times_np.max():.6f}]")
    print(f"[splat4d] sigma(exp(duration)) range: [{sigmas_np.min():.6f}, {sigmas_np.max():.6f}]")

    # -----------------------------
    # 输出坐标空间: train(normalized) vs colmap(original)
    # -----------------------------
    if output_space not in ("train", "colmap"):
        raise ValueError("--output-space must be 'train' or 'colmap'")

    space_xform: Optional[_SpaceTransform] = None
    if output_space == "colmap":
        if colmap_dir is None:
            raise ValueError("--colmap-dir is required when --output-space=colmap")
        colmap_to_train = _compute_colmap_to_train_transform(colmap_dir=Path(colmap_dir))
        space_xform = _build_train_to_colmap_transform(colmap_to_train=colmap_to_train)

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

                # 坐标空间对齐(可选): 把 ckpt 的训练坐标反变换回 COLMAP 原始空间.
                if space_xform is not None:
                    pos0 = (pos0 - space_xform.sub_trans) @ space_xform.linear_T
                    vel_np = vel_np @ space_xform.linear_T
                    scales_lin = scales_lin * np.float32(space_xform.scale_mult)
                    q_norm = _quat_mul_wxyz(space_xform.quat_left, q_norm)
                    q_norm = _normalize_quat_wxyz(q_norm)

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

    # 1) 预计算 SH per-band codebook + labels
    # - 静态 shN: 生成 1 份 labels(u16[N])
    # - 动态 shN: 生成 labelsByFrame(F 份 u16[N]),用于后续 delta-v1
    sh_band_centroids_bytes: Dict[int, bytes] = {}
    sh_band_codebook_count: Dict[int, int] = {}
    sh_band_labels_u16: Dict[int, np.ndarray] = {}
    sh_band_labels_by_frame: Dict[int, List[np.ndarray]] = {}

    if sh_bands > 0:
        if shn is None or shn_layout is None:
            raise RuntimeError("internal error: shn/shn_layout is None but sh_bands > 0")

        # 只需要 rest 的前 ((bands+1)^2-1) 个 coeff.
        rest_coeff_total = int((int(sh_bands) + 1) ** 2 - 1)
        coeff_total = int(shn_layout.coeff_count if shn_layout.is_dynamic else shn.shape[1])
        if coeff_total < rest_coeff_total:
            raise ValueError(f"shN coeff count too small: need >= {rest_coeff_total}, got {coeff_total}")

        if not shn_layout.is_dynamic:
            shn_np = _as_numpy_f32(shn)  # [N,K,3]
            per_band = _flatten_sh_rest_v2_per_band(shn_np, bands=int(sh_bands))

            for band_idx, (band_name, (band_flat, _coeff_count)) in enumerate(per_band.items()):
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

            for band, labels_u16 in sh_band_labels_u16.items():
                max_label = int(labels_u16.max(initial=0))
                if max_label >= int(sh_band_codebook_count[band]):
                    raise RuntimeError(
                        f"internal error: band={band} labels has out-of-range value {max_label} >= {sh_band_codebook_count[band]}"
                    )
        else:
            # 动态 shN: codebook 仍是“全局 persistent”,但 labels 要逐帧分配.
            band_defs = _iter_sh_rest_band_defs(bands=int(sh_bands))
            for band, band_name, coeff_offset, coeff_count in band_defs:
                band_seed = int(seed) + int(band - 1)

                sample = _sample_shn_dynamic_band_flat(
                    shn=shn,
                    layout=shn_layout,
                    keep_indices=shn_keep_indices,
                    frame_count=int(frame_count),
                    sample_size=int(shn_codebook_sample),
                    seed=int(band_seed),
                    coeff_offset=int(coeff_offset),
                    coeff_count=int(coeff_count),
                )
                centroids_f32 = _fit_sh_codebook_from_sample(
                    sample,
                    name=str(band_name),
                    codebook_size=int(shn_count),
                    seed=int(band_seed),
                    kmeans_iters=int(shn_kmeans_iters),
                )

                sh_band_codebook_count[int(band)] = int(centroids_f32.shape[0])

                labels_by_frame = _assign_sh_labels_dynamic_by_frame(
                    shn=shn,
                    layout=shn_layout,
                    keep_indices=shn_keep_indices,
                    frame_count=int(frame_count),
                    coeff_offset=int(coeff_offset),
                    coeff_count=int(coeff_count),
                    centroids_f32=centroids_f32,
                    assign_chunk=int(shn_assign_chunk),
                )
                sh_band_labels_by_frame[int(band)] = labels_by_frame

                # labels 范围 sanity.
                max_label = max(int(x.max(initial=0)) for x in labels_by_frame)
                if max_label >= int(sh_band_codebook_count[int(band)]):
                    raise RuntimeError(
                        f"internal error: band={band} labels has out-of-range value {max_label} >= {sh_band_codebook_count[int(band)]}"
                    )

                if shn_centroids_type == "f16":
                    sh_band_centroids_bytes[int(band)] = centroids_f32.astype("<f2", copy=False).tobytes(order="C")
                elif shn_centroids_type == "f32":
                    sh_band_centroids_bytes[int(band)] = centroids_f32.astype("<f4", copy=False).tobytes(order="C")
                else:  # pragma: no cover
                    raise ValueError(f"unknown shn_centroids_type: {shn_centroids_type}")

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

            # 坐标空间对齐(可选): 把 ckpt 的训练坐标反变换回 COLMAP 原始空间.
            if space_xform is not None:
                pos0 = (pos0 - space_xform.sub_trans) @ space_xform.linear_T
                vel_np = vel_np @ space_xform.linear_T
                scales_lin = scales_lin * np.float32(space_xform.scale_mult)
                q_norm = _quat_mul_wxyz(space_xform.quat_left, q_norm)
                q_norm = _normalize_quat_wxyz(q_norm)

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

        # ---- XFRM(可选) ----
        # 当我们导出到 colmap(original)空间时,把训练用的 colmap->train 归一化矩阵写进文件,
        # 便于离线工具或未来 importer 做一致性校验.
        if space_xform is not None:
            xfrm_offset = int(f.tell())
            xfrm_bytes = space_xform.colmap_to_train.astype("<f4", copy=False).tobytes(order="C")
            if len(xfrm_bytes) != 64:
                raise RuntimeError(f"internal error: xfrm_bytes size={len(xfrm_bytes)}, expected 64")
            f.write(xfrm_bytes)
            sections.append((_SECT_XFRM, 0, 0, 0, int(xfrm_offset), int(len(xfrm_bytes))))

        # ---- SH sections ----
        if sh_bands > 0:
            # centroids
            for band in range(1, int(sh_bands) + 1):
                centroids_offset = int(f.tell())
                centroids_bytes = sh_band_centroids_bytes[int(band)]
                f.write(centroids_bytes)
                sections.append((_SECT_SHCT, int(band), 0, 0, int(centroids_offset), int(len(centroids_bytes))))

            shn_is_dynamic = bool(shn_layout is not None and shn_layout.is_dynamic)

            if shn_labels_encoding == "full":
                if shn_is_dynamic:
                    raise RuntimeError("internal error: dynamic shN should not reach labelsEncoding=full path")

                # full: 仅写 1 份 labels blob.
                for band in range(1, int(sh_bands) + 1):
                    labels_offset = int(f.tell())
                    labels_u16 = sh_band_labels_u16[int(band)].astype("<u2", copy=False)
                    labels_bytes = labels_u16.tobytes(order="C")
                    f.write(labels_bytes)

                    sections.append(
                        (
                            _SECT_SHLB,
                            int(band),
                            0,
                            0,
                            int(labels_offset),
                            int(len(labels_bytes)),
                        )
                    )
            else:
                if delta_segments is None:
                    raise RuntimeError("internal error: delta_segments is None but delta-v1 requested")

                if not shn_is_dynamic:
                    # 静态 labels: 为避免重复占用空间,每个 band 只写 1 份 labels blob,
                    # 但仍然会为每个 segment 生成独立 SHLB entry(指向同一 offset).
                    labels_offset_by_band: Dict[int, int] = {}
                    labels_length_by_band: Dict[int, int] = {}
                    for band in range(1, int(sh_bands) + 1):
                        labels_offset = int(f.tell())
                        labels_u16 = sh_band_labels_u16[int(band)].astype("<u2", copy=False)
                        labels_bytes = labels_u16.tobytes(order="C")
                        f.write(labels_bytes)
                        labels_offset_by_band[int(band)] = int(labels_offset)
                        labels_length_by_band[int(band)] = int(len(labels_bytes))

                    for band in range(1, int(sh_bands) + 1):
                        label_count = int(sh_band_codebook_count[int(band)])
                        for seg_start, seg_count in delta_segments:
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

                            delta_offset = int(f.tell())
                            delta_bytes = _build_label_delta_v1_static(
                                segment_start_frame=int(seg_start),
                                segment_frame_count=int(seg_count),
                                splat_count=int(n_total),
                                label_count=int(label_count),
                                magic=b"SPL4DLB1",
                            )
                            if self_check_delta:
                                base_labels = sh_band_labels_u16[int(band)].astype(np.uint16, copy=False)
                                dec_start, dec_count, dec_splats, dec_labels, dec_frames = _decode_label_delta_v1(
                                    delta_bytes,
                                    expected_magic=b"SPL4DLB1",
                                    base_labels_u16=base_labels,
                                )
                                if dec_start != int(seg_start) or dec_count != int(seg_count):
                                    raise ValueError(
                                        f"self-check failed: delta header mismatch band={int(band)} "
                                        f"seg=({int(seg_start)},{int(seg_count)}) decoded=({dec_start},{dec_count})"
                                    )
                                if dec_splats != int(n_total) or dec_labels != int(label_count):
                                    raise ValueError(
                                        f"self-check failed: delta size mismatch band={int(band)} "
                                        f"splatCount={dec_splats} labelCount={dec_labels}"
                                    )
                                for rel, got in enumerate(dec_frames):
                                    if not np.array_equal(got, base_labels):
                                        raise ValueError(
                                            f"self-check failed: static delta decoded mismatch band={int(band)} "
                                            f"frame={int(seg_start)+int(rel)}"
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
                else:
                    # 动态 labels: 每个 segment 写自己的 base labels(允许 bytes 去重复用 offset),
                    # 并生成真实 delta-v1 updates.
                    labels_blob_cache_by_band: Dict[int, Dict[Tuple[bytes, int], Tuple[int, int]]] = {}

                    for band in range(1, int(sh_bands) + 1):
                        label_count = int(sh_band_codebook_count[int(band)])
                        labels_by_frame = sh_band_labels_by_frame[int(band)]

                        total_changes = 0
                        max_update = 0

                        cache = labels_blob_cache_by_band.setdefault(int(band), {})

                        for seg_start, seg_count in delta_segments:
                            # segment 边界处的“相邻帧变化量”也计入统计,但它不会写入 delta(因为该帧用 base labels 重置).
                            if int(seg_start) > 0:
                                boundary_prev = labels_by_frame[int(seg_start) - 1]
                                boundary_curr = labels_by_frame[int(seg_start)]
                                boundary_changes = int(np.count_nonzero(boundary_prev != boundary_curr))
                                total_changes += int(boundary_changes)
                                max_update = max(int(max_update), int(boundary_changes))

                            base_labels = labels_by_frame[int(seg_start)].astype("<u2", copy=False)
                            base_bytes = base_labels.tobytes(order="C")

                            # 可选去重: 如果多段 base labels 完全一致,复用同一 offset,减少文件膨胀.
                            digest = hashlib.blake2b(base_bytes, digest_size=16).digest()
                            key = (digest, int(len(base_bytes)))
                            if key in cache:
                                base_offset, base_length = cache[key]
                            else:
                                base_offset = int(f.tell())
                                f.write(base_bytes)
                                base_length = int(len(base_bytes))
                                cache[key] = (int(base_offset), int(base_length))

                            sections.append(
                                (
                                    _SECT_SHLB,
                                    int(band),
                                    int(seg_start),
                                    int(seg_count),
                                    int(base_offset),
                                    int(base_length),
                                )
                            )

                            delta_offset = int(f.tell())
                            res = _build_label_delta_v1_from_labels_by_frame(
                                labels_by_frame,
                                segment_start_frame=int(seg_start),
                                segment_frame_count=int(seg_count),
                                splat_count=int(n_total),
                                label_count=int(label_count),
                                magic=b"SPL4DLB1",
                            )
                            if self_check_delta:
                                dec_start, dec_count, dec_splats, dec_labels, dec_frames = _decode_label_delta_v1(
                                    res.delta_bytes,
                                    expected_magic=b"SPL4DLB1",
                                    base_labels_u16=base_labels,
                                )
                                if dec_start != int(seg_start) or dec_count != int(seg_count):
                                    raise ValueError(
                                        f"self-check failed: delta header mismatch band={int(band)} "
                                        f"seg=({int(seg_start)},{int(seg_count)}) decoded=({dec_start},{dec_count})"
                                    )
                                if dec_splats != int(n_total) or dec_labels != int(label_count):
                                    raise ValueError(
                                        f"self-check failed: delta size mismatch band={int(band)} "
                                        f"splatCount={dec_splats} labelCount={dec_labels}"
                                    )
                                for rel, got in enumerate(dec_frames):
                                    expected = labels_by_frame[int(seg_start) + int(rel)]
                                    if not np.array_equal(got, expected):
                                        raise ValueError(
                                            f"self-check failed: delta decoded mismatch band={int(band)} "
                                            f"frame={int(seg_start)+int(rel)}"
                                        )
                            f.write(res.delta_bytes)
                            sections.append(
                                (
                                    _SECT_SHDL,
                                    int(band),
                                    int(seg_start),
                                    int(seg_count),
                                    int(delta_offset),
                                    int(len(res.delta_bytes)),
                                )
                            )

                            total_changes += int(res.total_updates)
                            max_update = max(int(max_update), int(res.max_update_count))

                        # 日志统计(按全局相邻帧变化口径): changedPercent/avgUpdateCount/maxUpdateCount
                        frame_pairs = int(max(int(frame_count) - 1, 0))
                        if frame_pairs > 0:
                            changed_percent = float(total_changes) / float(int(n_total) * int(frame_pairs))
                            avg_update = float(total_changes) / float(int(frame_pairs))
                        else:
                            changed_percent = 0.0
                            avg_update = 0.0

                        print(
                            f"[splat4d] sh band={int(band)} delta stats: changedPercent={changed_percent*100.0:.4f}% "
                            f"avgUpdateCount={avg_update:.2f} maxUpdateCount={int(max_update)}"
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
        "--output-space",
        type=str,
        default="train",
        choices=["train", "colmap"],
        help=(
            "输出坐标空间: "
            "train=直接写 ckpt 的训练 normalized 坐标; "
            "colmap=把训练坐标反变换回 COLMAP 原始空间(需要 --colmap-dir)."
        ),
    )
    parser.add_argument(
        "--colmap-dir",
        type=Path,
        default=None,
        help="COLMAP sparse 模型目录(包含 cameras/images/points3D).当 --output-space=colmap 时必填.",
    )
    parser.add_argument(
        "--splat4d-format-version",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="文件格式版本: 0=auto(默认,推荐); 1=legacy无header(仅SH0); 2=header+sections(支持SH rest与deltaSegments)",
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
        "--shn-frame-axis",
        type=int,
        default=None,
        choices=[0, 1],
        help=(
            "当 checkpoint 的 splats['shN'] 是 4D 时,用于指定哪一轴是帧轴.\n"
            "- 0: shN shape=[F,N,K,3]\n"
            "- 1: shN shape=[N,F,K,3]\n"
            "当 exporter 无法唯一判定(例如两个轴都等于 --frame-count)时必须显式提供."
        ),
    )
    parser.add_argument(
        "--delta-segment-length",
        type=int,
        default=0,
        help="Delta segment length. 0 means single segment covering all frames (default: 0)",
    )
    parser.add_argument(
        "--self-check-delta",
        action="store_true",
        help="导出后自检 delta-v1: 解码复原逐帧 labels,并与 exporter 内部 labels 逐元素断言一致.",
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
        shn_frame_axis=args.shn_frame_axis,
        self_check_delta=bool(args.self_check_delta),
        shn_codebook_sample=int(args.shn_codebook_sample),
        shn_assign_chunk=int(args.shn_assign_chunk),
        shn_kmeans_iters=int(args.shn_kmeans_iters),
        seed=int(args.seed),
        output_space=str(args.output_space),  # type: ignore[arg-type]
        colmap_dir=args.colmap_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
