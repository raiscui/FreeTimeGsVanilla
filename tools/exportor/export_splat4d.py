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
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


SH_C0: float = 0.28209479177387814


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
    splat4d_version: int,
    temporal_threshold: float,
    min_sigma: float,
    chunk_size: int,
    base_opacity_threshold: float,
) -> None:
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

    if splat4d_version == 1:
        sigma_factor = math.sqrt(-2.0 * math.log(temporal_threshold))
        print(f"[splat4d] version=1(hard-window) temporal_threshold={temporal_threshold} -> sigma_factor={sigma_factor:.6f}")
    else:
        sigma_factor = 0.0
        print("[splat4d] version=2(gaussian) time=mu_t, duration=sigma (ignore --temporal-threshold)")

    print(f"[splat4d] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    required = ["means", "scales", "quats", "opacities", "sh0", "times", "durations", "velocities"]
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
        n_total = keep_count

    # 输出文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = _record_dtype()

    # 统计信息(用于 sanity check)
    times_np = _as_numpy_f32(times).reshape(-1)
    sigmas_np = np.exp(_as_numpy_f32(durations_log).reshape(-1))
    print(f"[splat4d] times range: [{times_np.min():.6f}, {times_np.max():.6f}]")
    print(f"[splat4d] sigma(exp(duration)) range: [{sigmas_np.min():.6f}, {sigmas_np.max():.6f}]")

    bytes_expected = n_total * 64
    print(f"[splat4d] output: {output_path} ({bytes_expected/1024/1024:.1f} MB)")

    with output_path.open("wb") as f:
        written = 0
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            n = end - start

            # ---- 4D 参数: 从(mu_x, mu_t, v, sigma)映射到(time0,start)与硬窗口 ----
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
                # mu_x(t) = mu_x + v*(t - mu_t) == pos0 + v*(t - t0)
                # => pos0 = mu_x + v*(t0 - mu_t)
                pos0 = means_np + vel_np * (t0.reshape(-1, 1) - mu_np.reshape(-1, 1))
                out_time = t0
                out_duration = duration
            else:
                # v2: 直接把时间高斯核参数写入 time/duration.
                # - time=mu_t
                # - duration=sigma
                pos0 = means_np
                out_time = mu_np.astype(np.float32, copy=False)
                out_duration = sigma_np.astype(np.float32, copy=False)

            # ---- 其它静态参数 ----
            scales_lin = np.exp(_as_numpy_f32(scales_log[start:end]))  # [n,3]
            q_norm = _normalize_quat_wxyz(_as_numpy_f32(quats[start:end]))  # [n,4]
            q8 = _quantize_quat_to_u8(q_norm)  # [n,4]

            # 颜色: 只写 SH0.
            f_dc = _as_numpy_f32(sh0[start:end]).reshape(n, 3)
            base_rgb = 0.5 + SH_C0 * f_dc
            rgb8 = _quantize_0_1_to_u8(base_rgb)  # [n,3]

            # opacity: logit -> alpha
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Export .splat4d from FreeTimeGsVanilla checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to ckpt_*.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output .splat4d path")

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
        help="仅 v1 使用: temporal opacity threshold in (0,1) used to convert sigma -> hard window (default: 0.01)",
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

    args = parser.parse_args(argv)

    export_splat4d_from_ckpt(
        ckpt_path=args.ckpt,
        output_path=args.output,
        splat4d_version=int(args.splat4d_version),
        temporal_threshold=args.temporal_threshold,
        min_sigma=args.min_sigma,
        chunk_size=args.chunk_size,
        base_opacity_threshold=args.base_opacity_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
