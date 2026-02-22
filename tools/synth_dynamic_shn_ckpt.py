#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成一个“合成动态 shN”的最小 FreeTimeGsVanilla checkpoint,用于验证:
- `.splat4d format v2` 的 delta-v1 可以产生非 0 updates.
- exporter 的 `--self-check-delta` 可以通过(复原 labels 与内部一致).

设计目标:
- 小: 默认 N=1024,F=5,避免占用大量磁盘与内存.
- 可复现: 固定 seed,并且每帧扰动的 splat 集合可重复.
- 符合 exporter 的必需字段结构: ckpt["splats"] 下包含 means/scales/quats/opacities/sh0/shN/times/durations/velocities.

最小 smoke 命令(验证“非 0 updates + self-check 通过”):

```bash
mkdir -p results/synth_delta_v1

python3 tools/synth_dynamic_shn_ckpt.py \
  --output results/synth_delta_v1/ckpt_synth_dynamic_shn_f5_n1024_axis0.pt \
  --frames 5 \
  --splats 1024 \
  --sh-bands 3 \
  --shn-frame-axis 0

python3 tools/exportor/export_splat4d.py \
  --ckpt results/synth_delta_v1/ckpt_synth_dynamic_shn_f5_n1024_axis0.pt \
  --output results/synth_delta_v1/out_v2_sh3_delta_seg2_k8_f32.splat4d \
  --splat4d-format-version 2 \
  --splat4d-version 2 \
  --sh-bands 3 \
  --frame-count 5 \
  --shn-frame-axis 0 \
  --shn-count 8 \
  --shn-centroids-type f32 \
  --shn-labels-encoding delta-v1 \
  --delta-segment-length 2 \
  --shn-codebook-sample 4000 \
  --shn-kmeans-iters 5 \
  --shn-assign-chunk 2048 \
  --self-check-delta
```

预期:
- exporter 输出的 `delta stats` 中,至少一个 band 的 `maxUpdateCount > 0`.
- `--self-check-delta` 不抛异常(ExitCode=0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import torch


def _rand_quat_wxyz(*, n: int, gen: torch.Generator) -> torch.Tensor:
    """
    生成随机四元数(wxyz),并做归一化.
    """
    q = torch.randn((n, 4), generator=gen, dtype=torch.float32)
    q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)

    # q 与 -q 表示同一旋转.这里强制 w>=0,避免量化时抖动过大.
    flip = q[:, 0] < 0
    q[flip] *= -1.0
    return q


def build_ckpt(
    *,
    splat_count: int,
    frame_count: int,
    sh_bands: int,
    shn_frame_axis: Literal[0, 1],
    seed: int,
) -> dict:
    """
    构建一个最小 ckpt dict,结构与真实训练 ckpt 对齐.
    """
    if splat_count <= 0:
        raise ValueError("splat_count must be > 0")
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    if sh_bands not in (1, 2, 3):
        raise ValueError("sh_bands must be in {1,2,3}")
    if shn_frame_axis not in (0, 1):
        raise ValueError("shn_frame_axis must be 0 or 1")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    n = int(splat_count)
    f = int(frame_count)
    rest_coeff_total = int((int(sh_bands) + 1) ** 2 - 1)  # bands=3 -> 15

    # -----------------------------
    # 4D splat fields
    # -----------------------------
    means = torch.randn((n, 3), generator=gen, dtype=torch.float32) * 0.5
    velocities = torch.randn((n, 3), generator=gen, dtype=torch.float32) * 0.05

    # times/durations 需要 shape=[N,1]
    times = torch.rand((n, 1), generator=gen, dtype=torch.float32)
    sigma = torch.rand((n, 1), generator=gen, dtype=torch.float32) * 0.06 + 0.02  # [0.02,0.08]
    durations = torch.log(sigma)

    # scales 是 log scale,确保 exp 后为正
    scales_lin = torch.rand((n, 3), generator=gen, dtype=torch.float32) * 0.08 + 0.01
    scales = torch.log(scales_lin)

    quats = _rand_quat_wxyz(n=n, gen=gen)

    # opacities 是 logit,这里让大多数 splat 可见
    opacities = torch.randn((n,), generator=gen, dtype=torch.float32) * 0.5 + 1.5

    # sh0: [N,1,3]
    sh0 = torch.randn((n, 1, 3), generator=gen, dtype=torch.float32) * 0.02

    # -----------------------------
    # 动态 shN: [F,N,K,3] 或 [N,F,K,3]
    # -----------------------------
    # base: 所有 splat 的基础 SH rest(每帧从 base 出发).
    base = torch.randn((n, rest_coeff_total, 3), generator=gen, dtype=torch.float32) * 0.02
    shn = base.unsqueeze(0).repeat(f, 1, 1, 1)  # [F,N,K,3]

    # 为了稳定地产生 label 变化,我们只对 band1 的 3 个 coeff 加一个大偏移.
    # 每帧选择一小撮 splat 做扰动,相邻帧的集合不同,因此必然出现 delta updates.
    perturb_count = min(64, n)
    for t in range(f):
        gen_t = torch.Generator(device="cpu").manual_seed(int(seed) + 1000 + int(t))
        idx = torch.randperm(n, generator=gen_t, dtype=torch.int64)[:perturb_count]
        shn[t, idx, 0:3, :] += 5.0

    if int(shn_frame_axis) == 1:
        shn = shn.permute(1, 0, 2, 3).contiguous()  # [N,F,K,3]

    splats = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "times": times,
        "durations": durations,
        "velocities": velocities,
        "shN": shn,
    }

    return {"splats": splats}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Synthesize a minimal dynamic-shN FreeTimeGsVanilla checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output ckpt_*.pt path")
    parser.add_argument("--splats", type=int, default=1024, help="Splat count N (default: 1024)")
    parser.add_argument("--frames", type=int, default=5, help="Frame count F (default: 5)")
    parser.add_argument("--sh-bands", type=int, default=3, choices=[1, 2, 3], help="SH bands for shN (default: 3)")
    parser.add_argument(
        "--shn-frame-axis",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 -> shN=[F,N,K,3], 1 -> shN=[N,F,K,3] (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (default: 12345)")

    args = parser.parse_args(argv)

    ckpt = build_ckpt(
        splat_count=int(args.splats),
        frame_count=int(args.frames),
        sh_bands=int(args.sh_bands),
        shn_frame_axis=int(args.shn_frame_axis),  # type: ignore[arg-type]
        seed=int(args.seed),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out))
    print(f"[synth] wrote ckpt: {out} (N={int(args.splats)}, F={int(args.frames)}, shn_frame_axis={int(args.shn_frame_axis)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
