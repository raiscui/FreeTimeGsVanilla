## Why

当前 `.splat4d format v2` 的 delta-v1 仍是“占位实现”.
多数导出文件的每帧 `updateCount=0`.
这导致 SH rest 无法真正随时间变化.
Unity 侧也无法验证 delta 管线的正确性.

现在做这件事的价值很明确.
我们已经有 `.splat4d v2` 的 section table,以及 delta header 的基本校验.
把 exporter 端的“真实 updates”补齐.
再把 Unity 端的“运行时应用”补齐.
就能形成可验证的闭环.
这也会为后续更贴近 DualGS 的压缩与流式策略打基础.

## What Changes

- FreeTimeGsVanilla exporter 端生成真实 delta-v1 updates.
  - 对比相邻帧 labels,只写 changed 的 `(splatId,newLabel)`.
  - 每帧 block 内 `splatId` 严格递增.
- exporter 支持动态 `splats["shN"]` 输入(4D),并保持静态 `shN` 行为不变.
- `.splat4d v2` 的 SH labels 改为 per-segment base labels.
  - 每个 segment 一份 base labels blob.
  - 不再让所有 segments 复用同一个 labels offset.
- gsplat-unity importer 读取并保存 per-segment base labels 与 delta bytes.
- gsplat-unity runtime 按 `TimeNormalized` 选帧应用 deltas.
  - 主线: GPU compute scatter 更新 `_SHBuffer` 的 SH rest.
  - 兜底: compute 不可用时禁用动态 SH,保持 frame0 静态效果,避免黑屏.
- 兼容性:
  - 如果 delta updates 全为 0,行为与现状一致.

本次明确不做:
- per-segment codebook.
- 稳定 permutation/reorder.
- delta-v2 event stream.
- motion(R-VQ/RANS)压缩 section.

## Capabilities

### New Capabilities

- `splat4d-delta-v1-exporter`: `.splat4d v2` delta-v1 真实 updates 生成与自检(含合成动态 ckpt).
- `gsplat-unity-splat4d-delta-v1-runtime`: Unity importer+runtime 应用 delta-v1(含 GPU compute scatter 与降级策略).

### Modified Capabilities

- (无)

## Impact

- 影响范围:
  - 本仓库: `tools/exportor/export_splat4d.py` 以及一个用于验证的合成 ckpt 小工具脚本.
  - 关联仓库: `/workspace/gsplat-unity` 的 importer,asset/runtime,以及新增 compute shader.
- 数据格式影响:
  - `.splat4d v2` 会新增/增大 per-segment base labels 数据,但 delta-v1 的目标是“updates 稀疏且总体更小”.
- 性能影响:
  - 运行时仅在目标帧变化时更新.
  - compute shader 不可用时会降级为静态 SH,不影响基本渲染链路.
