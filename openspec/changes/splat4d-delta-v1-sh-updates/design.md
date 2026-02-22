## Context

当前 `.splat4d format v2` 已具备 header+sections 的格式框架.
也已经定义并导出了 delta-v1 的 header.
但 exporter 端仍然在写“静态占位 delta”(多数 `updateCount=0`).
同时 gsplat-unity 端对 `.splat4d v2` 的 delta-v1 主要做校验.
解码与运行时渲染只使用 `startFrame=0` 的 base labels.
因此 SH rest 无法随时间变化,也无法验证 delta 的语义闭环.

本变更是跨仓库的.
- FreeTimeGsVanilla: 负责导出 `.splat4d v2`(含 SH codebook,base labels,delta bytes).
- `/workspace/gsplat-unity`: 负责导入 `.splat4d v2` 并在运行时应用 deltas.

一个现实约束是: 当前真实训练产物 ckpt 的 `splats["shN"]` 仍是静态 3D.
为了让“非 0 updates”可验证且可回归.
需要一个最小合成动态 ckpt.
用它来驱动 exporter 自检,并生成可给 Unity 测试使用的小样例.

## Goals / Non-Goals

**Goals:**
- exporter 端生成真实 delta-v1 updates.
  - 对比相邻帧 labels,只写 changed 的 `(splatId,newLabel)`.
  - 按 spec 约束保证 `splatId` 递增,`reserved=0`.
- `.splat4d v2` 支持 per-segment base labels.
  - 每段的 `SHLB` 能指向自己的 base labels bytes.
- Unity importer 保存 delta 所需数据(每段 base labels + delta bytes).
- Unity runtime 按 `TimeNormalized` 选帧应用 deltas.
  - 主线使用 GPU compute scatter 更新 `_SHBuffer` 的 SH rest.
  - compute 不可用时降级为静态 SH(保持 frame0),不崩溃.
- 保持兼容:
  - 静态 `shN` 或 `updateCount=0` 的文件,行为不变.

**Non-Goals:**
- per-segment codebook.
- 稳定 permutation/reorder.
- delta-v2 event stream.
- motion(R-VQ/RANS)压缩 section.
- SH0/DC 的逐帧变化与插值.

## Decisions

### 1) Unity 运行时采用 GPU compute scatter(主线)
选择原因:
- delta-v1 的 updates 是稀疏列表.
- 当 updateCount 增大时,CPU 多次小粒度 `SetData` 风险很高.
- GPU scatter 对“稀疏写入”更匹配,也更接近 DualGS 的扩展方向.

备选方案:
- CPU 解析 + 局部 `GraphicsBuffer.SetData`.
本次不作为主线实现.
仅作为 compute 不可用时的禁用/降级策略.

### 2) `.splat4d v2` 的 SHLB 改为 per-segment base labels
选择原因:
- delta-v1 的语义要求“每段起始帧有绝对状态”.
- 如果所有 segments 复用同一份 base labels,将无法表达“段起点 labels 不同”的场景.
- exporter 与 importer 都需要按 segment 正确加载 base labels.

折中点:
- 如果多个 segments 的 base labels bytes 完全相同,可以做 offset 去重.
这属于优化,不作为正确性前置条件.

### 3) 代码本期只做全局 persistent codebook
选择原因:
- per-segment codebook 会显著增加复杂度(聚类+导入+运行时选择).
- 本次目标是先把 delta-v1 的“语义闭环”打通.
- 后续如要更贴近 DualGS,再增量做 per-segment codebook.

### 4) 统一时间映射: `frame = round(t*(F-1))`
选择原因:
- 规则简单,可复现.
- 与 `.splat4d` 当前“按帧”存储 delta-v1 block 的模型匹配.
- 先不引入插值,避免额外复杂度与可视化歧义.

### 5) 回退 seek 的实现允许多种策略,但必须确定性
可选策略:
- Strategy A: forward-only 解析 + per-frame reverse cache(回退时应用反向更新).
- Strategy B: 直接从当前 segment 的 base labels 重新展开到目标帧(回退更慢,但实现更简单).

本次在 design 里只约束“结果正确且确定”.
具体选择由实现阶段根据现有 gsplat-unity 结构决定.

## Risks / Trade-offs

- [格式不一致] exporter 写出的 delta bytes 与 Unity 解析语义不一致 → 用 `--self-check-delta` + Unity EditMode 测试双重锁定,并用合成 ckpt 提供可复现样例.
- [资产体积增加] per-segment base labels 会增加 `.splat4d` 与 Unity 资产体积 → segmentLength 默认保持可控,并允许未来做 base labels 去重.
- [平台兼容性] 部分平台 compute kernel 可能不可用(例如 Metal 的常见限制) → 运行时提供明确降级路径,保持 frame0 静态渲染.
- [性能波动] updateCount 过大时 dispatch 频率与写入量上升 → 仅在 targetFrame 变化时更新,并复用 buffers,避免每帧分配.
- [跨仓库协同] exporter 与 importer/runtime 必须同步升级 → 通过 specs 固化二进制约束,并在 tasks 中明确两仓库改动点与验证命令.

## Migration Plan

1. FreeTimeGsVanilla: exporter 增加动态 `shN` 支持与 `--self-check-delta`.
2. FreeTimeGsVanilla: 提供合成动态 ckpt 脚本,用于生成最小 `.splat4d v2` 样例.
3. gsplat-unity: importer 增加 delta 数据承载字段,并 bump importer version,触发旧资产重导入.
4. gsplat-unity: 增加 compute shader 与 runtime 应用逻辑.
5. gsplat-unity: 新增 EditMode tests,锁定 delta-v1 解析语义.

回滚策略:
- 如 runtime 侧 compute shader 在某平台不稳定.
  - 先通过开关禁用动态 SH.
  - 仍保留 importer 的数据加载与资产字段,不影响静态渲染.

## Open Questions

- Unity 侧 `_SHBuffer` 的最终 layout(系数总数与 band offsets)需要以代码事实为准再落盘.
  - 如果当前实现不是 `3+5+7=15`,需要调整 compute 的写入映射.
- 回退 seek 的最优策略需要结合 gsplat-unity 现有 runtime bundle 实现来选型.
- 合成 ckpt 的字段最小集合需要以 exporter 的真实读取路径为准(避免缺字段导致验证脚本失效).
