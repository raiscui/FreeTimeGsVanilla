# Capability: gsplat-unity-splat4d-delta-v1-runtime

## ADDED Requirements

### Requirement: Unity importer MUST 读取并持久化 delta-v1 所需数据
当导入 `.splat4d format v2` 且 `labelsEncoding=delta-v1` 时,importer MUST:
- 读取每个 segment 的 base labels(来自 SHLB 指向的 labels bytes).
- 读取每个 segment 的 delta bytes(来自 SHDL section,包含 delta-v1 header+body).
- 把这些数据以“每 segment 一份”的形式持久化到 `GsplatAsset` 中(或等价的可序列化载体).

每个 segment 的元数据 MUST 至少包含:
- `StartFrame`
- `FrameCount`
- `BaseLabels`(长度 MUST 等于 `splatCount`)
- `DeltaBytes`(原始字节,用于运行时解析)

#### Scenario: 导入后资产包含完整的 segments 数据
- **WHEN** 导入的 `.splat4d v2` 文件包含多个 delta segments
- **THEN** `GsplatAsset` MUST 持有相同数量的 delta segments,且每段 `BaseLabels.Length == splatCount`

### Requirement: Unity importer MUST 用 frame0 初始化初始 SH 状态
在 importer 阶段,系统 MUST 能初始化“播放起点”的 SH rest 数据.
默认播放起点为 frame0.
因此 importer MUST 用 frame0 的 base labels(以及对应 centroids)生成初始 SH rest,用于首次上传到 GPU buffer.

#### Scenario: 首次渲染对应 frame0
- **WHEN** `.splat4d` 被导入并首次被渲染,且尚未进行任何时间推进
- **THEN** SH rest 状态 MUST 等价于 frame0 的 labels 对应的 centroids

### Requirement: Runtime MUST 按 `TimeNormalized` 映射到离散帧(Nearest)
运行时 `GsplatRenderer`(或等价组件) MUST 使用统一的帧映射规则:
- `targetFrame = round(TimeNormalized * (FrameCount - 1))`
- `targetFrame` MUST clamp 到 `[0, FrameCount-1]`

系统 MUST NOT 对 SH rest 做线性插值.
仅在 `targetFrame` 发生变化时才触发更新.

#### Scenario: `TimeNormalized` 映射到最近帧
- **WHEN** `FrameCount=3` 且 `TimeNormalized=0.6`
- **THEN** `targetFrame` MUST 为 1(即 `round(0.6*(3-1)) == 1`)

### Requirement: Runtime MUST 支持前进与回退(非单调时间)且结果确定
运行时 MUST 支持 `TimeNormalized` 非单调变化(例如拖动时间条回退).
无论时间是前进还是回退,最终的 SH rest 状态 MUST 与目标帧的 labels 一致.
同一时间点重复切换(前进后回退再前进) MUST 得到可重复的确定性结果.

#### Scenario: 回退 seek 后状态正确且可重复
- **WHEN** `TimeNormalized` 从较大值回退到较小值,并多次往返切换到同一目标帧
- **THEN** 每次到达该目标帧时,SH rest 状态 MUST 相同且与该帧 labels 一致

### Requirement: Compute shader 可用时 MUST 使用 GPU scatter 更新 SH rest
当运行环境支持 compute shader 且相关 kernel 可用时,运行时 MUST:
- 把某一帧的 updates 表示为稀疏列表(每条至少包含 `splatId` 与 `newLabel`).
- 通过 compute shader 对 `_SHBuffer` 执行 scatter 写入.
- 仅更新出现在 updates 列表中的 splat 的 SH rest 系数.

#### Scenario: updates 列表只影响对应 splat
- **WHEN** 某帧 updates 只包含 1 个 `splatId`
- **THEN** GPU 上只有该 `splatId` 的 SH rest 系数发生变化,其它 splat 的 SH rest MUST 保持不变

### Requirement: Compute shader 不可用时 MUST 降级为静态 SH(不崩溃)
当 compute shader 不可用(例如 `SystemInfo.supportsComputeShaders=false`)或 kernel 不可用时,系统 MUST:
- 输出明确 warning(便于定位).
- 禁用动态 SH 更新.
- 保持 frame0 的静态 SH 渲染结果.

系统 MUST NOT 因此崩溃或黑屏.

#### Scenario: compute 不支持时保持静态渲染
- **WHEN** 运行环境不支持 compute shader
- **THEN** 时间变化不会触发崩溃,并且 SH rest 状态 MUST 始终保持为 frame0

### Requirement: Unity MUST 提供 EditMode 测试验证 delta-v1 解析语义
gsplat-unity MUST 提供 EditMode(编辑器)测试,用于锁定以下行为:
- importer 能导入一个最小 `.splat4d v2` 样例(含 delta-v1).
- 测试内置的纯 C# delta-v1 解释器能从 `BaseLabels + DeltaBytes` 复原每帧 labels.
- 复原的 labels 序列 MUST 与测试定义的期望序列完全一致.

该测试 MUST 不依赖 GPU 执行(不要求 dispatch compute).

#### Scenario: 测试复原 labels 序列与期望一致
- **WHEN** 运行 EditMode tests
- **THEN** delta-v1 的复原 labels 序列 MUST 与测试期望完全一致
