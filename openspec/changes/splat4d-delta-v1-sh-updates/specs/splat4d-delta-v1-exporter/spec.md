# Capability: splat4d-delta-v1-exporter

## ADDED Requirements

### Requirement: 支持静态与动态 `splats["shN"]` 输入
exporter MUST 支持两类 `splats["shN"]`:
1. 静态: shape 为 `[N,K,3]`.
2. 动态: shape 为 `[F,N,K,3]` 或 `[N,F,K,3]`.

当输入为静态 `shN` 时,exporter MUST 保持兼容行为.
即使 `--shn-labels-encoding=delta-v1`,也 MUST 导出全 0 的 delta blocks(updateCount 全为 0).

#### Scenario: 静态 `shN` 保持兼容
- **WHEN** `splats["shN"]` 的 shape 为 `[N,K,3]`
- **THEN** exporter 只导出 base labels,并且所有 delta blocks 的 `updateCount` MUST 为 0

#### Scenario: 动态 `shN` 允许产生非 0 updates
- **WHEN** `splats["shN"]` 的 shape 为 `[F,N,K,3]` 且 `F == --frame-count`
- **THEN** exporter MUST 按帧生成 labels,并且 delta blocks MAY 出现 `updateCount>0`

### Requirement: 动态 `shN` 的帧轴判定与 `--shn-frame-axis`
当 `splats["shN"]` 为 4D 时,exporter MUST 能把“帧轴”判定出来.
判定规则 MUST 满足:
- 若仅有一个轴能与 `--frame-count` 匹配,exporter MUST 自动使用该轴作为帧轴.
- 若存在歧义(例如两个轴都能与 `--frame-count` 匹配),exporter MUST 失败并提示用户显式提供 `--shn-frame-axis 0|1`.
- 若两个轴都无法与 `--frame-count` 匹配,exporter MUST 失败并输出清晰错误信息.

#### Scenario: 自动判定帧轴
- **WHEN** `splats["shN"]` 的 shape 为 `[F,N,K,3]` 且 `F == --frame-count`
- **THEN** exporter MUST 把 axis=0 视为帧轴,并继续导出

#### Scenario: 判定歧义时 fail-fast
- **WHEN** `splats["shN"]` 的两个候选轴都能匹配 `--frame-count`
- **THEN** exporter MUST 退出并要求提供 `--shn-frame-axis 0|1`

### Requirement: 动态 `shN` MUST 与 keep mask 同步裁剪
当 exporter 启用基于 opacity 的 splat 过滤(例如 `--base-opacity-threshold`)时,会生成 keep mask 并改变最终 splatCount.
若 `splats["shN"]` 为动态 4D,exporter MUST 对所有帧同步应用同一个 keep mask.
裁剪后 `shN` 的 splat 维度 MUST 与最终导出的 `splatCount` 完全一致.

#### Scenario: keep mask 同步应用到所有帧
- **WHEN** exporter 过滤掉某些 splat,导致 `splatCount` 发生变化
- **THEN** 所有帧的 `shN` MUST 使用同一个 keep mask 裁剪,并与导出的 `splatCount` 一致

### Requirement: SH codebook MUST 是全局 persistent(非 per-segment)
当 `--sh-bands>0` 时,exporter MUST 为每个 band 生成 codebook(centroids).
同一 band 的 codebook MUST 在全文件范围内保持一致.
exporter MUST NOT 为不同 segment 生成不同 codebook.

#### Scenario: 多 segments 仍共享同一份 codebook
- **WHEN** `--delta-segment-length` 使得导出文件包含多个 segments
- **THEN** 同一 band 的 `labelCount(codebookCount)` MUST 对所有 segments 一致,并使用同一套 centroids

### Requirement: exporter MUST 生成逐帧 labels,并满足范围约束
exporter MUST 为每个 band 生成逐帧 labels.
对任意帧 `t` 与 splat `i`,label 值 MUST 满足:
- 类型为 `u16`.
- `0 <= label < labelCount`.

#### Scenario: labels 满足 `u16` 与范围约束
- **WHEN** exporter 生成某个 band 的 labels
- **THEN** 每个 label MUST 在 `[0,labelCount)` 内,并可被安全编码为 `u16`

### Requirement: delta-v1 的二进制布局与语义 MUST 严格符合约定
当 `--shn-labels-encoding=delta-v1` 且存在动态 labels 时,exporter MUST 生成 delta-v1 updates.
其二进制布局(小端) MUST 满足:
- Header:
  - `magic`: 8 bytes,值 MUST 为 `SPL4DLB1`
  - `version`: u32,值 MUST 为 1
  - `segmentStartFrame`: u32
  - `segmentFrameCount`: u32
  - `splatCount`: u32
  - `labelCount`: u32
- Body:
  - 按 segment 内帧顺序,从 `startFrame+1` 到 `startFrame+frameCount-1`
  - 每帧 block:
    - `updateCount`: u32
    - `updateCount` 个 update:
      - `splatId`: u32
      - `newLabel`: u16
      - `reserved`: u16,值 MUST 为 0

delta 的语义 MUST 是“相对上一帧”的增量.
同一帧 block 内的 `splatId` MUST 严格递增.

#### Scenario: 某 splat 在相邻帧发生 label 变化时写入 update
- **WHEN** frame `t-1` 与 `t` 的 labels 存在差异,且 splatId 为 `i` 的 label 从 `a` 变为 `b`
- **THEN** frame `t` 的 block MUST 包含一条 update,其 `(splatId,newLabel)` 为 `(i,b)`,并且该帧内 `splatId` 仍保持严格递增

### Requirement: `.splat4d v2` 的 base labels MUST 支持 per-segment
exporter MUST 能表达“每个 segment 都有自己的 base labels(起始帧的绝对 labels)”.
对任意 segment:
- base labels MUST 等于该 segment 的 `labelsByFrame[startFrame]`.
- `.splat4d v2` 的 section table MUST 指向正确的 labels bytes.
- 若多个 segments 的 base labels bytes 完全一致,offset/length MAY 复用(去重可选),但 SHLB 记录 MUST 逐 segment 存在.

#### Scenario: 多 segments 时每段都有 base labels
- **WHEN** 导出文件包含多个 segments
- **THEN** 每个 segment MUST 具备一份可独立读取的 base labels(通过其 SHLB 元数据定位),并等于该段 `startFrame` 的 labels

### Requirement: `--self-check-delta` MUST 能验证 delta-v1 可逆
当启用 `--self-check-delta` 时,exporter MUST 在导出后进行一致性断言:
- 从每个 segment 的 base labels 开始.
- 逐帧应用 delta blocks.
- 复原得到每帧 labels.
- 复原结果 MUST 与 exporter 内部计算得到的逐帧 labels 完全一致(逐元素相等).

若不一致,exporter MUST 失败并输出清晰的错误信息.

#### Scenario: 自检通过时导出成功
- **WHEN** exporter 对导出的 delta-v1 进行自检且复原 labels 与内部 labels 一致
- **THEN** exporter MUST 正常退出,并报告自检通过

### Requirement: 仓库 MUST 提供最小“合成动态 ckpt”以验证非 0 updates
本仓库 MUST 提供一个小型工具脚本,用于生成可被 exporter 读取的“合成动态 ckpt”.
该 ckpt MUST 满足:
- `shN` 为 4D,并且至少在 1 个相邻帧之间产生 label 变化,从而让 delta-v1 出现 `updateCount>0`.
- 生成过程 MUST 可复现(固定 seed 或等价手段).

#### Scenario: 合成 ckpt 导出后至少出现一次 `updateCount>0`
- **WHEN** 使用合成动态 ckpt 运行 exporter 并启用 delta-v1
- **THEN** 导出的任意 band 的 delta blocks 中 MUST 至少出现 1 帧 `updateCount>0`
