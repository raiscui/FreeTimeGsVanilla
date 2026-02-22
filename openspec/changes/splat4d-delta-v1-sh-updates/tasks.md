## 1. Exporter: 动态 shN 输入与 labels 生成

- [x] 1.1 在 `tools/exportor/export_splat4d.py` 增加 `--shn-frame-axis 0|1`,并在 4D `shN` 判定歧义时 fail-fast
- [x] 1.2 扩展 `splats[\"shN\"]` shape 支持 `[F,N,K,3]`/`[N,F,K,3]`,并对 keep mask 做同步裁剪
- [x] 1.3 保持 SH codebook 为全局 persistent,实现跨帧采样并拟合 centroids(不做 per-segment codebook)
- [x] 1.4 逐帧分配 labels,生成 `labelsByFrame[F,N]`(u16),并保持静态 `shN` 行为不变

## 2. Exporter: delta-v1 真实 updates + per-segment base labels

- [x] 2.1 生成 per-segment base labels(段起始帧绝对 labels),并按 segment 写入 SHLB(允许可选去重,但不依赖)
- [x] 2.2 生成 delta-v1 blocks,对比相邻帧 labels 仅写 changed `(splatId,newLabel)`,并写入 SHDL(delta bytes)
- [x] 2.3 约束与校验: 每帧 block 内 `splatId` 严格递增,`reserved` 恒为 0,header 字段与 segment 元数据一致
- [x] 2.4 导出日志与自检: 输出 `changedPercent/avgUpdateCount/maxUpdateCount`,并做 segment 边界一致性校验(默认开启,异常直接报错)

## 3. Exporter: 自检与合成 ckpt(可重复验证)

- [x] 3.1 新增 `tools/` 下的合成动态 ckpt 脚本,生成可复现的动态 `shN` ckpt(N=1024,F=5),确保至少产生一次 `updateCount>0`
- [x] 3.2 在 exporter 增加 `--self-check-delta`,实现 delta-v1 解码复原 labels,并与内部 `labelsByFrame` 逐元素断言一致
- [x] 3.3 写一个最小 smoke 命令/说明(README 或脚本注释),用于一键验证“非 0 updates + self-check 通过”

## 4. Unity: 资产结构与 importer 持久化 delta

- [x] 4.1 修改 `/workspace/gsplat-unity/Runtime/GsplatAsset.cs`,增加可序列化的 delta segment DTO(BaseLabels+DeltaBytes+StartFrame+FrameCount)与 per-band 字段(隐藏字段即可)
- [x] 4.2 修改 `/workspace/gsplat-unity/Editor/GsplatSplat4DImporter.cs`,在 delta-v1 时读取每个 segment 的 SHLB labels 与 SHDL delta bytes,并填充到 `GsplatAsset`
- [x] 4.3 bump importer 版本(例如 `[ScriptedImporter(3,\"splat4d\")]`),确保旧资产自动重导入并填充新字段

## 5. Unity: Runtime 应用 delta(GPU compute scatter)

- [x] 5.1 新增 `/workspace/gsplat-unity/Runtime/Shaders/GsplatShDelta.compute`,实现按 updates 对 `_SHBuffer` 做 scatter 写入(按 band 分 kernel,并复用 centroids buffer)
- [x] 5.2 修改 `/workspace/gsplat-unity/Runtime/GsplatRenderer.cs`,增加帧选择逻辑(`targetFrame=round(TimeNormalized*(F-1))`)与 `currentFrame` 缓存,只在帧变化时更新
- [x] 5.3 实现 delta-v1 解析与应用逻辑(含 segment 边界处理),支持 forward/backward seek,并输出每帧 updates 列表
- [x] 5.4 增加 compute 不可用的降级路径(明确 warning + 禁用动态 SH + 保持 frame0),避免黑屏/崩溃

## 6. Unity: 测试与用户可见文档

- [x] 6.1 新增 `/workspace/gsplat-unity/Tests/Editor/GsplatSplat4DImporterDeltaV1Tests.cs`,构造最小 `.splat4d v2` 样例并验证 base+delta 复原 labels 序列(纯 C#,不跑 GPU)
- [x] 6.2 更新 `/workspace/gsplat-unity/CHANGELOG.md`,记录 `.splat4d v2 delta-v1` 运行时动态 SH 支持,以及 compute 不可用时的降级行为

## 7. 验证

- [x] 7.1 FreeTimeGsVanilla: 运行 `python3 -m compileall -q tools/exportor tools src datasets` 通过
- [x] 7.2 FreeTimeGsVanilla: 用合成 ckpt 导出 `.splat4d v2` 并 `--self-check-delta` 通过,且 delta 中至少 1 帧 `updateCount>0`
- [ ] 7.3 gsplat-unity: 运行 EditMode tests,确保新增 delta-v1 测试通过
- [ ] 7.4 (可选) Unity 手动 smoke: 播放/拖动 `TimeNormalized`,观察 SH rest 在有 updates 的帧发生变化且可重复
