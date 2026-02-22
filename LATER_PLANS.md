# LATER_PLANS

## 2026-02-20 13:36:01 UTC
- 暂无.

## 2026-02-20 14:37:35 UTC
- 若后续要进一步提高可移植性,可评估改用 uv 新版的 `tool.uv.extra-build-dependencies` 取代 `no-build-isolation-package` + `dependency-metadata` 组合,以减少“非隔离构建”的副作用.

## 2026-02-20 15:15:44 UTC
- 可考虑补一个最小 `pytest` 测试骨架(例如对关键函数做 import/shape/IO 的 smoke-test),让 PR 有可回归验证.
- 可考虑引入 `ruff`/`black`(或仅 `ruff format`)并写入 `pyproject.toml`,统一代码风格,降低 review 成本.

## 2026-02-21 01:59:48 UTC
- 如果后续需要更快/更稳的逐帧点云初始化,可以补一个 triangulation 后端: `colmap point_triangulator`(固定相机位姿)作为 fallback.
  - 思路参考: SpacetimeGaussians 的 `script/pre_no_prior.py` + `thirdparty/gaussian_splatting/helper3dg.py`.
- 如果后续需要更快的抽帧,可以给预处理脚本增加 ffmpeg 抽帧选项(比 OpenCV 更快,也更容易做 trim/crop/hdr/lut).

## 2026-02-21 09:19:20 UTC
- 建议补一个 exporter 的 `--self-check`:
  - 校验 `layout.width*height >= splatCount`.
  - 校验 per-frame 文件存在性与尺寸(抽样验证 1-2 帧即可).
  - 校验 `meta.json.streams.*` 关键字段长度与 frameCount 一致.

## 2026-02-21 12:39:32 UTC
- 若后续希望更贴近 DualGS(2409.08353)的压缩/流式策略,可以增量实现:
  - delta-v1 生成真实 update: 对比相邻帧 labels,只写 changed 的 `(splatId,newLabel)`(当前我们主要覆盖 FreeTimeGS 常见的“静态 SH”场景).
  - per-segment codebook: 当前 exporter 的 SH codebook 是全局静态,而 DualGS 在每个 segment 内做聚类以更贴合局部统计.
  - 稳定 permutation/reorder: 参考论文的“排序提升 codec 压缩率”思路,对 labels/indices 的 2D 图做一致性重排以提升 WebP/zip 压缩(需要同时保证 splat identity 或输出 permutation 给 importer).

## 2026-02-21 14:33:33 UTC
- DualGS 的两条“二期压缩路线”(对齐低端设备/超长序列)可以作为后续迭代方向:
  - SH change events: 用 quadruples `(t,d,i,k)` + sort + length encoding 代替 per-frame block,可作为 delta-v2.
  - Motion: R-VQ + "temporal quantization(11-bit in our setting)" + RANS(lossless),可考虑给 `.splat4d format v2` 增加可选的 motion 压缩 section.

## 2026-02-22 14:47:09 UTC
- “`.splat4d` delta-v1 真实 updates + Unity runtime 应用”已实现落地(Exporter + gsplat-unity importer/runtime).
- 待在具备 Unity Editor 的环境里补跑验证:
  - gsplat-unity EditMode tests(对应 `openspec/changes/splat4d-delta-v1-sh-updates/tasks.md` 的 7.3).
  - (可选) Unity 手动 smoke: 播放/拖动 `TimeNormalized` 观察 SH rest 变化(7.4).
