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

## 2026-02-21 09:45:50 UTC
- `.sog4d` exporter 后续可升级实现 `meta.json.version=2` 的 per-band palette:
  - `sh1/sh2/sh3` 分别拟合 codebook + labels(delta-v1),质量与码率通常更好.
- scipy `kmeans2` 可能出现 empty cluster 的 warning:
  - 可考虑增加 retry(重采样/换 seed)或 fallback(自动降 `shn_count`),让默认行为更“稳”.
