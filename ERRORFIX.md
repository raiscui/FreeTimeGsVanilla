# ERRORFIX

## 2026-02-20 13:36:01 UTC - uv sync 构建 fused-ssim 失败

### 问题
- 执行 `uv sync --locked` 失败.
- 失败点: 构建 `fused-ssim` 时抛 `ModuleNotFoundError: No module named 'torch'`.

### 原因
- `fused-ssim` 在构建阶段导入 `torch`,但未声明为 build 依赖.
- uv 默认的 build isolation 环境没有 `torch`,因此构建失败.

### 修复思路
- 用 uv 的 `no-build-isolation-package` 让该包关闭隔离构建,确保构建阶段能访问已安装的 `torch`.

### 验证方式
- 重新执行 `uv sync --locked`,确认依赖能安装完成并生成 `.venv/`.

### 实际修复(落地)
- 升级 uv: `python -m pip install -U uv`(从 `0.7.12` 升级到 `0.10.4`).
- 安装系统编译器: `apt-get install -y build-essential`(否则 `lapjv` 会因缺少 `c++` 构建失败).
- 更新 `pyproject.toml`:
  - 增加 `tool.uv.required-version = ">=0.10.4"`.
  - 使用 `no-build-isolation-package = ["fused-ssim", "gsplat"]` 避免这两个包在隔离构建环境缺 `torch`.
  - 提供 `[[tool.uv.dependency-metadata]]` 以减少解析阶段触发构建.
  - 移除未使用的 `torch_scatter` 依赖,避免额外的 CUDA/C++ 编译负担.
- 更新锁文件: `uv lock`.

### 验证结果
- `uv sync --locked` 成功完成.
- 运行导入检查通过:
  - `.venv/bin/python -c "import torch, fused_ssim, gsplat; print(torch.__version__)"`.

## 2026-02-21 05:43:50 UTC - bar-release 预处理时 COLMAP feature_extractor 被 SIGKILL

### 问题
- 执行 `python src/preprocess_mp4_freetimegs.py ...` 时,在跑 COLMAP 的 `feature_extractor` 阶段异常退出.
- Python 抛 `subprocess.CalledProcessError`,并显示:
  - `colmap feature_extractor ... died with <Signals.SIGKILL: 9>`

### 原因
- bar-release 的输入图片分辨率较高(2110x3760).
- COLMAP 默认 `SiftExtraction.num_threads=-1` 会使用满线程提特征.
  - CPU SIFT 每线程会占用大量内存,高分辨率下容易触发 OOM.
  - 进程随后被系统 OOM killer 以 SIGKILL(9) 强制终止.

### 修复思路
- 让 COLMAP 的 SIFT 提取变得"更稳":
  - 限制 SIFT 线程数(减少峰值内存).
  - 限制 SIFT 最大输入边长(减少单线程内存).
  - 同时限制 matcher 的线程数(避免额外 CPU/RAM 压力).

### 实际修复(落地)
- 修改 `src/preprocess_mp4_freetimegs.py`:
  - 新增 CLI 参数:
    - `--colmap-sift-num-threads`(默认 1)
    - `--colmap-sift-max-image-size`(默认 2000)
    - `--colmap-match-num-threads`(默认 1)
  - 在 `_run_colmap_mapper(...)` 调用中传入:
    - `--SiftExtraction.num_threads`
    - `--SiftExtraction.max_image_size`
    - `--SiftMatching.num_threads`

### 验证方式
- 对 bar-release 做快速冒烟(帧段 [0,5)):
  - `python src/preprocess_mp4_freetimegs.py ... --start-frame 0 --end-frame 5 --overwrite`
  - 确认 COLMAP 能跑完 feature_extractor/exhaustive_matcher/mapper,并进入 RoMA 三角化阶段.

### 验证结果
- COLMAP 在该参数配置下稳定跑通,不再出现 SIGKILL.
- 预处理成功产出逐帧点云:
  - `results/bar-release_smoke_work/triangulation/points3d_frame000000.npy` 等.

## 2026-02-22 02:33:37 UTC - `.splat4d` gaussian 导出无 header 导致 Unity 走 legacy v1

### 问题
- 用户导出 `.splat4d` 期望表达 timeModel=2(gaussian,time=mu_t,duration=sigma),但生成的文件无 `SPL4DV02` header.
- Unity importer 如果仅靠 header 判定 v1/v2,会走 legacy v1 导入,并按 window(time0+duration)裁剪,表现为"薄层/稀疏".

### 原因
- `tools/exportor/export_splat4d.py` 有两套正交概念:
  - `--splat4d-version`: time/duration 语义(1=window,2=gaussian).
  - `--splat4d-format-version`: 文件格式(1=legacy无 header,2=header+sections).
- 旧默认 `--splat4d-format-version=1`,导致只写 `--splat4d-version 2` 时仍会输出 legacy 无 header 文件,从而触发 Unity 侧误判.

### 修复思路
- 从根源消除默认值误用:
  - 增加 `--splat4d-format-version 0=auto` 并作为默认.
  - 当 `--splat4d-version=2` 时,auto 自动选择 format v2(header+sections),确保带 `SPL4DV02` 与 timeModel 字段.
- 对高风险组合保留兼容但给 warning:
  - `--splat4d-version 2` + `--splat4d-format-version 1` 仍允许导出,但在 stderr 打印醒目 warning.

### 实际修复(落地)
- 修改 `tools/exportor/export_splat4d.py`:
  - `--splat4d-format-version` 支持 `0|1|2`,默认 `0`.
  - auto 规则: `splat4d-version=2` 或 `sh-bands>0` 时选择 format v2,否则选择 format v1.
- 同步更新文档:
  - `README.md`
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md`

### 验证方式
- 语法冒烟:
  - `python3 -m compileall -q src datasets tools/exportor`
- 最小导出 + magic 检查:
  - 用最小 ckpt 导出 `--splat4d-version 2`(不指定 format),确认输出前 8 bytes 为 `SPL4DV02`.
