# 笔记: uv sync 构建失败(PEP517 build isolation)

## 现象
- `uv sync --locked` 在构建 git 依赖 `fused-ssim` 时失败.
- 报错核心为 `ModuleNotFoundError: No module named 'torch'`.

## 根因分析
- `uv sync` 默认使用 PEP 517 build isolation.
- `fused-ssim` 的构建脚本在构建阶段导入了 `torch`.
- 但它没有把 `torch` 声明为构建期依赖(build-system.requires),导致隔离构建环境里没有 `torch`,从而失败.

## 可行修复
### 修复A(长期): 修正第三方包的构建依赖声明
- 需要 fork 或 patch `fused-ssim` 的构建配置.

### 修复B(当前推荐): uv 配置 no-build-isolation
- 在 `pyproject.toml` 的 `[tool.uv]` 配置 `no-build-isolation-package = ["fused-ssim"]`.
- uv 会进行两阶段安装: 先把运行依赖(包括 `torch`)装进环境,再对该包关闭隔离构建.

## 2026-02-20 14:37:35 UTC 追加: 实际落地与踩坑

### uv 版本差异
- `uv 0.7.12` 不支持 `tool.uv.extra-build-dependencies`,导致无法在 build isolation 环境里补齐 `torch` 作为构建依赖.
- 升级到 `uv 0.10.4` 后,下载侧自带重试机制,并且整体流程更稳定.

### 系统依赖
- `lapjv` 会编译 C++ 扩展,需要系统编译器,否则会报 `c++: No such file or directory`.
- Ubuntu 22.04 可通过 `apt-get install -y build-essential` 解决.

### 项目层面的优化
- `torch_scatter` 在本仓库代码中未被引用,属于高成本依赖(需要编译 CUDA/C++).
- 已从 `pyproject.toml` 移除并同步更新 `uv.lock`,显著降低首次安装时间与失败概率.

## 2026-02-20 15:15:44 UTC 追加: 贡献者指南(AGENTS.md)素材

### 仓库现状速记
- 依赖管理: `pyproject.toml` + `uv`(默认 `.venv/`,命令 `uv sync --locked`).
- 代码结构: `src/` 为训练/Viewer 脚本,`datasets/` 为 COLMAP I/O 与数据集工具,`run_*.sh` 为快捷入口.
- 测试现状: 未发现 `pytest`/`tests/` 目录,只能提供 smoke-check(例如 `python -m compileall src datasets`).
- 提交风格: Git 历史以简短祈使句为主("Add ..."/"Fix ..."/"Refactor ..."),无统一 conventional commits.

## 2026-02-20 16:55:24 UTC 追加: FreeTimeGsVanilla vs FreetimeGS_NO 对比笔记

### 结论摘要(先给结论再给证据)
- `FreeTimeGsVanilla` 更偏“最小可跑 + 更接近论文的显式速度/时长建模”,但把“逐帧三角化点云”留给外部工具链,本仓库只消费 `points3d_frame%06d.npy`.
- `FreetimeGS_NO` 更偏“3DGS(GraphDECO)代码系的非官方改造版”,内置了从多相机 mp4 抽帧和跑 COLMAP(首帧)的脚本.
- `FreetimeGS_NO` 的代码层面支持按时间帧组织数据(`frame000000/`,`frame000001/`...),每帧都可以有自己的相机位姿.
  - 但它提供的默认预处理脚本会把首帧的 `sparse/` 复制到其它帧,因此“默认工具链”实际上假设相机位姿静态.
- `FreetimeGS_NO` 没看到“逐时间帧重新三角化点云”的工具链脚本,更多是复用 COLMAP 首帧点云做初始化.

### 证据(关键文件与关键片段)
- `FreetimeGS_NO/README.md` 明确这是“非官方实现”,并提到当前分支重点:
  - "用一个生命周期函数代替了独立的 帧-透明度 变量"
  - "正在开发: 让高斯点沿着直线运动优化"
- `FreetimeGS_NO/scene/dataset_readers.py` 的 `read4DGSSceneInfo(...)` 会枚举 `frame*` 目录并把 `time_idx` 写入每个 Camera:
  - `frame_dirs = sorted([d for d in os.listdir(path) if d.startswith('frame') ...])`
  - `for time_idx, frame_dir in enumerate(frame_dirs): ... readColmapCameras(... time_idx=time_idx)`
  - 这说明它的数据组织是“按时间帧一帧一个 COLMAP”而不是“按相机一个文件夹”.
- `FreetimeGS_NO/data_preprocess.py` 的 `get_colmap_single(...)`:
  - `offset == 0` 时跑 `python utils/convert.py ...`(会执行 COLMAP feature/matcher/mapper).
  - 然后调用 `copy_sparse_to_frames(...)` 把首帧的 `sparse/` 复制到其它帧目录.
  - `offset != 0` 时只跑 `colmap image_undistorter ... --input_path <scene>/distorted/sparse/0` 并删除 `inputs/`.
  - 这意味着默认链路不会为每个时间帧重新估计相机位姿/三角化新点云.
- `FreeTimeGsVanilla` 的输入假设在 `README.md` 与代码里更“显式”:
  - `src/combine_frames_fast_keyframes.py` 直接读 `points3d_frame%06d.npy`/`colors_frame%06d.npy`,并用 `t -> t+1` 的 kNN 去估速度.
  - `datasets/FreeTime_dataset.py` 读取 `data_dir/images/<cam_folder>/<frame>` 并从 `data_dir/sparse/0` 取每个相机的固定 `camtoworld`.


# 笔记: EasyVolcap 与 SpacetimeGaussians 的视频/图片预处理脚本调研

## 2026-02-21 01:59:48 UTC

## 目标
- 识别两仓库中可借鉴的“视频抽帧/图片整理/COLMAP 运行/相机参数与稀疏点云导出”脚本.
- 给出具体文件路径,便于直接打开阅读或迁移逻辑.

## 结论先行
- `zju3dv/EasyVolcap` 的脚本更偏“工程化工具箱”: ffmpeg 抽帧,子序列裁剪,跑 COLMAP,以及 COLMAP<->EasyVolcap 格式转换.
- `oppo-us-research/SpacetimeGaussians` 的脚本更偏“数据集专用 pipeline”: 用 OpenCV 抽帧,按 frame 组织 COLMAP 工程,并用 `colmap point_triangulator` 在固定相机位姿下逐帧三角化点云(非常贴合需要 per-frame 点云的场景).

## EasyVolcap: 值得直接借鉴的脚本

### 1) 多相机视频抽帧(ffmpeg)
- `scripts/preprocess/extract_videos.py`
  - 核心能力: 遍历 `data_root/videos/` 下每路视频,输出到 `data_root/images/<video_stem>/%06d.jpg`.
  - 额外能力: 可选 trim(`-ss/-t`),crop+scale,tonemap(hdr),lut.
  - 关键点: `args.cmd` 是模板字符串,最终用 `args.cmd.format(video_path=..., output_path=...)` 生成 ffmpeg 命令并执行.

### 2) 一键跑 COLMAP(含 undistort + 导出 txt/ply)
- `scripts/colmap/run_colmap.py`
  - 典型流程: feature_extractor -> matcher -> mapper(估位姿) -> bundle_adjuster -> model_orientation_aligner -> image_undistorter -> model_converter(TXT/PLY).
  - 还集成了“把前景 mask 反相做 bkgd_mask 并膨胀”的逻辑,用于给 COLMAP 提供 mask(可选).

### 3) 单帧跨相机整理成 COLMAP 输入(静态位姿常用)
- `scripts/colmap/arrange_images.py`
  - 把 `images/<cam>/<frame>` 的同一帧,symlink 到 `colmap/images/<cam>.jpg`.
  - 用途: 只用一帧做静态相机标定/位姿估计,后续把相机位姿复用到全序列.

### 4) COLMAP 相机参数导出为 EasyVolcap 格式
- `scripts/colmap/colmap_to_easyvolcap.py`
  - 读取 `cameras.(bin|txt)` 与 `images.(bin|txt)`,把内参/外参写到 EasyVolcap 的 `intri.yml/extri.yml`(由 `write_camera(...)` 落盘).

### 5) 其它偏“素材整理”的脚本
- `scripts/preprocess/extract_subseq.py`: 抽取子序列并重命名帧号,便于把长视频裁成训练子集.
- `scripts/tools/compress_videos.py`: 统一压缩 mp4(内部仍是 ffmpeg,封装在 `generate_video`).

### License
- EasyVolcap 根目录 `license` 为 MIT.

## SpacetimeGaussians: 值得借鉴的脚本

### 1) 从 mp4 抽帧(OpenCV VideoCapture)
- `script/pre_n3d.py`
  - `extractframes(videopath, startframe, endframe, downscale, ext)` 用 `cv2.VideoCapture` 逐帧读,直接写 `{frame_idx}.{ext}`.
  - 优点: 不依赖 ffmpeg,可控性强.
  - 缺点: 速度/稳定性通常不如 ffmpeg(尤其是硬件解码与复杂编码格式).

### 2) “无先验位姿”的逐帧点云三角化 pipeline(非常关键)
- `script/pre_no_prior.py`
  - 逻辑概述(脚本注释里写得很清楚):
    1. 视频抽帧到 `frames/`.
    2. 每帧把多相机图软链接到 `point/colmap_{t}/input/`.
    3. 选一个参考帧 `colmap_{ref}` 跑一次完整 COLMAP(得到相机位姿+内参).
    4. 把参考帧的相机参数写回其它帧的 `input.db` 与 `manual/*.txt`,作为“固定相机”的先验.
    5. 对其它帧跑 `colmap point_triangulator` 得到每帧稀疏点云.

### 3) 直接封装 COLMAP 命令: point_triangulator(固定相机位姿)
- `thirdparty/gaussian_splatting/helper3dg.py`
  - `getcolmapsinglen3d(...)`/`getcolmapsingleimdistort(...)`/`getcolmapsingleimundistort(...)`:
    - feature_extractor + exhaustive_matcher + point_triangulator(输入 `manual/` 作为已知相机) + image_undistorter.
  - 这段逻辑很适合“相机静态,每帧都要点云”的需求.

### 4) Python 侧写 COLMAP 数据库与 manual 模型文件
- `thirdparty/colmap/pre_colmap.py`: 基于 COLMAP 官方 `database.py` 改的 `COLMAPDatabase`,可 `add_camera/add_image` 写入 prior pose.
- `script/utils_pre.py`: `write_colmap(...)` 把 `manual/images.txt + cameras.txt` 和 `input.db` 一起生成.

### 5) 畸变/去畸变的专用数据集 pipeline
- `script/pre_immersive_distorted.py`/`script/pre_immersive_undistorted.py`
  - 针对 `models.json` 描述的 fisheye 相机,用 `cv2.fisheye.initUndistortRectifyMap` + `cv2.remap` 生成 COLMAP 输入图像.

### License
- 根目录 `LICENSE` 为 MIT,但额外声明 “受 gaussian-splatting 的 use limitation 影响”,需要同时关注 `thirdparty/gaussian_splatting/LICENSE.md`.

## 对 FreeTimeGsVanilla 的直接启发(不落地实现,先记要点)
- 如果你输入是“多相机视频”: `EasyVolcap/scripts/preprocess/extract_videos.py` 的 ffmpeg 抽帧逻辑更像生产可用版本(支持 trim/crop/hdr/lut).
- 如果你要“每帧一个 points3d_frame%06d.npy”: SpacetimeGaussians 的 `pre_no_prior.py + helper3dg.py` 展示了一个非常直接的做法:
  - 先用参考帧跑一次 COLMAP 求相机位姿.
  - 然后对每个时间帧只做 `point_triangulator`(固定相机),快速得到 per-frame 稀疏点.


## 2026-02-21 03:41:48 UTC 追加: mp4->4DGS(RoMA 全帧初始化)落地要点

### romatch(RoMA) 最小可用 API 速记
- 典型用法(来自 RoMa README 的 demo 片段,已通过 import 验证):
  - `from romatch import roma_outdoor`
  - `roma_model = roma_outdoor(device=device)`
  - `warp, certainty = roma_model.match(imA_path, imB_path, device=device)`
  - `matches, certainty = roma_model.sample(warp, certainty)`
  - `kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)`

### 文档工具链补齐(用于 Mermaid 校验与终端可读渲染)
- 增加本地工具:
  - `tools/mermaid-validator`: 校验 Markdown 里的 ` ```mermaid ` code block 语法(用 beautiful-mermaid 渲染作为判据).
  - `tools/beautiful-mermaid-rs --ascii`: 把 Mermaid 渲染成 Unicode 文字图,便于在终端/对话中阅读.

### 关键代码改造点
- `src/combine_frames_fast_keyframes.py`:
  - 新增 `--mode {keyframes,all_frames}`.
  - 统一 `--frame-end` 语义为 end exclusive.
  - all_frames 模式会把每帧点云都写入 NPZ,用于 `use_stratified_sampling=True`.


## 2026-02-21 09:19:20 UTC 追加: checkpoint(.pt) -> `.sog4d` 导出器落地记录

### 背景与目标
- 目标: 把 FreeTimeGS checkpoint(`ckpt_*.pt`)导出为 Unity 可导入播放的 `.sog4d`(ZIP bundle).
- 参考规格与施工图:
  - `tools/exportor/spec.md`
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md`
  - `tools/exportor/export_splat4d.py`

### 实现结论(当前最小可用)
- 已新增脚本: `tools/exportor/export_sog4d.py`
- 当前 exporter 默认实现 `bands=0`:
  - streams: `position`/`scale`/`rotation`/`sh`(sh0+opacity)
  - 不导出 SH rest(`shN`),先保证“能导入能播放”.
- 关键点:
  - position: per-frame range + u16 hi/lo 两张 lossless WebP.
  - scale: kmeans2(log-domain) 拟合 codebook + u16 indices WebP(静态内容复用).
  - rotation: quat(wxyz) -> u8 WebP(静态内容复用).
  - sh0.webp: RGB 为 `f_dc` 的 codebook 索引,A 为每帧 `opacity(t)`.

### 实际导出(你指定的 ckpt)
- 输入 ckpt:
  - `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`(num_GS=1,335,131)
- 输出 `.sog4d`:
  - `results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full.sog4d`
  - 约 1.1G

### 关键命令(可复现)
```bash
source .venv/bin/activate
python tools/exportor/export_sog4d.py \
  --ckpt-path results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt \
  --output-path results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full.sog4d \
  --frame-count 61 \
  --layout-width 2048 \
  --webp-method 0 \
  --zip-compression stored \
  --overwrite
```

### 快速冒烟(更快,用于验证工具链)
- `results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_f5_k50k.sog4d`
- 参数: `--frame-count 5 --max-splats 50000 --layout-width 1024`


## 2026-02-21 09:45:50 UTC 追加: `.sog4d` exporter 支持 SH rest(bands>0, v1 + delta-v1)

### 新增能力
- `tools/exportor/export_sog4d.py` 新增参数:
  - `--sh-bands 0..3`
  - `--shn-count`(v1 palette 的码字数)
  - `--shn-centroids-type f16|f32`
  - `--shn-labels-encoding full|delta-v1`(默认 delta-v1)
- 当 `--sh-bands > 0` 时,会额外写入:
  - `shN_centroids.bin`: little-endian 的 `float16/float32`,无 header.
  - `frames/00000/shN_labels.webp`: u16 labels 的 RG 小端数据图.
  - `sh/shN_delta_00000.bin`: delta-v1.由于 FreeTimeGS 的 SH 通常静态,每帧 `updateCount=0`,因此 delta 文件极小.

### 默认值调整(让大数据更稳)
- `--shn-count` 默认 512(而不是 4096).
- `--shn-codebook-sample` 默认 100k,`--shn-kmeans-iters` 默认 10.
  - 原因: scipy 的 `kmeans2` 属于 CPU 侧实现,K 和 sample 过大时会非常慢甚至不可用.
  - 需要更高质量时,可以手动调大这些参数,但要预期导出耗时显著增加.

### 实际导出(你指定的 ckpt,含 SH rest)
- 冒烟(5 帧 + 5 万 splats, bands=3):
  - `results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_f5_k50k_sh3_v1delta.sog4d`
- 全量(61 帧 + 133 万 splats, bands=3, shNCount=512):
  - `results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full_sh3_v1delta_k512.sog4d`

### 你问的两个 ckpt 的区别(核心是 Gaussian 数量不同)
- `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`:
  - `n_gaussians=1,335,131`(文件约 978MB)
- `results/bar-release_result_run2/ckpts/ckpt_29999.pt`:
  - `n_gaussians=199,958`(文件约 147MB)
- 两者 step 都是 29999,但训练时的初始化/采样密度不同,导致最终点数差距很大.


## 2026-02-21 09:57:20 UTC 追加: `.splat4d` exporter 支持 v2(gaussian 时间核)

### 背景
- `tools/exportor/export_splat4d.py` 原本只有 v1 语义:
  - 把 FreeTimeGS 的时间高斯核近似成 hard window(time0+duration).
- 但 FreeTimeGS 的 checkpoint 本身就是高斯时间核(更准确).
  - 因此新增 v2 语义,直接写入 `mu_t` 与 `sigma`,让 runtime 侧可直接算 `exp(-0.5 * ((t - mu)/sigma)^2)`.

### 新增 CLI
- `--splat4d-version 1|2`
  - v1: hard window(保持兼容,仍使用 `--temporal-threshold`).
  - v2: gaussian(time=mu_t,duration=sigma),会忽略 `--temporal-threshold`.

### 实际导出(你指定的大 ckpt, v2)
- 输出:
  - `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_gaussian.splat4d`(约 81.5MB)
- 命令:
```bash
source .venv/bin/activate
python tools/exportor/export_splat4d.py \
  --ckpt results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt \
  --output results/bar_release_full/out_0_61/exports/ckpt_29999_v2_gaussian.splat4d \
  --splat4d-version 2
```


## 2026-02-21 10:06:56 UTC 追加: 文档与参数对齐

### 更新点
- `README.md` 增加 `Export (Unity)` 小节,把 `.sog4d`/`.splat4d` 的导出方法与参数写成可复制命令.
- `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 的“参数建议”更新为以 `--help` 为准:
  - `.sog4d`: `--sh-bands`,`--shn-count`,`--shn-centroids-type`,`--shn-labels-encoding` 等.
  - `.splat4d`: `--splat4d-version 1|2`.
- 文档里明确标注尚未实现的项:
  - `.sog4d meta.version=2` 的 per-band palette(`sh1/sh2/sh3`).
  - 可配置的 delta segment length(当前固定 1 个 segment 覆盖全帧).
