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


## 2026-02-21 12:24:53 UTC 追加: DualGS(2409.08353)还能借鉴什么

> 论文: "Robust Dual Gaussian Splatting for Immersive Human-centric Volumetric Videos" (arXiv:2409.08353)

### 1) 压缩与流式播放(对我们 exporter 最直接有用)

- **Segment 化(固定帧长)作为“压缩与播放”的基本单位**
  - 论文明确把 LUT 宽度绑定到 segment 帧数,并给了一个常用值:
    - 原文短引述: "same LUT width to be the segment frame length (50)".
  - 启发:
    - 我们的 `delta-v1` 也天然是 segment 结构,实现 `--delta-segment-length` 后就能对齐这种“按段随机访问/流式加载”的需求.

- **SH 用 persistent codebook + delta indices**
  - 论文描述了一种很贴近我们 `.sog4d` 的方案:
    - 对每个 segment 做 kmeans 得到 codebook.
    - segment 内每帧不存完整 labels,只存变化.
  - 关键观察:
    - 原文短引述: "only 1% ... change" (在他们设置下,SH indices 跨帧变化很少).
  - 启发:
    - 即使 FreeTimeGS 的 SH 通常静态,也应把 delta 基础设施做完整(多 segment),这样未来如果 SH 真随时间微调,仍可无缝扩展.

- **“排序”提升图像/视频编码器压缩率**
  - 论文在 LUT 压缩里提到对 skin Gaussians 按平均 opacity/scale 排序,以提升 2D 连续性,从而让 codec 更好压缩.
  - 启发:
    - 我们的 labels/indices 是写进 WebP 的 2D RG 图,也可能从“稳定一致的重排”中获益.
    - 但这会影响 splat identity,需要同时输出 permutation 或确保 Unity 侧同样的重排,属于二期工程.

### 2) 表达与优化(不一定马上落地,但值得记住)

- **Dual representation(关节高斯+皮肤高斯)**
  - 他们把运动(骨架/关节)与表面细节(皮肤/纹理)拆开,用绑定权重把 skin attach 到 joint,提升可控性与鲁棒性.
  - 启发:
    - 对“人”类场景,这种结构化先验可能比纯 4D 速度模型更稳.
    - 我们当前是通用动态场景,短期不直接照搬,但可以作为“专项 human pipeline”的备选路线.

- **Coarse-to-fine + ARAP 正则**
  - 他们用更强的几何正则与分阶段优化提升跟踪稳定性.
  - 启发:
    - 若我们后续遇到“时间维抖动/漂移”,可能需要把 motion/shape 的约束显式写进 loss(而不是只靠渲染重建误差).

### 3) 对我们这次任务的直接映射

- 论文的 "four codebooks" 思路,非常贴近我们设计的 `.sog4d meta.version=2`:
  - 把 SH rest 按 band 拆成 `sh1/sh2/sh3` 三套 palette(DC 单独处理),每套维度更低,聚类更稳.
- 论文的 segment frame length(50) 思路,对应我们要补的:
  - `--delta-segment-length` 把 `[0,frameCount)` 切成多个 segment.
  - 每个 segment 写 base labels + delta 文件,为 streaming/随机访问铺路.


## 2026-02-21 12:39:32 UTC 追加: 已落地 per-band(v2)与 delta segment length

### 已实现的能力(落地到 `.sog4d` exporter)
- `tools/exportor/export_sog4d.py` 新增:
  - `--sh-version 1|2`
    - 1: 单一 shN palette,输出 `meta.json.version=1`(保持旧行为)
    - 2: per-band palettes(`sh1/sh2/sh3`),输出 `meta.json.version=2`
  - `--delta-segment-length`
    - 0: 单 segment 覆盖全帧(保持旧行为)
    - >0: 按段切分并写多段 `deltaSegments`

### v2(per-band)的 bundle 布局要点
- centroids:
  - `sh/sh1_centroids.bin`,`sh/sh2_centroids.bin`,`sh/sh3_centroids.bin`
- delta-v1:
  - 每个 segment 会写:
    - base labels: `frames/{startFrame}/sh1_labels.webp` 等(只在 segment 起始帧写)
    - delta: `sh/sh1_delta_{startFrame}.bin` 等(每个 segment 一个)

### 额外稳定性改良
- `scipy.cluster.vq.kmeans2` 在 K 较大时可能出现 empty cluster warning.
  - exporter 侧对 SH kmeans 增加了捕获 warning + 最多 3 次重试,避免输出质量不稳.


## 2026-02-21 13:40:00 UTC 追加: DualGS(2409.08353)在“压缩/播放”上还能借鉴的点(补充细节)

> 论文: "Robust Dual Gaussian Splatting for Immersive Human-centric Volumetric Videos" (arXiv:2409.08353v1, 2024-09-12)

### 1) Segment 作为压缩与播放的基本单位(更明确的动机)
- 他们在压缩部分开头就明确:
  - "divide the sequences into multiple segments"(按段切序列).
  - 并给出常用值: "50 in our setting".
- 这能解释为什么我们要把 `--delta-segment-length` 做成一等参数:
  - segment 是 random access/流式加载的最小单元.
  - `segmentLength=50` 在 30fps 下约 1.67s,对 VR 播放是合理的缓存粒度.

### 2) Opacity/Scale 的 LUT + codec 压缩(对 `.sog4d` 更直接,但 `.splat4d` 也可借鉴)
- 他们把 opacity 与 scaling 排成 2D LUT:
  - height=gaussian 数量.
  - width=segment frame length(=segment 长度).
- 然后做一个很工程化的小技巧:
  - "sort the LUT by the average value of each row" 来提升 2D 连续性.
  - 再用 WebP/JPEG 压成 8-bit 图.
- 对我们当前格式的启发:
  - `.sog4d` 本身就是“数据图(WebP)”,可以考虑引入稳定的全局 permutation(同时写入 permutation 以保持 splatId),来进一步提升 codec 压缩率.
  - 但这会触及 splat identity 的稳定性,属于二期工程.

### 3) SH 的 persistent codebook 与“稀疏变化事件”编码(比我们当前 delta-v1 更激进)
- 他们对 d-order(d=0,1,2,3)的 SH 係数做 kmeans 得到 4 套 codebook:
  - "we obtain four codebooks ... of length L(8192 in our setting)".
- 他们观察到 SH indices 的时间变化极稀疏:
  - "only one percent ... change between adjacent frames".
- 因此他们不是按“每帧 block”存 delta,而是存 change events:
  - 保存四元组 `(t, d, i, k)`(帧 t, 阶 d, gaussian i, 新 index k).
  - 并提到: order 不影响 decode,然后按前两维排序并做 length encoding.
- 对我们 `.splat4d/.sog4d` 的启发:
  - 我们当前 delta-v1 是 per-frame block,实现简单且可随机访问.
  - 若未来真的遇到“SH 少量变化但帧数很长”的场景,可以考虑增加一种更紧凑的 event-delta 编码(例如 delta-v2),以进一步降低 IO.

### 4) Motion 侧的压缩: R-VQ + RANS(短期不落地,但值得记下来)
- 他们对 joint motion 做 Residual-Vector Quantization(R-VQ).
- 对 temporal quantization(文中给了 11-bit 的设置)后,用 RANS 做无损压缩:
  - 这说明 motion 流也可以非常小,不会成为 per-frame 资产的主要开销.
- 对我们可能的映射:
  - `.splat4d` 当前 4D 字段是 float32,属于“先能用”.
  - 如果后续目标是移动端/超长序列,再考虑把 velocity/time/sigma 做定点量化 + entropy coding.


## 2026-02-21 14:33:33 UTC 追加: 从 DualGS 压缩章节再抽取可直接复用的工程细节

> 关键原文短引述(工程相关,只摘最关键的短句):
- "each segment consisting of f frames(50 in our setting)."
- "sort the LUT by the average value of each row."
- "temporal quantization(11-bit in our setting)"
- "four codebooks of length L(8192 in our setting)."
- "save this integer quadruples (t, d, i, k)."

### 1) Segment 是 IO/缓存/随机访问的最小颗粒度
- 他们的压缩与播放器都以 segment 为基本单位,并给了常用值 50 帧(30fps 下约 1.67s).
- 对我们最直接的映射:
  - `.sog4d`: 已用 `deltaSegments` 对齐 segment 的概念.
  - `.splat4d format v2`: section entry 带 `(startFrame,frameCount)` 并支持 `--delta-segment-length`.

### 2) LUT 排序是“白嫖压缩率”的小技巧(但会碰到 identity 问题)
- 他们把 opacity/scaling 排成 2D LUT(height=splatCount,width=segmentLen),再交给 WebP/JPEG.
- 额外做了一个很工程化的技巧: 按每行平均值排序来提升 2D 连续性.
- 对我们:
  - `.sog4d` 的 labels/indices WebP 也可能吃到类似收益.
  - 代价是会影响 splatId->pixel 的稳定映射,因此需要:
    - 输出 permutation,并在 importer 侧恢复;或
    - 仅在“完全不需要跨帧 identity”的支线格式里做.

### 3) SH delta 还可以更紧凑: event stream(可作为 delta-v2)
- 我们当前的 delta-v1 是 per-frame block,随机访问友好,实现也简单.
- DualGS 的做法更像“事件流”:
  - 只存首帧 indices.
  - 后续只存变化事件 quadruples `(t,d,i,k)`(并排序+length encoding).
- 如果未来出现“超长序列 + 少量 SH 变化”的典型场景,这条路线会明显更省 IO.

### 4) Motion 压缩路线: 先量化,再无损熵编码(移动端很关键)
- 他们对 joint motion 做 R-VQ,并明确提到:
  - "temporal quantization(11-bit in our setting)"
  - 使用 RANS 做 lossless 压缩
- 对 `.splat4d` 的映射建议:
  - 可以保持 format v2 的 section 机制不变,增加一个可选的 motion 压缩 section:
    - velocity/time/sigma 做定点量化(例如 10-16 bit),再做熵编码.
  - 这样可以在不破坏现有 float32 路径的情况下,逐步逼近“低端设备可播”的目标.


## 2026-02-22 02:33:37 UTC 追加: `.splat4d` exporter 默认值导致 Unity 走 legacy v1 导入

### 现象(来自 Unity 侧离线统计与导入行为)
- `ckpt_29999_v2_gaussian.splat4d` 文件名带 v2,但实际上无 `SPL4DV02` header.
- Unity importer 会把它当作 v1(legacy无 header)导入,并按 window(time0+duration)裁剪.
- 对 time/duration 分布更像 gaussian(mu+sigma)的场景,按 window 裁剪会出现"薄层/稀疏"的可见性伪影.

### 根因(本仓库 exporter 的两个正交概念被默认值混淆)
- `tools/exportor/export_splat4d.py` 里有两套概念:
  - `--splat4d-version`: time/duration 的语义(1=window,2=gaussian).
  - `--splat4d-format-version`: 文件格式(1=legacy无 header,2=header+sections).
- 旧默认值是 `--splat4d-format-version=1`.
  因此只写 `--splat4d-version 2` 会输出"gaussian 语义 + legacy 无 header"的文件.
  Unity 侧如果仅靠 header 判断 v1/v2,就会走错导入路径.

### 修复(已落地)
- 给 exporter 增加更安全的默认值:
  - `--splat4d-format-version` 新增 `0=auto` 并作为默认.
  - auto 规则:
    - `--splat4d-version=2` 时,默认输出 format v2(header+sections),确保带 `SPL4DV02` header 与 timeModel.
    - `--sh-bands>0` 时,也会自动选择 format v2(legacy 承载不了 SH rest/deltaSegments).
    - 其它情况默认输出 format v1(legacy).
- 当用户显式指定 `--splat4d-version 2` + `--splat4d-format-version 1` 时,在 stderr 打印醒目的 warning,避免再踩坑.
- 同步更新文档:
  - `README.md` 补充 auto 默认的备注.
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 把 timeModel vs formatVersion 拆开说明,并强调 gaussian 建议输出带 header.

### 冒烟验证(本机最小 ckpt)
```bash
source .venv/bin/activate
python3 tools/exportor/export_splat4d.py \
  --ckpt /tmp/splat4d_smoke_ckpt.pt \
  --output /tmp/splat4d_smoke_gaussian_auto.splat4d \
  --splat4d-version 2
python3 -c "print(open('/tmp/splat4d_smoke_gaussian_auto.splat4d','rb').read(8))"
```
- 预期输出前 8 bytes 为 `b'SPL4DV02'`,表示 header 存在,Unity importer 会稳定走 v2.

## 2026-02-22 03:53:03 UTC 追加: 高质量 `.splat4d v2 + sh3 + seg50` 实际导出产物

### 产物
- 输入 ckpt: `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
- 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f16.splat4d`
- 文件头 magic: `SPL4DV02`(说明带 header,Unity importer 应走 v2)

### 参数(核心)
- `--splat4d-version 2`(timeModel=2,gaussian)
- `--sh-bands 3`(per-band SH rest: sh1/sh2/sh3)
- `--frame-count 61 --delta-segment-length 50`(deltaSegments,共 2 段)
- `--shn-count 512 --shn-centroids-type f16 --shn-codebook-sample 200000 --shn-kmeans-iters 10`

### 用途
- 这份文件用于验证 Unity 侧的 v2 importer 是否已完整支持:
  - header v2 + timeModel=2(gaussian)
  - per-band SH centroids + base labels + delta-v1 blocks

## 2026-02-22 07:10:28 UTC 追加: `gsplat-unity` 是否需要同步更新(per-band SH rest + deltaSegments)

### 结论
- `/workspace/gsplat-unity` 已经实现了 `.splat4d format v2` 的:
  - per-band SH rest codebooks(`SHCT`)+labels(`SHLB`)解码到 `GsplatAsset.SHs`
  - delta-v1 的 segment 覆盖性校验与 delta header 校验(`SHDL`)
- 因此,以当前 FreeTimeGsVanilla exporter 的输出(我们现在 delta-v1 默认 `updateCount=0`)来看,`gsplat-unity` 不需要额外改动就能正确导入并渲染(至少在 importer 逻辑层面已经对齐).

### 证据(关键文件与关键引述)
- importer 文件: `/workspace/gsplat-unity/Editor/GsplatSplat4DImporter.cs`
  - v2 header/sections 常量:
    - `static readonly byte[] k_magicV2 = ... "SPL4DV02"`(用于区分 v1/v2)
    - section kinds: `RECS/META/SHCT/SHLB/SHDL`
  - 直接引述(原文,用于说明范围):
    - "v2: header + section table,用于承载 SH rest 与更准确的时间核语义." (文件内 v2 说明注释)
    - "5) SH rest(per-band)解码(可选)" (v2 import 流程第 5 步)
  - 解码入口:
    - 当 `header.shBands > 0` 时调用 `TryDecodeShBandsFromV2(...)` 去读取 `SHCT/SHLB/SHDL`.

### 兼容性备注(重要,避免未来踩坑)
- 当前 `GsplatSplat4DImporter.cs` 对 `labelsEncoding=delta-v1` 的处理策略是:
  - 会校验 segments 必须连续覆盖 `[0,frameCount)` 并校验 delta header.
  - 但实际解码时只取 `startFrame=0` 的 base labels(`SHLB`)来重建 SH rest,不会应用后续帧的 delta updates.
- 这和我们当前 exporter 的默认实现是匹配的(目前 `.splat4d` 的 delta-v1 body 主要是 `updateCount=0` 的静态占位).
- 如果未来我们在 exporter 里真正生成非 0 的 update(让 SH 随时间变化),Unity 侧需要再补:
  - delta updates 的应用逻辑(按帧累积更新 labels,或在 GPU 上应用).

---

## 2026-02-22 15:39:40 +0800: `.sog4d` exporter(meta.json) 与 gsplat-unity 读者实现对齐修复

### 现象
- gsplat-unity 离线校验失败:
  - `[sog4d][error] meta.json.format 非法: None`

### 根因
- `tools/exportor/export_sog4d.py` 写出的 meta.json:
  - 缺少顶层 `format="sog4d"`.
  - 把 float3 数组写成 `[[x,y,z], ...]`,而 Unity `JsonUtility` 解析 `Vector3` 需要 `{x,y,z}`.

### 修复(已落地)
- `tools/exportor/export_sog4d.py`:
  - 补齐 `meta.format="sog4d"`.
  - `streams.position.rangeMin/rangeMax` 与 `streams.scale.codebook` 输出改为 `[{"x":..,"y":..,"z":..}, ...]`.
- `tools/exportor/spec.md`:
  - 补齐上述 MUST 约束,避免未来再复发.

---

## 2026-02-22 10:25:00 UTC: Unity 中点云偏移/歪倒(训练 normalize 空间 vs COLMAP 原始空间)

### 现象
- Unity 中导入 `.splat4d` 后:
  - 高斯点云整体偏移.
  - 三轴旋转也不符合预期,看起来像"歪倒".
  - 与原始 COLMAP 相机视角/重心轴线不一致.

### 根因(坐标空间不一致)
- 训练侧使用 `FreeTimeParser(normalize=True)`:
  - 会对 COLMAP 原始空间应用归一化 transform `T = T2@T1`(相机中心缩放 + PCA 对齐).
  - ckpt 中的 `means/velocities/scales/quats` 都处于 "train normalized 空间".
- Unity 的 `.splat4d` importer 不做坐标轴/归一化的隐式翻转与反变换.
  - 因此当 Unity 侧拿 "原始 COLMAP 相机位姿/坐标" 做参考时,会出现整体错位.

### 修复(已落地到 exporter)
- `tools/exportor/export_splat4d.py` 新增:
  - `--output-space train|colmap`(默认 train).
  - `--colmap-dir <sparse/0>`:
    - 读取 COLMAP 的 `cameras/images/points3D`.
    - 复现训练侧的 `colmap->train` 归一化 transform.
    - 对导出 record 的 `(position, velocity, scale, rotation)` 统一应用 `T^{-1}`,把训练坐标反变换回 COLMAP 原始空间.
  - v2 文件额外写入 `XFRM` section(64B,16xf32):
    - 保存 `colmap->train` transform,用于离线 debug/一致性校验(不影响现有 importer).

### 端到端验证(本机导出产物)
- 输入 ckpt:
  - `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
- 输出(新增 colmap 空间版本):
  - `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f16_colmap.splat4d`
- header/sections:
  - magic=`SPL4DV02`
  - section table 包含 `XFRM`


## 2026-02-22 13:31:47 UTC

### OpenSpec: `.splat4d` delta-v1 真实 updates + Unity 运行时应用

- 已创建 OpenSpec change:
  - `openspec/changes/splat4d-delta-v1-sh-updates/`
- 已生成 artifacts(用于后续按任务逐条落地):
  - `proposal.md`: 明确动机,scope,非目标.
  - `design.md`: 明确跨仓库实现路径与关键取舍(GPU scatter/per-segment base labels/时间映射等).
  - `specs/**/spec.md`: 固化 exporter 与 Unity 的规范化要求(可测试场景).
  - `tasks.md`: 可追踪的实现清单.
