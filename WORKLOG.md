# WORKLOG

## 2026-02-20 13:36:01 UTC
- 初始化文件上下文工作模式: 新建 `task_plan.md`, `notes.md`, `WORKLOG.md`, `LATER_PLANS.md`, `ERRORFIX.md`.

## 2026-02-20 14:37:35 UTC
- 定位 `uv sync --locked` 构建失败根因: `fused-ssim`/`gsplat` 在构建期导入 `torch`,但未声明构建期依赖,触发 PEP517 build isolation 缺包.
- 将 uv 升级到 `0.10.4`,用于获得更稳定的下载重试与安装体验.
- 安装系统编译工具链 `build-essential`,用于编译 `lapjv` 等 C/C++ 扩展依赖.
- 调整 `pyproject.toml`:
  - 增加 `tool.uv.required-version = ">=0.10.4"`.
  - 保留 `no-build-isolation-package = ["fused-ssim", "gsplat"]` 并补齐 `dependency-metadata` 以降低解析阶段的构建需求.
  - 移除未使用的 `torch_scatter` 依赖,降低安装成本.
- 更新 `uv.lock`,并验证 `uv sync --locked` 成功完成.

## 2026-02-20 15:15:44 UTC
- 新增 `AGENTS.md`: 为本仓库补齐贡献者指南,覆盖目录结构,uv 环境命令,smoke-check,以及提交/PR 约定.

## 2026-02-20 16:55:24 UTC
- 对比 `FreeTimeGsVanilla` 与 `BCEF/FreetimeGS_NO` 的数据格式与相机位姿假设,并确认 `FreetimeGS_NO` 默认预处理脚本仅首帧跑 COLMAP,其它帧复用首帧 `sparse/`(相机位姿静态假设),且未提供逐时间帧三角化点云的完整工具链.

## 2026-02-21 01:59:48 UTC
- 调研 `zju3dv/EasyVolcap` 与 `oppo-us-research/SpacetimeGaussians` 的数据预处理脚本,定位并阅读关键入口文件.
- 结论沉淀到 `notes.md`:
  - EasyVolcap: `scripts/preprocess/extract_videos.py`(ffmpeg 抽帧),`scripts/colmap/run_colmap.py`(一键跑 COLMAP),`scripts/preprocess/extract_subseq.py`(裁子序列).
  - SpacetimeGaussians: `script/pre_n3d.py`(OpenCV 抽帧),`script/pre_no_prior.py` + `thirdparty/gaussian_splatting/helper3dg.py`(参考帧求位姿,逐帧 point_triangulator 三角化点云),以及 `thirdparty/colmap/pre_colmap.py`(写 COLMAP DB/先验位姿).

## 2026-02-21 03:41:48 UTC
- 落盘本次任务计划: 追加 `# 任务计划: mp4->4DGS(RoMA 全帧初始化)` 到 `task_plan.md`,并创建规格文档 `specs/mp4_to_4dgs_pipeline.md`.
- 补齐 Mermaid 文档工具链:
  - 新增 `tools/mermaid-validator`(校验 Markdown 里的 Mermaid code block).
  - 新增 `tools/beautiful-mermaid-rs`(把 Mermaid 渲染为终端 Unicode 文字图).
  - 对 `specs/mp4_to_4dgs_pipeline.md` 进行了语法校验并渲染验证.
- 安装系统依赖: `apt-get install -y colmap`,并通过 `colmap --help` 验证可用.
- Python 依赖更新: `pyproject.toml` 增加 `romatch`,同步更新 `uv.lock`,并执行 `uv sync --locked` 验证 `import romatch`.
- 新增预处理入口 `src/preprocess_mp4_freetimegs.py`:
  - mp4 抽帧到 `data_dir/images/<cam>/%06d.jpg`.
  - 参考帧跑一次 COLMAP 得到 `data_dir/sparse/0`.
  - 用 RoMA + RANSAC + 三角化,逐帧生成 `triangulation_dir/points3d_frame%06d.npy` 与 `colors_frame%06d.npy`.
- 改造 `src/combine_frames_fast_keyframes.py`:
  - 新增 `--mode {keyframes,all_frames}`.
  - 统一 `--frame-end` 语义为 end exclusive,并更新 time 归一化与 NPZ metadata.
- 训练侧补齐论文式 config: 在 `src/simple_trainer_freetime_4d_pure_relocation.py` 增加 `paper_stratified_small`.
- 新增一键脚本 `run_mp4_pipeline.sh`,并同步更新 `run_pipeline.sh` 的 `--frame-end` 传参.
- 更新 `README.md`: 修正 Python 版本徽章为 3.12+,补充 mp4 pipeline 用法,并更新 combine 示例的 frame_end 语义.

## 2026-02-21 03:55:16 UTC
- 对齐 time 归一化口径: 在 `src/simple_trainer_freetime_4d_pure_relocation.py` 的 `load_init_npz/load_init_npz_stratified/load_init_npz_keyframe` 中,把 velocity scaling 与 duration rescale 统一为使用 `max(total_frames - 1, 1)`,与 combine 输出的 time 定义保持一致.

## 2026-02-21 04:13:44 UTC
- 完成一次 git 提交: `d537095`(message: "Add mp4->4DGS RoMA all-frames pipeline").
- 由于环境缺少 git 身份配置,在仓库本地设置了(非 global):
  - `user.name = codex-cli`
  - `user.email = codex-cli@localhost`
- 说明: 如需使用你自己的作者信息,可用 `git commit --amend --reset-author` 或手动 `git config user.name/user.email` 后再 amend.

## 2026-02-21 07:13:07 UTC
- 用测试视频 `/cloud/cloud-s3fs/SelfCap/bar-release/videos` 做端到端验证,跑通 mp4->抽帧->参考帧 COLMAP->RoMA 逐帧三角化->combine(all_frames)->train 全流程.
- 核心产物(帧段 [0,61), config=paper_stratified_small):
  - init_npz: `results/bar-release_work/init_0_61.npz`(约 1,333,338 points).
  - ckpt: `results/bar-release_result_run2/ckpts/ckpt_29999.pt`.
  - traj video: `results/bar-release_result_run2/videos/traj_4d_step29999.mp4`.
- 为避免高分辨率图片下 COLMAP `feature_extractor` 被 OOM killer SIGKILL:
  - 在 `src/preprocess_mp4_freetimegs.py` 增加并默认启用 `--colmap-sift-num-threads=1`,`--colmap-sift-max-image-size=2000`,`--colmap-match-num-threads=1`.
- 按用户选择(B),将四文件上下文记录(`task_plan.md`,`notes.md`,`WORKLOG.md`,`LATER_PLANS.md`,`ERRORFIX.md`)与收尾修复(`README.md`,`specs/mp4_to_4dgs_pipeline.md`,`src/preprocess_mp4_freetimegs.py`)一并纳入 git 提交.

## 2026-02-21 07:56:05 UTC
- 进一步增强“高分辨率更稳”的默认行为:
  - `src/preprocess_mp4_freetimegs.py` 新增 `--colmap-sift-max-num-features` 与 `--colmap-oom-retries`.
  - 当 COLMAP `feature_extractor` 因 OOM 被 SIGKILL 时,会自动降级 `max_image_size/max_num_features` 并重试,降低手工排障成本.
- 训练侧默认更稳:
  - `paper_stratified_small` 默认 `data_factor=4`,`max_samples=2_000_000`.
  - `run_mp4_pipeline.sh` 支持用环境变量覆盖 `DATA_FACTOR/MAX_SAMPLES/COLMAP_*`,不用改脚本就能适配更大分辨率数据.
- 将 git 提交的作者信息统一改为你的 author(`raiscui <vdcoolzi@gmail.com>`),方便后续直接 push 到远端.

## 2026-02-21 09:19:20 UTC
- 新增 `.sog4d` 导出工具: `tools/exportor/export_sog4d.py`.
  - 按 `tools/exportor/spec.md` 的 streams/layout/timeMapping 规则写 `meta.json` 与 per-frame WebP 数据图.
  - 当前最小可用实现为 `bands=0`(只导出 sh0+opacity),先保证 Unity 能导入能播放.
- 已完成对你指定 checkpoint 的导出:
  - 输入: `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
  - 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full.sog4d`(约 1.1G)

## 2026-02-21 09:45:50 UTC
- 扩展 `.sog4d` exporter 支持导出 SH rest(bands>0):
  - `tools/exportor/export_sog4d.py` 新增 `--sh-bands 1..3`,实现 v1: `shN_centroids.bin` + labels(WebP) + delta-v1.
  - 默认用 `delta-v1`(FreeTimeGS 的 SH 通常静态,delta 文件基本全是 `updateCount=0`).
- 已完成对同一 checkpoint 的“含 SH rest”全量导出:
  - 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full_sh3_v1delta_k512.sog4d`

复现命令:
```bash
source .venv/bin/activate
python tools/exportor/export_sog4d.py \
  --ckpt-path results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt \
  --output-path results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full_sh3_v1delta_k512.sog4d \
  --frame-count 61 \
  --layout-width 2048 \
  --sh-bands 3 \
  --shn-count 512 \
  --shn-labels-encoding delta-v1 \
  --webp-method 0 \
  --zip-compression stored \
  --overwrite
```
