# 任务计划: uv sync 构建失败修复

## 目标
在本仓库使用 `uv sync --locked` 能稳定完成依赖安装,并生成可用的 `.venv/`.

## 阶段
- [x] 阶段1: 计划和设置
- [x] 阶段2: 研究/收集信息
- [x] 阶段3: 执行/构建
- [x] 阶段4: 审查和交付

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(更长期稳定)
- 思路: 为所有会在构建期 `import torch` 的第三方包,在仓库层面补齐正确的 PEP517 构建依赖声明.
- 代价: 需要 fork 这些 git 依赖(如 `fused-ssim`, `gsplat`),或引入 patch 机制,维护成本更高.
- 收益: 不依赖 `--no-build-isolation`,对任意构建工具更兼容.

### 方向B: 先能用,后面再优雅(推荐,改动小)
- 思路: 使用 uv 的 `no-build-isolation-package` 配置,让这些包在安装时关闭构建隔离.
- 代价: 依赖 uv 的两阶段安装逻辑,但本仓库已经使用 uv,风险可控.
- 收益: 改动小,可快速 unblock,并且可逐个包精确控制.

## 关键问题
1. `fused-ssim` 在 build backend 阶段导入 `torch`,但没有声明构建期依赖,导致 uv 的 build isolation 环境缺少 `torch`.
2. 除 `fused-ssim` 外,`gsplat` 也会在构建期导入 `torch`,因此需要同样的处理策略.
3. 依赖里存在不必要的编译型包会显著拉长安装时间,应优先删除未使用的依赖项(例如本仓库原先的 `torch_scatter`).

## 做出的决定
- [2026-02-20 13:36:01 UTC] 选择方向B: 先用 uv 配置 `no-build-isolation-package` 解除阻塞,后续若仍有同类包再增补名单.
- [2026-02-20 14:37:35 UTC] 升级 uv 到 `0.10.4`,用于获得更稳定的下载重试机制,并避免旧版本在构建顺序上的坑.
- [2026-02-20 14:37:35 UTC] 安装系统编译工具链 `build-essential`,用于编译 `lapjv` 等 C/C++ 扩展依赖.
- [2026-02-20 14:37:35 UTC] 移除未在代码中使用的 `torch_scatter` 依赖,降低安装成本,并同步更新 `uv.lock`.
- [2026-02-20 14:37:35 UTC] 在 `pyproject.toml` 增加 `tool.uv.required-version = \">=0.10.4\"`,避免用户用旧 uv 触发同类错误.

## 遇到错误
- [2026-02-20 13:36:01 UTC] `uv sync --locked` 构建 `fused-ssim` 失败,报 `ModuleNotFoundError: No module named 'torch'`.
- [2026-02-20 14:37:35 UTC] `lapjv` 构建失败,报 `error: command 'c++' failed: No such file or directory`(缺少系统编译器).
- [2026-02-20 14:37:35 UTC] 部分包下载出现 timeout,需要 uv 的重试能力或调大 `UV_HTTP_TIMEOUT`.

## 状态
**目前在阶段4**: 已完成修复与验证,`uv sync --locked` 可成功安装依赖并生成可用的 `.venv/`.


# 任务计划: 生成 AGENTS.md(Contributor Guide)

## 目标
为本仓库新增一份简洁的 `AGENTS.md`,用作贡献者指南(结构,常用命令,风格,测试,PR 约定).

## 阶段
- [x] 阶段1: 计划和设置
- [x] 阶段2: 研究/收集信息
- [x] 阶段3: 执行/构建
- [x] 阶段4: 审查和交付

## 关键问题
1. 依赖管理/环境创建的权威方式是什么(README 的 pip vs 实际使用的 uv)?
2. 本仓库是否存在测试框架与可复用的 smoke-check?
3. Git 历史里的提交信息大致风格是什么,能否提炼成可执行的约定?

## 做出的决定
- [2026-02-20 15:13:50 UTC] 以仓库现状为准: 用 `uv sync --locked` + `.venv/` 作为默认开发方式,并在文档中同时保留最小可运行示例命令.

## 状态
**目前在阶段4**: 已生成 `AGENTS.md`(约 265 words),覆盖结构,uv 命令,smoke-check,以及提交/PR 约定.


# 任务计划: 对比 FreeTimeGsVanilla vs FreetimeGS_NO(数据格式/相机位姿/三角化链路)

## 目标
回答两仓库在训练方案上的关键差异,并明确 `FreetimeGS_NO` 是否要求相机位姿静态,以及是否内置“逐时间帧三角化点云”的工具链.

## 阶段
- [x] 阶段1: 计划和设置
- [x] 阶段2: 研究/收集信息
- [x] 阶段3: 执行/构建
- [x] 阶段4: 审查和交付

## 关键问题
1. 两仓库的输入数据格式是否一致(按相机组织 vs 按时间帧组织)?
2. `FreetimeGS_NO` 的代码是否支持“每帧不同相机位姿”,还是仅能复用首帧的静态位姿?
3. `FreetimeGS_NO` 是否提供 per-time 的三角化点云生成脚本,还是依赖 COLMAP 首帧点云/外部工具?

## 做出的决定
- [2026-02-20 16:55:24 UTC] 以代码为准做判断: 直接阅读 `FreetimeGS_NO` 的 `README.md`,`scene/dataset_readers.py`,`data_preprocess.py`,`utils/convert.py` 来确认数据组织方式与相机位姿假设.

## 状态
**目前在阶段4**: 已完成对比结论整理,准备输出给用户并给出选型建议.


# 任务计划: 调研 EasyVolcap 与 SpacetimeGaussians 的视频/图片预处理工具链

## 目标
从 `zju3dv/EasyVolcap` 与 `oppo-us-research/SpacetimeGaussians` 中,找出可借鉴的“视频抽帧/图片整理/相机位姿重建(COLMAP)/格式转换”等预处理代码,并给出可直接复用的文件路径与接入建议.

## 阶段
- [x] 阶段1: 计划和设置
- [x] 阶段2: 研究/收集信息
- [ ] 阶段3: 执行/构建(可选: 抽取成可复用脚本)
- [x] 阶段4: 审查和交付

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(更体系化)
- 思路: 按“数据源 -> 抽帧 -> COLMAP -> 格式转换 -> 本仓库消费”的链路,梳理两仓库成熟脚本,抽象出一套更稳的最小预处理 pipeline,并补齐文档与可复用命令.
- 代价: 需要更深入理解对方仓库的数据组织与依赖(可能包含额外的第三方工具/子模块).
- 收益: 形成可持续复用的标准化流程,后续接新数据成本更低.

### 方向B: 先能用,后面再优雅(推荐,聚焦可用片段)
- 思路: 只挑“最有用且最独立”的脚本(例如 ffmpeg 抽帧,跑 COLMAP,把 COLMAP 输出转换为某种训练格式),给出具体文件路径与调用方式,让你能快速拼起来用.
- 代价: 可能缺少完全一键化,需要你在本仓库侧做少量 glue.
- 收益: 成本最低,最快把数据跑起来.

## 关键问题
1. 两仓库是否提供从视频(mp4)抽帧到图片序列的脚本,是否依赖 ffmpeg?
2. 两仓库是否提供“一键跑 COLMAP”(特征/匹配/建图/稀疏模型)的脚本,以及输出目录约定?
3. 两仓库是否提供“COLMAP -> 自己格式”的转换脚本(例如生成 `images/` 结构,相机参数 yaml/json,或 nerf/3dgs 常见格式)?
4. 哪些脚本最容易被本仓库(FreeTimeGsVanilla)直接借鉴: 多相机目录组织,时间维度组织,以及点云/位姿导出?

## 做出的决定
- [2026-02-21 01:59:48 UTC] 以“可复用的预处理脚本”为优先级,先做代码级检索与文件路径定位,再决定是否需要抽取成我方仓库脚本.

## 状态
**目前在阶段4**: 已完成交付,结论与关键文件路径已记录到 `notes.md`.


# 任务计划: mp4->4DGS(RoMA 全帧初始化)

## 目标
一键脚本: 输入"每路相机一个 mp4",自动跑通 mp4->抽帧->参考帧 COLMAP->RoMA 逐帧三角化->全帧 combine->训练,最终在 `results/.../ckpts/ckpt_29999.pt` 产出可用的 4D Gaussians(30000 steps 的最终 step=29999),并生成渲染视频用于验证.

## 阶段
- [x] 阶段1: 计划和设置(先落盘 plan/specs)
- [x] 阶段2: 依赖与前置检查(colmap/romatch/mermaid 工具链)
- [x] 阶段3: 执行/构建(抽帧+参考 COLMAP+RoMA 三角化+combine all_frames+train)
- [x] 阶段4: 审查和交付(编译/导入检查,回写四文件)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(论文同款,已选)
- 用 `romatch`(RoMA) 进行 anchor<->其它相机的 dense matching.
- 用 OpenCV 做 `USAC_MAGSAC` RANSAC 过滤外点,并在已知静态相机位姿下逐帧三角化点云.
- 每帧输出 `points3d_frame%06d.npy` + `colors_frame%06d.npy`.
- `combine_frames_fast_keyframes.py` 增加 `--mode all_frames`,把每帧点都写入 NPZ 以支持 `use_stratified_sampling=True` 的论文式初始化.

### 方向B: 先能用,后面再优雅(兜底方案)
- 参考帧跑一次 COLMAP `mapper` 求静态相机位姿(与方向A一致).
- 其余帧用 `colmap point_triangulator` 固定相机位姿逐帧三角化点云.
- 参考实现来源: SpacetimeGaussians 的 `script/pre_no_prior.py` + `thirdparty/gaussian_splatting/helper3dg.py`(已在 `notes.md` 记录路径).

## 关键约束与已选决定
1. 原始输入是"每路相机一个 mp4"(以文件名 stem 作为相机名).
2. 相机位姿静态(只需要参考帧跑一次 COLMAP).
3. 先跑通帧段: `start_frame=0,end_frame=61`(end exclusive).
4. 产物策略: 默认只保留必要产物(可用 `--keep-intermediate` 开关保留中间文件便于排障).
5. 初始化点云: 每帧点云,并改造 combine 支持全帧写入.

## 状态
**目前在阶段4**: 已完成交付.代码与脚本已落地,并通过 smoke-check(compileall/import).工作记录已回写到 `WORKLOG.md`/`notes.md`/`LATER_PLANS.md`.

## 状态更新
- [2026-02-21 04:13:44 UTC] 已完成 git commit: `d537095`.
  - 为了让提交在本机可执行,已在本仓库设置本地 git 身份(非 global): `codex-cli <codex-cli@localhost>`.

# 任务计划: 用 bar-release(mp4) 做端到端验证

## 目标
使用测试视频目录 `/cloud/cloud-s3fs/SelfCap/bar-release` 作为输入(每路相机一个 mp4),跑通本仓库新增的一键 pipeline:
`mp4 -> images -> ref COLMAP -> RoMA triangulation -> init.npz(all_frames) -> train -> ckpt_29999.pt + videos`.

## 阶段
- [x] 阶段1: 检查输入数据(mp4 数量/命名/帧数/分辨率)
- [x] 阶段2: 预处理(mp4 抽帧 + 参考帧 COLMAP + RoMA 逐帧三角化)
- [x] 阶段3: combine(all_frames) 生成 init.npz
- [x] 阶段4: 训练与产物验证(ckpt/videos)
- [x] 阶段5: 回写记录与必要修复(WORKLOG/ERRORFIX/LATER_PLANS)

## 方案方向(至少二选一)

### 方向A: 不惜代价,完整验证(你要的最终状态)
- 跑到 `results/.../ckpts/ckpt_29999.pt` 确认产物齐全.
- 代价: 时间更长.

### 方向B: 先能用,快速冒烟(推荐先做)
- 先跑完 阶段1-3,并启动训练跑少量 step(确认 loss/backprop/IO 没问题).
- 代价: 不能立刻得到最终 ckpt.
- 收益: 快速定位 RoMA API/三角化/数据格式类问题.

## 关键约束与已选决定
1. 依旧假设"相机位姿静态",只在参考帧跑一次 COLMAP.
2. 先用帧段: `start_frame=0,end_frame=61`(end exclusive),参考帧 `reference_frame=0`.
3. 默认训练 config: `paper_stratified_small`.
4. 产物目录策略: 产物写入 `results/`(gitignored),避免污染仓库.

## 状态
**目前在阶段5**: 已完成交付.端到端训练已跑通并产出 ckpt+视频.四文件记录已回写.本次收尾修复与四文件已按用户选择(B)纳入 git 提交.

## 状态更新
- [2026-02-21 05:23:00 UTC] 已检查输入视频:
  - mp4_dir: `/cloud/cloud-s3fs/SelfCap/bar-release/videos`
  - 相机数: 18 路(文件名 `02.mp4`..`19.mp4`)
  - 分辨率: 2110x3760
  - 帧率: 60 fps
  - 帧数: 3540 帧(59s)
- [2026-02-21 06:13:11 UTC] 已完成 full 预处理 + all_frames combine:
  - 帧段: [0, 61)
  - data_dir: `results/bar-release_work/data`
  - triangulation_dir: `results/bar-release_work/triangulation`
  - init_npz: `results/bar-release_work/init_0_61.npz`(1,333,338 points)
  - 训练启动参数: `paper_stratified_small` + `--data-factor 4` + `--max-samples 200000` + `--max-steps 30000`
- [2026-02-21 06:30:34 UTC] 重新启动训练(避免上一次训练会话中断导致未落盘 ckpt 的风险):
  - result_dir: `results/bar-release_result_run2`
- [2026-02-21 06:51:48 UTC] 训练完成并产出结果(帧段 [0,61), config=paper_stratified_small, data_factor=4, max_samples=200000):
  - ckpt: `results/bar-release_result_run2/ckpts/ckpt_29999.pt`(注意: 训练 step 是 0-based,30000 steps 的最终 step=29999)
  - traj video: `results/bar-release_result_run2/videos/traj_4d_step29999.mp4`
  - val renders: `results/bar-release_result_run2/renders/val_step29999_*.png`
  - stats: `results/bar-release_result_run2/stats/*_step29999.json`
- [2026-02-21 07:13:07 UTC] 按用户选择(B)执行收尾:
  - 将本次修复(含 COLMAP OOM/SIGKILL 规避参数)与四文件上下文记录一并纳入 git 提交.


# 任务计划: 改提交作者信息 + 高分辨率更稳(自动降级/训练下采样)

## 目标
1. 把我最近两次提交(`d537095`,`55fffaf`)的 author 改成你的名字与邮箱.
2. 让更高分辨率(例如 4K/8K)数据跑 pipeline 更稳:
   - COLMAP 被 OOM killer(SIGKILL)时自动降级重试,而不是直接失败.
   - 训练默认启用更安全的下采样(data_factor),避免显存/时间爆炸.

## 阶段
- [x] 阶段1: 计划与约束确认
- [x] 阶段2: 实现稳态改进(COLMAP 自动降级重试 + 训练默认下采样)
- [x] 阶段3: 本地验证(compileall/import/mermaid-validator)
- [x] 阶段4: 改写 git 历史(重写 author)并提交

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(推荐,更稳)
- COLMAP feature_extractor SIGKILL 时,自动重试并逐步降低:
  - `SiftExtraction.max_image_size`
  - `SiftExtraction.max_num_features`
- 同时把 `paper_stratified_small` 的默认 `data_factor` 调到更稳的值(例如 4).
- 优点: 任何高分辨率数据都更“开箱即用”.
- 代价: 代码会多一点参数与分支,但逻辑是可控的.

### 方向B: 先能用,后面再优雅(最小改动)
- 只把 `paper_stratified_small` 默认 `data_factor=4`,并在 README 里提示:
  - 若 COLMAP SIGKILL,手动调低 `--colmap-sift-max-image-size` 和线程数.
- 优点: 改动更少.
- 缺点: 仍需要人肉排障,不够“稳”.

## 做出的决定
- [2026-02-21 07:16:21 UTC] 选择方向A: 自动降级重试 + 默认训练下采样,优先确保高分辨率数据稳定跑通.

## 关键问题
1. 你的 git 作者信息是:
   - user.name = raiscui
   - user.email = vdcoolzi@gmail.com
2. 你希望 author 改写后保留原提交时间,还是接受“重写提交时间为当前时间”?
   - 说明: `git commit --amend --reset-author` 通常会把 author date 更新为当前时间.

## 状态
**目前在阶段4**: 已完成交付.稳态改进已提交,且已把近期提交的 author 统一改为 `raiscui <vdcoolzi@gmail.com>`(并尽量保持原提交时间不变).

## 状态更新
- [2026-02-21 07:44:38 UTC] 已完成稳态改进与本地验证:
  - `src/preprocess_mp4_freetimegs.py`: COLMAP feature_extractor SIGKILL(OOM) 自动降级重试.
  - `src/simple_trainer_freetime_4d_pure_relocation.py`: `paper_stratified_small` 默认 `data_factor=4`,`max_samples=2_000_000`.
  - `run_mp4_pipeline.sh`: 增加可选环境变量覆盖(DATA_FACTOR/MAX_SAMPLES/COLMAP_*).
  - 验证: `python -m compileall src datasets` + `import torch, gsplat, romatch` + `./tools/mermaid-validator ...` 均通过.
- [2026-02-21 07:57:40 UTC] 已完成提交与 author 改写:
  - 本地 git 身份已更新: `raiscui <vdcoolzi@gmail.com>`.
  - 已提交稳态增强: `2ec4ed5`.
  - 已重写近期提交的 author:
    - `4ca57de`(原 `d537095`)
    - `45a229f`(原 `55fffaf`)


# 任务计划: FreeTimeGS checkpoint(.pt) -> `.sog4d` exporter + 导出 bar_release_full ckpt

## 目标
1. 在本仓库实现一个可复用的导出工具: 把 FreeTimeGS 的 checkpoint(`ckpt_*.pt`)导出为 Unity 可导入的 `.sog4d`(ZIP bundle).
2. 用该工具把:
   - `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
   导出为 `.sog4d`,产物落到 `results/` 下(不进入 git).

## 阶段
- [x] 阶段1: 计划与规格对齐(阅读 `tools/exportor/*.md`,明确 meta.json/streams/layout/timeMapping)
- [x] 阶段2: 实现 exporter(先做 bands=0 的最小可用,再扩展)
- [x] 阶段3: 本地验证(小规模导出 5 帧 + 自检 meta/layout)
- [x] 阶段4: 全量导出(帧段 61 帧)并记录产物路径
- [x] 阶段5: 回写记录(WORKLOG/notes/ERRORFIX/LATER_PLANS)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(最终形态,更快更干净)
- 直接写 `.sog4d`:
  - checkpoint -> 逐帧采样(pos/opacity) -> 量化 -> 写 WebP 数据图 -> 写 meta.json -> 打包 ZIP.
- 优点: 无中间 PLY,IO 更少,更容易 chunk 控制内存峰值.
- 缺点: 需要在 exporter 里实现更多格式细节(但 `tools/exportor/spec.md` 已经把规则写清楚).

### 方向B: 先能用,后面再优雅(依赖 pack 工具)
- checkpoint -> per-frame PLY -> 调用现成 pack(`ply_sequence_to_sog4d.py pack`).
- 优点: pack/validate 逻辑复用,实现压力小.
- 缺点: 本仓库当前不存在 pack 工具实现,而且会产生大量 PLY 中间文件,IO 很大.

## 做出的决定
- [2026-02-21 09:12:00 UTC] 选择方向A: 直接写 `.sog4d`.
  - 先实现 `bands=0`(只导出 sh0+opacity),确保 Unity 能导入和播放.
  - 后续如确实需要更高质量的光照,再增量实现 `bands>0` 的 SH rest(palette + labels + delta-v1).

## 关键约束与已选决定
1. `.sog4d` 必须满足 frame-to-frame splat identity 稳定:
   - `splatCount` 固定.
   - layout 固定(row-major,width/height 固定).
2. per-frame 文件路径模板必须包含 `{frame}`,并按至少 5 位 0-padding.
3. 先导出帧数: `frameCount=61`,timeMapping 用 uniform(0..1 均匀).
4. 默认 zip 压缩: stored(因为 WebP 本身已压缩,再 deflate 通常收益不大).

## 状态
**目前在阶段5**: 已完成交付.导出器已落地并成功导出指定 ckpt 的 `.sog4d`.

## 状态更新
- [2026-02-21 09:19:20 UTC] 完成 exporter 落地与导出:
  - 新增脚本: `tools/exportor/export_sog4d.py`(当前实现 `bands=0`).
  - 冒烟导出(5 帧 + 5 万 splats):
    - `results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_f5_k50k.sog4d`
  - 全量导出(61 帧 + 133 万 splats):
    - `results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full.sog4d`(约 1.1G)


# 任务计划: `.sog4d` exporter 支持 bands>0(SH rest, delta-v1)

## 目标
1. 扩展 `tools/exportor/export_sog4d.py`,在 `--sh-bands > 0` 时导出 SH rest:
   - 写入 palette(centroids): `shN_centroids.bin`(little-endian,f16/f32).
   - 写入 labels:
     - base labels: `frames/00000/shN_labels.webp`(u16 label,RG 小端).
     - delta: `sh/shN_delta_00000.bin`(delta-v1,覆盖 [0,frameCount)).
2. 用扩展后的 exporter,把:
   - `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
   再导出一份带 SH rest 的 `.sog4d`(建议新文件名避免覆盖 bands=0 版本).

## 阶段
- [x] 阶段1: 计划与规格复读(spec/importer 约束)
- [x] 阶段2: 实现 SH rest(v1 + delta-v1)
- [x] 阶段3: 冒烟导出与自检(meta/zip 内容)
- [x] 阶段4: 全量导出(61 帧)并记录产物路径
- [x] 阶段5: 文档与提交(四文件回写 + git commit)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(更高质量/压缩更好)
- 采用 `meta.json.version=2` 的 per-band palette:
  - `sh1/sh2/sh3` 分别拟合 codebook + labels(delta-v1).
  - 优点: 单个向量维度更低(9/15/21),聚类更稳,同等 labelCount 下通常误差更小.
  - 代价: 实现与 bundle 文件数量更多.

### 方向B: 先能用,后面再优雅(本次落地)
- 采用 `meta.json.version=1` 的单一 shN palette:
  - `shN_centroids.bin` + `shN_labels.webp` + `delta-v1`.
  - 优点: 实现最直接,先把链路跑通.
  - 代价: 向量维度较高(最多 45),同等 K 下误差可能更大;后续可再升级到方向A.

## 做出的决定
- [2026-02-21 09:37:04 UTC] 先落地方向B: v1 + delta-v1.
  - 理由: 你已经确认"SH 通常静态",delta-v1 几乎是零成本压缩.
  - 后续: 若 Unity 侧对 SH 质量要求更高,再升级实现方向A(v2).

## 关键问题
1. checkpoint 的 `shN` 系数通常是静态的,因此 delta-v1 预计每帧 `updateCount=0`.
2. `.sog4d` 要求 frame-to-frame splat identity 稳定.
   - labels 的裁剪必须是“全局裁剪”,不能每帧裁剪.
3. KMeans 计算成本可能成为瓶颈,需要提供采样与 chunk 参数避免内存/时间爆炸.

## 状态
**目前在阶段5**: 已完成交付.已回写四文件并完成 git 提交.

## 状态更新
- [2026-02-21 09:43:10 UTC] 已完成实现与冒烟导出:
  - exporter: `tools/exportor/export_sog4d.py` 新增 `--sh-bands 1..3`(v1: `shN_centroids.bin` + labels + delta-v1).
  - 冒烟产物(5 帧 + 5 万 splats, bands=3, delta-v1):
    - `results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_f5_k50k_sh3_v1delta.sog4d`
- [2026-02-21 09:45:50 UTC] 已完成全量导出(61 帧 + 133 万 splats, bands=3, shNCount=512, delta-v1):
  - `results/bar_release_full/out_0_61/exports/ckpt_29999_f61_full_sh3_v1delta_k512.sog4d`
- [2026-02-21 09:49:05 UTC] 已完成回写与 git 提交:
  - commit: `23da9b7`

# 任务计划: `export_splat4d.py` 支持 v2(高斯时间核语义)

## 目标
扩展 `tools/exportor/export_splat4d.py`,支持输出 `.splat4d` v2:
- v1(保持兼容): hard window 语义.
  - `time=time0`,`duration=window_length`.
  - 通过 `--temporal-threshold` 把 FreeTimeGS 的高斯时间核近似成窗口.
- v2(新增): Gaussian 语义(更贴近 FreeTimeGS checkpoint).
  - `time=mu_t`(checkpoint 的 `times`)
  - `duration=sigma`(=exp(checkpoint 的 `durations`),并 clamp `min_sigma`)
  - runtime 侧可直接用 `exp(-0.5 * ((t - mu_t)/sigma)^2)` 做 temporal opacity.

## 阶段
- [x] 阶段1: 计划与约束确认(明确 v1/v2 的字段语义)
- [x] 阶段2: 实现 v2 导出(新增 CLI 参数,并保持 v1 不变)
- [x] 阶段3: 冒烟导出与 sanity check(文件大小/写入进度/基本统计)
- [x] 阶段4: 文档与提交(四文件回写 + git commit)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(推荐)
- exporter 同时支持 v1/v2,并在 CLI help 与 docstring 里写清楚:
  - 什么时候该用 v1(旧 importer 仅支持窗口).
  - 什么时候该用 v2(新 importer 支持高斯核,更准确).
- 额外做一个最小 sanity check:
  - 打印 times/sigma 的 min/max,让用户能快速判断是否存在异常 outlier.

### 方向B: 先能用,后面再优雅
- 只加一个 `--splat4d-version 2`,其余先不改,不补输出解释.

## 做出的决定
- [2026-02-21 09:57:20 UTC] 选择方向A: 同时支持 v1/v2,并补清晰的 CLI 文档与最小 sanity 输出.

## 状态
**目前在阶段4**: 已完成交付.已回写四文件并完成 git 提交.

## 状态更新
- [2026-02-21 09:57:20 UTC] 已完成实现与冒烟导出:
  - `tools/exportor/export_splat4d.py` 新增 `--splat4d-version 1|2`(v2=gaussian time kernel).
  - 产物: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_gaussian.splat4d`
- [2026-02-21 10:00:23 UTC] 已完成回写与 git 提交:
  - commit: `45f254e`


# 任务计划: 更新导出说明文档(参数对齐,补齐导出方法)

## 目标
把仓库内关于导出工具的说明文档更新到“以代码为准”的状态:
- `export_sog4d.py`(支持 `--sh-bands > 0` 的 v1 SH rest + delta-v1)的参数与示例命令与当前实现一致.
- `export_splat4d.py`(支持 `--splat4d-version 2`)的参数与示例命令与当前实现一致.
- 文档里若出现未实现的参数(例如 `--sh-version`, `--delta-segment-length`),要明确标注“暂未实现”或移除,避免误导.

## 阶段
- [x] 阶段1: 对齐事实(以 `--help` 为准核对参数名/默认值)
- [x] 阶段2: 更新 `README.md`(补一个最小 Export/Unity 小节)
- [x] 阶段3: 更新 `tools/exportor/FreeTimeGsCheckpointToSog4D.md`(参数表 + 示例命令)
- [x] 阶段4: 回写与提交(WORKLOG/notes/task_plan + git commit)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(推荐,更少踩坑)
- 同时更新:
  - `README.md`(给“怎么用”的入口)
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md`(给“为什么这么做”的施工图)
- 并给出两类可复制粘贴示例:
  - `.sog4d`: `bands=0` 与 `bands=3(delta-v1)` 各一条
  - `.splat4d`: v1 与 v2 各一条

### 方向B: 先能用,后面再优雅
- 只更新 `tools/exportor/FreeTimeGsCheckpointToSog4D.md`,不动 README.

## 做出的决定
- [2026-02-21 10:04:29 UTC] 选择方向A: README + exporter 文档一起更新,以减少参数出入导致的重复沟通.

## 状态
**目前在阶段4**: 已完成交付.文档已更新并完成 git 提交.

## 状态更新
- [2026-02-21 10:06:56 UTC] 已完成文档更新:
  - `README.md` 增加 `Export (Unity)` 小节,对齐 exporter 参数与示例命令.
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 更新参数建议为以 `--help` 为准,并标注未实现项.
- [2026-02-21 10:07:59 UTC] 已完成 git 提交:
  - commit: `981f0d1`


# 任务计划: 参考 DualGS(2409.08353)实现 SH per-band(v2)与可配置 delta segment length

## 目标
在 `tools/exportor/export_sog4d.py` 落地两项能力,用于更贴近 DualGS 的压缩思路,并为 Unity 侧 streaming/随机访问做准备:
- SH rest 支持 per-band palette(`sh1/sh2/sh3`),输出 `meta.json.version=2`.
- delta-v1 支持按 segment 切分,可配置 `delta segment length`,输出多段 `deltaSegments`.

## 阶段
- [x] 阶段1: 计划和约束确认(明确 `.sog4d` v1/v2 与 `.splat4d` 的边界)
- [x] 阶段2: 论文要点摘录(提炼 DualGS 可借鉴点并落到 notes.md)
- [x] 阶段3: 代码实现(exporter v2 per-band + segment delta)
- [x] 阶段4: 冒烟验证(小帧数+小 splat 数,验证 meta 与文件产物一致)
- [x] 阶段5: 文档与四文件回写(README/FreeTimeGsCheckpointToSog4D.md/WORKLOG/notes/LATER_PLANS)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(更贴近论文,更可扩展)
- per-band v2 + delta-v1 全实现:
  - 每个 segment 写 base labels(首帧) + delta 文件.
  - delta body 按“与上一帧不同的 splatId/newLabel”写 update.
- 优点: 真正支持 SH 随时间变化(论文提到仅约 1% 变化),segment 可随机访问.
- 代价: 需要实现 per-frame labels 的生成/对比(当前 FreeTimeGS 通常是静态 SH,短期收益有限).

### 方向B: 先能用,后面再优雅(本次落地,风险最小)
- per-band v2 + segment delta-v1(静态 SH 语义):
  - labels 仍按“静态”处理(跨帧一致).
  - delta 文件按 segment 切分,每帧 updateCount=0.
- 优点: 立刻获得 per-band(质量/聚类更稳)与 segment 切分(满足格式/streaming 需求).
- 代价: 若未来 SH 真随时间变化,仍需补方向A 的 update 生成逻辑.

## 关键问题
1. 你本次提到的 "splat4d per-band/delta segment" 更像是 `.sog4d` 的能力(因为 `.splat4d` 是 64B/record 的无头二进制,目前只能承载 SH0).
2. Unity importer 是否已支持:
   - `meta.json.version=2` 的 `streams.sh.sh1/sh2/sh3`.
   - `labelsEncoding="delta-v1"` 的多段 `deltaSegments`.
3. `delta segment length` 的默认值是否要对齐论文的 50,还是保持当前“1 段覆盖全帧”的行为以保证兼容.

## 做出的决定
- [2026-02-21 12:24:53 UTC] 先落地方向B:
  - 理由: FreeTimeGS 的 SH 通常静态,delta-v1 的 update 基本为 0,优先把 per-band 与 segment 基础设施打通.
  - 默认兼容: `--delta-segment-length` 默认保持“单 segment 覆盖全帧”,避免无意改变旧输出.

## 状态
**目前在阶段5**: 已完成交付.代码+文档+四文件均已回写,并完成冒烟验证。

## 状态更新
- [2026-02-21 12:39:32 UTC] 已完成实现:
  - `tools/exportor/export_sog4d.py` 新增 `--sh-version 1|2`,`--delta-segment-length`.
  - `meta.json.version=2` 支持 per-band palettes(`sh1/sh2/sh3`),并按 spec 写入 `deltaSegments`.
  - delta-v1 支持多 segment,并保持默认 `0=单 segment 覆盖全帧` 的兼容行为.
  - SH kmeans 增加 empty cluster 捕获与重试,提升稳定性.
- [2026-02-21 12:39:32 UTC] 已完成验证与文档回写:
  - `python -m compileall` 通过,并导出 v1/v2 冒烟 `.sog4d` 验证 meta.json 与 zip 内容一致.
  - `README.md`,`tools/exportor/FreeTimeGsCheckpointToSog4D.md`,`notes.md`,`WORKLOG.md`,`LATER_PLANS.md`,`task_plan.md` 已同步更新.


# 任务计划: 升级 `.splat4d`(v2)以对齐 `.sog4d v2` 的能力(per-band SH + deltaSegments)

## 目标
让 `.splat4d` 同时具备:
- SH rest 的 per-band 量化与存储(`sh1/sh2/sh3`).
- 可配置的 delta segment length(用于 labels 的分段与随机访问/流式基础设施).
- 明确且可版本化的文件格式,并与 Unity 侧 importer/runtime 行为一致(包含时间核语义).

## 阶段
- [x] 阶段1: 计划与格式设计(含向后兼容策略)
- [x] 阶段2: 论文要点补充摘录(DualGS 压缩/播放链路还能借鉴什么)
- [x] 阶段3: FreeTimeGsVanilla exporter 落地(`export_splat4d.py`)
- [x] 阶段4: Unity importer/runtime 落地(`gsplat-unity`)
- [x] 阶段5: 冒烟验证 + 文档/四文件回写

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(推荐,单文件+可扩展)
- 设计 `.splat4d` v2: 以 header + section table 的方式扩展单文件格式.
- 在 Unity importer 里同时支持:
  - v1(无 header,64B/record).
  - v2(有 header,包含 SH per-band 与 deltaSegments).
- 时间核语义显式化:
  - window(time0+duration)与 gaussian(mu+sigma)二选一,避免 exporter/runtime 口径不一致.
- 优点:
  - 结构可扩展,后续加 streaming/更多属性不会再破坏格式.
  - 单文件分发更顺手,也更接近 `.sog4d` 的 bundle 体验.
- 代价:
  - Unity importer/runtime 必须同步升级.

### 方向B: 先能用,后面再优雅(多文件 sidecar,改动小)
- 保持 `.splat4d` 仍为 v1 record 数组.
- 额外输出 sidecar:
  - `xxx.splat4d.meta.json`
  - `xxx.splat4d.sh1_centroids.bin`/`sh1_labels.bin`/`sh1_delta_*.bin` 等
- Unity importer 通过约定路径一起加载.
- 优点:
  - 不需要改动 `.splat4d` 本体,实现更快.
- 代价:
  - 多文件生命周期管理麻烦,更容易丢文件/路径错.
  - 后续想要 streaming 时仍要再补“bundle化”工作.

## 关键问题
1. `.splat4d v1` 是无 header 的 64B record 数组,如何在不破坏兼容的情况下增加 SH 与 deltaSegments?
2. Unity 侧目前按 hard-window(time0+duration)做可见性裁剪与排序,而 FreeTimeGS checkpoint 的时间核是 gaussian(mu+sigma).升级后必须口径一致.
3. deltaSegments 是基于离散 frame 的概念,而 `.splat4d` 的播放时间是归一化连续值.需要明确 timeMapping 与 frameIndex 的映射策略(即使一期先不做真正的 SH 随时间更新,也要把格式铺好).

## 做出的决定
- [2026-02-21 13:40:00 UTC] 选择方向A:
  - 理由: 用户目标是“升级 `.splat4d` 并对齐 `.sog4d v2` 能力”,单文件+可版本化最不易腐化.
  - 兼容策略: Unity importer 通过 magic 检测 v2,否则回退 v1 解析.
- [2026-02-21 13:40:00 UTC] delta-v1 一期先实现“静态 labels”(updateCount=0)与可配置 segment 切分:
  - 理由: FreeTimeGS 的 `shN` 通常静态,先把基础设施打通,后续再补“真实 changed label”生成逻辑.

## 状态
**目前在阶段5**: 已完成冒烟验证,并完成文档/四文件回写.

## 状态更新
- [2026-02-21 14:10:00 UTC] 已完成 `.splat4d v2` 格式与实现落地:
  - FreeTimeGsVanilla: `tools/exportor/export_splat4d.py` 支持:
    - `--splat4d-format-version 2`(header+sections)
    - per-band SH(`--sh-bands 1..3`,导出 `sh1/sh2/sh3` centroids+labels+deltaSegments)
    - `--delta-segment-length`(分段长度,0=单段覆盖全帧)
  - Unity(gsplat-unity):
    - `Editor/GsplatSplat4DImporter.cs` 支持 v1/v2 `.splat4d` 导入,并在 v2 下解码 SH rest.
    - `Runtime/Shaders/Gsplat.compute`/`Runtime/Shaders/Gsplat.shader` 增加 `TimeModel=window|gaussian` 与 `TemporalCutoff`,并实现 gaussian 时间核的可见性与平滑淡入淡出.
    - `Runtime/GsplatSortPass.cs`/`Runtime/GsplatSorter.cs`/`Runtime/GsplatRendererImpl.cs` 补齐新 uniform 传参链路.
- [2026-02-21 14:10:00 UTC] 已完成一次冒烟导出(含 SH3 + deltaSegments):
  - 输出: `results/bar-release_result_run2/exports_smoke/ckpt_29999_v2_sh3_seg10.splat4d`
  - 参数: `--shn-count 64 --delta-segment-length 10 --frame-count 61 --splat4d-version 2 --splat4d-format-version 2`
- [2026-02-21 14:33:33 UTC] 已完成本仓库侧冒烟验证(导出+解析校验):
  - 导出: `results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_v2_sh3_seg10_op99_k64.splat4d`
  - 解析校验(用 Python 直接读取 header/section table):
    - header: `magic=SPL4DV02`, `sectionCount=47`, `splatCount=91596`, `shBands=3`, `timeModel=2`, `frameCount=61`
    - sections: `RECS=1`,`META=1`,`SHCT=3`,`SHLB=21`,`SHDL=21`
    - RECS: `length == splatCount * 64`
    - delta: magic=`SPL4DLB1`,header 中的 `(segmentStartFrame,segmentFrameCount)` 与 section entry 一致


# 任务计划: 配置 Git 远端并推送到 raiscui/FreeTimeGsVanilla

## 目标
把当前 `main` 分支的最新提交与本次未提交改动,推送到 `https://github.com/raiscui/FreeTimeGsVanilla.git`.
同时保留原上游远端,用于后续同步.

## 阶段
- [ ] 阶段1: 计划和设置
- [ ] 阶段2: 确认变更范围
- [ ] 阶段3: 提交本地变更
- [ ] 阶段4: 配置远端并 push
- [ ] 阶段5: 验证远端状态

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(推荐,符合 fork/upstream 习惯)
- 将当前 `origin` 重命名为 `upstream`(保留原仓库 `OpsiClear/FreeTimeGsVanilla`).
- 新增 `origin` 指向 `raiscui/FreeTimeGsVanilla`.
- 之后 `main` 跟踪 `origin/main`,需要同步上游时用 `upstream`.

### 方向B: 先能用,后面再优雅(改动最少)
- 保持现有 `origin` 不变.
- 新增一个新远端名(如 `raiscui`)指向你的仓库.
- push 时显式 `git push raiscui main`.

## 关键问题
1. 本次提交是否包含 `.codex/` 与 `openspec/config.yaml` 这类“开发工作流”文件?
2. 是否需要把 `.vscode/settings.json` 这类个人 IDE 外观配置一并推送?

## 做出的决定
- [2026-02-21 16:06:41 UTC] 选择方向A:
  - 理由: 让后续 push 默认指向你的仓库,同时保留上游用于同步.
  - 操作: `origin -> upstream`,新增 `origin=https://github.com/raiscui/FreeTimeGsVanilla.git`.
- [2026-02-21 16:06:41 UTC] 提交范围策略:
  - 代码/文档/四文件上下文(`task_plan.md`,`notes.md`,`WORKLOG.md`,`LATER_PLANS.md`)全部纳入提交.
  - `.codex/` 与 `openspec/config.yaml` 作为项目工作流辅助文件一并纳入.
  - `.vscode/settings.json` 暂不提交(纯外观配置,容易污染团队仓库).

## 状态
**目前在阶段1**: 已读取 git 状态,已追加本任务记录,准备开始配置远端与提交/push.
