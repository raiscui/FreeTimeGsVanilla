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

## 2026-02-21 09:57:20 UTC
- 扩展 `.splat4d` exporter 支持 v2(gaussian 时间核语义):
  - `tools/exportor/export_splat4d.py` 新增 `--splat4d-version 1|2`.
    - v1: hard window(保持兼容).
    - v2: gaussian(time=mu_t,duration=sigma),更贴近 FreeTimeGS checkpoint.
- 已完成对大 ckpt 的 v2 导出:
  - `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_gaussian.splat4d`(约 81.5MB)

## 2026-02-21 10:06:56 UTC
- 更新导出说明文档,对齐当前 exporter 的真实参数名与示例命令:
  - `README.md` 增加 `Export (Unity)` 小节,包含 `.sog4d`(bands=0/sh3)与 `.splat4d`(v1/v2)的可复制命令.
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 更新参数建议,移除/标注未实现的参数,以 `--help` 为准对齐.

## 2026-02-21 12:39:32 UTC
- 参考 DualGS(2409.08353)补齐 `.sog4d` exporter 的两项能力:
  - per-band SH rest(v2): `--sh-version 2` 输出 `meta.json.version=2`,并生成 `sh1/sh2/sh3` 三套 palette + labels.
  - 可配置 delta segment length: `--delta-segment-length` 控制 delta-v1 的 segment 切分(0=保持旧行为: 单段覆盖全帧).
- `.sog4d` v2(per-band)输出布局要点:
  - centroids: `sh/sh1_centroids.bin`,`sh/sh2_centroids.bin`,`sh/sh3_centroids.bin`
  - delta-v1:
    - base labels: 只在 segment 起始帧写 `frames/{startFrame}/sh1_labels.webp` 等
    - delta: `sh/sh1_delta_{startFrame}.bin` 等(每段 1 个)
- 稳定性改良:
  - SH kmeans 捕获 scipy empty cluster warning 并最多重试 3 次,避免导出质量不稳/终端噪声.
- 文档同步:
  - `README.md` 增加 per-band(v2)导出示例.
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 更新参数表,移除“未实现”标注并补 `--sh-version/--delta-segment-length`.

验证:
```bash
python -m compileall -q src datasets tools/exportor
python -c "import torch, gsplat; print('ok')"

source .venv/bin/activate
python tools/exportor/export_sog4d.py \
  --ckpt-path results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt \
  --output-path results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_f3_k50k_sh3_perband_seg2_retry.sog4d \
  --frame-count 3 \
  --layout-width 1024 \
  --max-splats 50000 \
  --sh-bands 3 \
  --sh-version 2 \
  --shn-count 64 \
  --shn-labels-encoding delta-v1 \
  --delta-segment-length 2 \
  --shn-codebook-sample 20000 \
  --shn-kmeans-iters 5 \
  --zip-compression stored \
  --overwrite
```

## 2026-02-21 14:33:33 UTC
- 升级 `.splat4d` exporter 支持 format v2(header+sections),用于把 `.sog4d v2` 的两项能力带到 `.splat4d`:
  - per-band SH rest(`sh1/sh2/sh3`): `--sh-bands 1..3` 导出每个 band 的 codebook(centroids)与 base labels(u16).
  - deltaSegments(可配置段长): `--shn-labels-encoding delta-v1` + `--frame-count` + `--delta-segment-length` 写入多段 delta blocks(一期默认 updateCount=0,为未来 SH 动态变化铺路).
- 已完成冒烟导出+解析校验:
  - 输出: `results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_v2_sh3_seg10_op99_k64.splat4d`
  - 校验: header/section table 与数据体一致(sectionCount=47; RECS bytes=splatCount*64; delta magic=`SPL4DLB1`).

复现命令:
```bash
source .venv/bin/activate
python tools/exportor/export_splat4d.py \
  --ckpt results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt \
  --output results/bar_release_full/out_0_61/exports_smoke/ckpt_29999_v2_sh3_seg10_op99_k64.splat4d \
  --splat4d-format-version 2 \
  --splat4d-version 2 \
  --base-opacity-threshold 0.99 \
  --sh-bands 3 \
  --frame-count 61 \
  --shn-count 64 \
  --shn-centroids-type f16 \
  --shn-labels-encoding delta-v1 \
  --delta-segment-length 10 \
  --shn-codebook-sample 20000 \
  --shn-kmeans-iters 5
```

## 2026-02-21 16:32:32 UTC
- 使用 direnv 创建"私有 .envrc"(不进 git),用于本地代理与 GitHub PAT 管理.
- 新增 `.direnv/git-askpass.sh` 并设置可执行,让 git 在非交互模式下通过 `GIT_ASKPASS` 读取 `GITHUB_TOKEN` 完成 https 认证.
- 在 `.git/info/exclude` 追加忽略规则: `.envrc`,`.envrc.private`,`.direnv/`,`.vscode/`,避免误提交并让 `git status` 保持干净.
- 执行 `direnv allow`,并验证 `direnv export bash` 正常导出相关环境变量.

## 2026-02-21 16:39:25 UTC
- 完成推送到你的仓库: `git push -u origin main` -> `https://github.com/raiscui/FreeTimeGsVanilla.git`.
- 验证远端 `origin/main` 与本地 `HEAD` 一致(sha=`2965f15`),并确认本地 `main` 已跟踪 `origin/main`.

## 2026-02-22 02:33:37 UTC
- 修复 `.splat4d` 导出脚本的一个高频误用点: 当用户想导出 timeModel=2(gaussian)时,旧默认会输出 legacy v1(无 header)文件,导致 Unity importer 仅靠 header 判定时走错 v1 路径,出现"薄层/稀疏"的裁剪伪影.
- exporter 改动:
  - `tools/exportor/export_splat4d.py` 新增 `--splat4d-format-version 0=auto` 并作为默认值.
  - auto 规则: `--splat4d-version=2` 或 `--sh-bands>0` 时默认输出 format v2(header+sections,magic=`SPL4DV02`),否则输出 legacy v1.
  - 当用户显式指定 `--splat4d-version 2` + `--splat4d-format-version 1` 时,在 stderr 打印醒目 warning,避免继续误用.
- 文档同步:
  - `README.md` 补充 `--splat4d-format-version` 的 auto 默认说明.
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 更新为 `0|1|2`,并把 timeModel vs formatVersion 拆开说明.
- 验证:
  - `python3 -m compileall -q src datasets tools/exportor`
  - 用最小 ckpt 导出并检查二进制头:
    - `python3 tools/exportor/export_splat4d.py --splat4d-version 2`(不指定 format)会打印 `format=auto -> v2` 且输出文件前 8 bytes 为 `SPL4DV02`.

## 2026-02-22 02:37:54 UTC
- 使用修复后的默认值,重新导出一份"带 header"的 gaussian `.splat4d`,用于直接替换 Unity 侧的旧文件:
  - 输入 ckpt: `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
  - 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_gaussian_fmt2.splat4d`(magic=`SPL4DV02`)

## 2026-02-22 03:53:03 UTC
- 重新用同一 ckpt 导出一份“高质量 v2(gaussian) + per-band SH rest + deltaSegments”的 `.splat4d format v2`,用于 Unity 最终观感测试:
  - 输入 ckpt: `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
  - 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f16.splat4d`
  - 关键参数:
    - `--splat4d-version 2`(timeModel=2,gaussian)
    - `--sh-bands 3`(sh1/sh2/sh3 per-band codebook)
    - `--frame-count 61 --delta-segment-length 50`(deltaSegments,分 2 段)
    - `--shn-count 512 --shn-centroids-type f16 --shn-codebook-sample 200000`
  - 快速验证:
    - 读取文件前 8 bytes 为 `SPL4DV02`

## 2026-02-22 07:10:28 UTC
- 对齐关联项目 `/workspace/gsplat-unity` 的 importer 能力,确认是否需要同步改动以支持:
  - `.splat4d format v2` 的 per-band SH rest codebooks
  - 可配置的 delta segment length(delta-v1 segments)
- 结论: 当前 `gsplat-unity` 已经实现相关解析/解码逻辑,无需额外改动即可消费本仓库导出的 `.splat4d v2`:
  - `Editor/GsplatSplat4DImporter.cs` 支持 `SPL4DV02` header+sections,并读取 `SHCT/SHLB/SHDL` 解码到 `GsplatAsset.SHs`.
  - 对 `labelsEncoding=delta-v1` 会校验 segments 覆盖性与 delta header(当前 exporter 默认 `updateCount=0`,因此仅用 base labels 也能得到正确结果).

## 2026-02-22 15:39:40 +0800
- 修复 `.sog4d` exporter 的 meta.json schema,避免导入 Unity 前就失败:
  - `tools/exportor/export_sog4d.py`:
    - meta.json 顶层补齐 `format="sog4d"`(gsplat-unity 侧用于 fail-fast).
    - `streams.position.rangeMin/rangeMax` 与 `streams.scale.codebook` 的 float3 数组改为 Vector3 JSON:
      - 输出为 `[{"x":..,"y":..,"z":..}, ...]`,而不是 `[[x,y,z], ...]`.
      - 目的: 兼容 Unity `JsonUtility` 解析 `Vector3[]`.
  - `tools/exportor/spec.md`:
    - 补齐 `meta.json.format` 的 MUST 约束.
    - 明确 float3 的 JSON 序列化形态必须为 `{x,y,z}`.
- 验证:
  - `python3 -m py_compile tools/exportor/export_sog4d.py`
  - `python3 -m compileall -q tools/exportor`

## 2026-02-22 09:11:06 UTC
- 合并分叉: 将 `00eb763` 合并进本地 `main`,并解决冲突(WORKLOG/notes/task_plan).
  - merge commit: `6929ad7`
  - 状态记录 commit: `0b57500`
- 已推送到 `origin/main`(使用 `direnv exec . git push ...` 读取 `.envrc.private` 凭据).

## 2026-02-22 10:20:30 UTC
- 修复 Unity 中 `.splat4d` 点云整体偏移/歪倒(训练 normalize 空间 vs COLMAP 原始空间不一致):
  - `tools/exportor/export_splat4d.py` 新增:
    - `--output-space train|colmap`(默认 train)
    - `--colmap-dir <sparse/0>`(当 output-space=colmap 时必填)
  - 导出时复现训练侧 `colmap->train` transform,并对 `(position, velocity, scale, rotation)` 统一应用 `T^{-1}` 导出回 COLMAP 原始空间.
  - `.splat4d v2` 额外写入 `XFRM` section(64B,16xf32)记录 `colmap->train` transform,用于离线 debug(不影响现有 importer).
- 重新导出(用于 Unity 实测):
  - 输入 ckpt: `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
  - 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f16_colmap.splat4d`
  - 验证: magic=`SPL4DV02`,section table 含 `XFRM`.
- 文档同步:
  - `README.md` 增加 Unity/COLMAP 坐标对齐说明与示例命令.
  - `tools/exportor/FreeTimeGsCheckpointToSog4D.md` 补齐新参数说明.

## 2026-02-22 13:31:47 UTC
- 输出 OpenSpec change: `.splat4d delta-v1 真实 updates + Unity 运行时应用`.
  - change dir: `openspec/changes/splat4d-delta-v1-sh-updates/`
  - artifacts: `proposal.md`, `design.md`, `specs/**/spec.md`, `tasks.md`
  - 状态: apply-ready(后续可以直接按 `tasks.md` 进入实现阶段)

## 2026-02-22 14:47:09 UTC
- 完成 `.splat4d format v2` delta-v1 的“真实 updates”闭环(Exporter + Unity runtime).
- FreeTimeGsVanilla(exporter):
  - `tools/exportor/export_splat4d.py` 支持动态 `splats["shN"]`(4D),生成逐帧 labels 并写入 delta-v1 updates(仅写 changed `(splatId,newLabel)`).
  - per-segment base labels: 每个 segment 写一份 SHLB base labels,并为每段写 SHDL(delta bytes),保证 segment 边界语义正确.
  - 新增 `--shn-frame-axis` 与 `--self-check-delta`,并输出 delta stats(changedPercent/avg/maxUpdateCount).
  - 新增 `tools/synth_dynamic_shn_ckpt.py` 作为可重复验证样例(确保至少出现 `updateCount>0`).
- gsplat-unity(importer + runtime):
  - `Editor/GsplatSplat4DImporter.cs`(v3)在 delta-v1 时把 per-segment 的 base labels/delta bytes 持久化到 `Runtime/GsplatAsset.cs`.
  - `Runtime/GsplatRenderer.cs` 运行时按 `TimeNormalized` 选帧应用 deltas,用 compute scatter 更新 `SHBuffer`.
  - 新增 `Runtime/Shaders/GsplatShDelta.compute` 与 `Tests/Editor/GsplatSplat4DImporterDeltaV1Tests.cs`,并更新 `CHANGELOG.md`.
- 验证(当前容器可执行部分):
  - `python3 -m compileall -q tools/exportor tools src datasets` 通过.
  - 合成 ckpt + exporter 自检通过,并确认存在非 0 updates:
    - `python3 tools/synth_dynamic_shn_ckpt.py --output results/synth_delta_v1_verify/ckpt_synth_dynamic_shn_f5_n1024_axis0.pt --frames 5 --splats 1024 --sh-bands 3 --shn-frame-axis 0`
    - `python3 tools/exportor/export_splat4d.py --ckpt results/synth_delta_v1_verify/ckpt_synth_dynamic_shn_f5_n1024_axis0.pt --output results/synth_delta_v1_verify/ckpt_synth_dynamic_shn_f5_n1024_axis0_v2_sh3_delta_seg2_k8_f32.splat4d --splat4d-format-version 2 --splat4d-version 2 --sh-bands 3 --frame-count 5 --shn-frame-axis 0 --shn-count 8 --shn-centroids-type f32 --shn-labels-encoding delta-v1 --delta-segment-length 2 --shn-codebook-sample 4000 --shn-kmeans-iters 5 --shn-assign-chunk 2048 --self-check-delta`
    - exporter 输出: `sh band=1 delta stats ... maxUpdateCount=120`.
- 待补跑(需要 Unity Editor 环境):
  - 运行 gsplat-unity 的 EditMode tests(含新增 delta-v1 测试).
  - (可选) 手动拖动/播放 `TimeNormalized` 观察 SH rest 随帧变化.

## 2026-02-22 16:24:34 UTC
- 按你的请求,从真实大 ckpt 输出一份“高质量 + 最新 exporter 行为”的 `.splat4d format v2`(用于 Unity):
  - 输入 ckpt: `results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`
  - 输出 splat4d: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f32_colmap_latest.splat4d`
  - 关键特性:
    - format v2(header+sections,`SPL4DV02`)
    - timeModel=2(gaussian)
    - SH per-band(1..3) + labelsEncoding=delta-v1 + segments(frameCount=61,segLen=50)
    - output-space=colmap(包含 `XFRM` section 记录 `colmap->train` transform,导出时对 records 做 train->colmap 反变换)
- 轻量校验:
  - header: splatCount=1,335,131,shBands=3,timeModel=2,frameCount=61
  - sections: `RECS/META/XFRM` + 每 band 的 `SHCT/SHLB/SHDL`,并且 segments 覆盖 `[0,50]` 与 `[50,11]`.

## 2026-02-22 16:43:40 UTC
- 按你要求更新 `README.md` 的“高质量输出命令写法”:
  - `Export (Unity) -> Export .splat4d (binary)` 增加一条“已验证的真实大 ckpt”导出命令,直接输出到:
    - `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f32_colmap_latest.splat4d`
  - 同时把 Export 小节里的示例命令统一改为 `python3 ...`,避免环境里 `python` 指向不一致导致的误用.
