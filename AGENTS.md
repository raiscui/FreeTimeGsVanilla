# Repository Guidelines

## 关联项目
`/workspace/gsplat-unity`
这个gsplat-unity 是 splat4d sog4d 格式 在 unity 的显示解码功能项目, 本项目计算生成的4d gsplat, 要能使用 此gsplat-unity 在unity中 加载 并解码播放,如果 编码在 本项目中更改,那么 gsplat-unity 的解码要做对应变更.


## proxy
- 运行网络命令前要注册代理: `export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897`

## Project Structure & Module Organization
- `src/`: Core pipeline scripts.
  - `combine_frames_fast_keyframes.py`: Build keyframes + velocity NPZ.
  - `simple_trainer_freetime_4d_pure_relocation.py`: Main 4D GS trainer.
  - `viewer_4d.py`: Interactive viewer for trained checkpoints.
- `datasets/`: COLMAP I/O + dataset utilities (e.g. `FreeTime_dataset.py`).
- `assets/`: Small, documentation-friendly media (e.g. `assets/demo.gif`).
- `run_*.sh`: Convenience entrypoints for common runs.
- Generated outputs belong in `results/` (gitignored) and should not be committed.

## Build, Test, and Development Commands
This repo uses `uv` and a local virtualenv at `.venv/` (Python `>=3.12`).
- Install deps (reproducible): `uv sync --locked`
- Activate env: `source .venv/bin/activate`
- Full pipeline (combine -> train): `bash run_pipeline.sh <input_dir> <data_dir> <result_dir> <start> <end> <step> <gpu_id> [config]`
- Train only (example): `CUDA_VISIBLE_DEVICES=0 python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe_small ...`
- Viewer (example): `CUDA_VISIBLE_DEVICES=0 python src/viewer_4d.py --ckpt results/ckpts/ckpt_29999.pt --port 8080 ...`

## Coding Style & Naming Conventions
- Python: 4-space indentation, keep functions focused, prefer explicit names.
- Keep changes minimal and consistent with surrounding code (research code favors readability over abstraction).
- Do not commit large artifacts (see `.gitignore`: `*.mp4`, `*.pt`, `*.npz`, `results/`).

## Testing Guidelines
There is no formal test suite yet. Before opening a PR, run at least:
- Syntax/smoke: `python -m compileall src datasets`
- Import sanity: `python -c "import torch, gsplat; print('ok')"`

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (e.g. "Add ...", "Fix ...", "Refactor ...").
- PR description should include: intent, key files changed, and a single reproducible command.
- Visual changes: attach a small GIF/screenshot (prefer `assets/`, keep it small).
- Dependency changes: update `pyproject.toml` and commit the updated `uv.lock`.
