#!/usr/bin/env bash
# 一键跑通: mp4 -> 抽帧 -> 参考帧 COLMAP -> RoMA 逐帧三角化 -> combine(all_frames) -> train

set -euo pipefail

MP4_DIR="${1:-}"
WORK_DIR="${2:-}"
RESULT_DIR="${3:-}"
START_FRAME="${4:-}"
END_FRAME="${5:-}"
GPU_ID="${6:-}"
CONFIG="${7:-paper_stratified_small}"

if [[ -z "${MP4_DIR}" || -z "${WORK_DIR}" || -z "${RESULT_DIR}" || -z "${START_FRAME}" || -z "${END_FRAME}" || -z "${GPU_ID}" ]]; then
  echo "Usage:"
  echo "  bash run_mp4_pipeline.sh <mp4_dir> <work_dir> <result_dir> <start_frame> <end_frame> <gpu_id> [config]"
  echo ""
  echo "Example:"
  echo "  bash run_mp4_pipeline.sh ./mp4 ./work ./results 0 61 0 paper_stratified_small"
  echo ""
  echo "Notes:"
  echo "  - frame range is [start_frame, end_frame) (end exclusive)."
  echo "  - To overwrite existing outputs, set OVERWRITE=1."
  exit 1
fi

source .venv/bin/activate

DATA_DIR="${WORK_DIR}/data"
TRI_DIR="${WORK_DIR}/triangulation"
INIT_NPZ="${WORK_DIR}/init_${START_FRAME}_${END_FRAME}.npz"

mkdir -p "${WORK_DIR}"
mkdir -p "${RESULT_DIR}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

echo "========================================"
echo "MP4 -> 4DGS Pipeline (RoMA all-frames)"
echo "========================================"
echo "mp4_dir:        ${MP4_DIR}"
echo "work_dir:       ${WORK_DIR}"
echo "data_dir:       ${DATA_DIR}"
echo "tri_dir:        ${TRI_DIR}"
echo "init_npz:       ${INIT_NPZ}"
echo "result_dir:     ${RESULT_DIR}"
echo "frame range:    [${START_FRAME}, ${END_FRAME})"
echo "gpu:            ${GPU_ID}"
echo "config:         ${CONFIG}"
echo "overwrite:      ${OVERWRITE:-0}"
echo "========================================"

echo ""
echo "[Step 1/3] Preprocess: extract frames + ref COLMAP + RoMA triangulate"
python src/preprocess_mp4_freetimegs.py \
  --mp4-dir "${MP4_DIR}" \
  --data-dir "${DATA_DIR}" \
  --triangulation-dir "${TRI_DIR}" \
  --start-frame "${START_FRAME}" \
  --end-frame "${END_FRAME}" \
  --reference-frame "${START_FRAME}" \
  "${OVERWRITE_FLAG[@]}"

echo ""
echo "[Step 2/3] Combine: all_frames -> init.npz"
python src/combine_frames_fast_keyframes.py \
  --mode all_frames \
  --input-dir "${TRI_DIR}" \
  --output-path "${INIT_NPZ}" \
  --frame-start "${START_FRAME}" \
  --frame-end "${END_FRAME}" \
  --keyframe-step 1

echo ""
echo "[Step 3/3] Train: 4D Gaussians"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/simple_trainer_freetime_4d_pure_relocation.py "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --init-npz-path "${INIT_NPZ}" \
  --result-dir "${RESULT_DIR}" \
  --start-frame "${START_FRAME}" \
  --end-frame "${END_FRAME}" \
  --max-steps 30000 \
  --eval-steps 30000 \
  --save-steps 30000

echo ""
echo "========================================"
echo "DONE"
echo "  result_dir: ${RESULT_DIR}"
echo "========================================"

