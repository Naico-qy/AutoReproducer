#!/usr/bin/env bash
# reproduce.sh — PaperBench Reproduction entrypoint for FARE / Robust CLIP.
#
# Steps:
#   1. Install Python dependencies (idempotent)
#   2. Run a SHORT training (smoke run by default; the FARE^2 ViT-L/14 training
#      from the paper takes ~12-24 GPU hours on 4xA100, which is infeasible
#      under PaperBench's wall-clock budget; long runs can be enabled by
#      setting `FARE_LONG_RUN=1` in the environment).
#   3. Run zero-shot classification evaluation on a small subset of datasets
#      and write metrics JSON to /output/metrics.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "${OUTPUT_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
mkdir -p "${CHECKPOINT_DIR}"

echo "[reproduce.sh] === Step 1: install dependencies ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[reproduce.sh] === Step 2: training ==="
EPSILON="${FARE_EPSILON:-0.00784313725490196}"   # 2/255 by default (FARE^2)
SMOKE_FLAG="--smoke"
if [[ "${FARE_LONG_RUN:-0}" == "1" ]]; then
  SMOKE_FLAG=""   # full 2-epoch ImageNet run
fi
python train.py \
  --config configs/default.yaml \
  --output_dir "${CHECKPOINT_DIR}" \
  --epsilon "${EPSILON}" \
  ${SMOKE_FLAG}

CHECKPOINT_PATH="${CHECKPOINT_DIR}/fare_final.pt"

echo "[reproduce.sh] === Step 3: evaluation ==="
# In smoke mode we evaluate a subset of datasets (CIFAR10/100/STL10 are tiny
# and don't require ImageNet-scale auth/downloads).
EVAL_DATASETS="${FARE_EVAL_DATASETS:-cifar10 cifar100 stl10}"

python eval.py \
  --config configs/default.yaml \
  --checkpoint "${CHECKPOINT_PATH}" \
  --metrics_path "${OUTPUT_DIR}/metrics.json" \
  --datasets ${EVAL_DATASETS} \
  ${SMOKE_FLAG}

echo "[reproduce.sh] DONE. Metrics:"
cat "${OUTPUT_DIR}/metrics.json"
