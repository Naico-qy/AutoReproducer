#!/usr/bin/env bash
# Reproduce.sh — entrypoint for PaperBench Full / Reproduction grading.
#
# This runs a small smoke-quality training+eval cycle of the LBCS pipeline
# on F-MNIST with k=1000 and writes /output/metrics.json.  A long full-paper
# reproduction (T=500, epochs=100, multi-dataset, multi-seed) is far beyond a
# 24h budget; we prioritize a clean evidence-of-correctness run that still
# executes the LBCS algorithm end-to-end.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"

OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "${OUTPUT_DIR}"

echo "[reproduce.sh] installing dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "[reproduce.sh] running LBCS smoke training (F-MNIST, k=1000) ..."
python train.py \
    --config configs/default.yaml \
    --dataset fmnist \
    --k 1000 \
    --epsilon 0.2 \
    --output-dir "${OUTPUT_DIR}" \
    --data-root "${HERE}/data_root" \
    --smoke

echo "[reproduce.sh] running evaluation ..."
python eval.py \
    --config configs/default.yaml \
    --output-dir "${OUTPUT_DIR}"

echo "[reproduce.sh] final metrics.json:"
cat "${OUTPUT_DIR}/metrics.json"
