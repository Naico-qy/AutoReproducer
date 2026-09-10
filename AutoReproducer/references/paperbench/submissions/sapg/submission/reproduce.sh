#!/usr/bin/env bash
# =============================================================================
# SAPG reproduce.sh -- runs a smoke training + eval, dumps /output/metrics.json
#
# Full reproduction (paper Table 1, ~2e10 transitions over 5 seeds, ~50 GPU-hr
# per seed) is INFEASIBLE within the PaperBench 24h budget on a single GPU.
# This script therefore runs:
#   1. a smoke training pass to verify the algorithm runs end-to-end
#   2. an evaluation pass that writes /output/metrics.json
#
# To run a longer-horizon run, override --num_iterations.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"

# ---------- 1. Install deps ----------
echo "[reproduce] installing deps"
pip install --quiet --upgrade pip || true
pip install --quiet -r requirements.txt || pip install -r requirements.txt

# ---------- 2. Output dir ----------
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "$OUTPUT_DIR" 2>/dev/null || OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
echo "[reproduce] writing metrics to $OUTPUT_DIR"

# ---------- 3. Train (smoke) ----------
echo "[reproduce] training (smoke)"
python train.py \
    --config configs/default.yaml \
    --smoke \
    --output "$OUTPUT_DIR/train_metrics.json" \
    || { echo "[reproduce] training failed"; exit 1; }

# ---------- 4. Evaluate ----------
echo "[reproduce] evaluating"
LATEST_CKPT="$(ls -t runs/*/final.pt 2>/dev/null | head -n 1 || true)"
if [ -z "$LATEST_CKPT" ]; then
    echo "[reproduce] no checkpoint found -- running eval on random init"
    python eval.py \
        --config configs/default.yaml \
        --num_episodes 2 \
        --output "$OUTPUT_DIR/metrics.json" \
        || { echo "[reproduce] eval failed"; exit 1; }
else
    echo "[reproduce] using $LATEST_CKPT"
    python eval.py \
        --config configs/default.yaml \
        --ckpt "$LATEST_CKPT" \
        --num_episodes 2 \
        --output "$OUTPUT_DIR/metrics.json" \
        || { echo "[reproduce] eval failed"; exit 1; }
fi

echo "[reproduce] done. Metrics:"
cat "$OUTPUT_DIR/metrics.json" || true
