#!/usr/bin/env bash
# reproduce.sh — PaperBench reproduction entrypoint.
#
# Runs a short Adam → L-BFGS → NNCG training pipeline on the convection
# PDE (the cheapest of the three studied PDEs in Rathore et al. 2024)
# and writes /output/metrics.json.  The full 41 000-iteration sweep
# from Section 2.2 is too large to fit comfortably in PaperBench's
# 24 hour budget when also covering reaction + wave with 5 seeds, so
# we use a shortened schedule by default; flip SMOKE=0 below to run
# the full schedule.

set -euo pipefail

cd "$(dirname "$0")"

# ---- 1. Dependencies --------------------------------------------------
# In the PaperBench reproduction container, pip / torch are already
# installed; we still try to install our requirements (no-op if
# already satisfied).
if command -v pip >/dev/null 2>&1; then
    pip install --no-cache-dir --quiet -r requirements.txt || true
fi

# ---- 2. Output directory ----------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "$OUTPUT_DIR"

# ---- 3. Smoke-mode toggle --------------------------------------------
# SMOKE=1 (default) → fast, ~few minutes on CPU/GPU; SMOKE=0 → paper config.
SMOKE_FLAG="${SMOKE:-1}"
if [ "$SMOKE_FLAG" = "1" ]; then
    SMOKE_ARG="--smoke"
    echo "[reproduce] SMOKE mode enabled (set SMOKE=0 for full 41k iters)."
else
    SMOKE_ARG=""
    echo "[reproduce] FULL mode (paper schedule: 41 000 iters + 2 000 NNCG)."
fi

# ---- 4. Train + evaluate, one PDE per loop ---------------------------
ALL_METRICS="$OUTPUT_DIR/metrics.json"
echo "{" > "$ALL_METRICS"
FIRST=1

for PDE in convection reaction wave; do
    RUN_DIR="$OUTPUT_DIR/$PDE"
    mkdir -p "$RUN_DIR"
    echo "[reproduce] === Training PDE: $PDE ==="
    python train.py \
        --config configs/default.yaml \
        --pde "$PDE" \
        --output_dir "$RUN_DIR" \
        $SMOKE_ARG

    echo "[reproduce] === Evaluating PDE: $PDE ==="
    python eval.py \
        --config configs/default.yaml \
        --checkpoint "$RUN_DIR/model.pt" \
        --output_dir "$RUN_DIR"

    if [ $FIRST -eq 0 ]; then echo "," >> "$ALL_METRICS"; fi
    FIRST=0
    echo "  \"$PDE\": $(cat "$RUN_DIR/metrics.json")" >> "$ALL_METRICS"
done

echo "}" >> "$ALL_METRICS"
echo "[reproduce] wrote $ALL_METRICS"
cat "$ALL_METRICS"
