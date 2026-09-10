#!/usr/bin/env bash
# ----------------------------------------------------------------------
# BBox-Adapter — PaperBench reproduction entrypoint.
#
# This script (i) installs dependencies, (ii) runs a smoke-quality
# online-adaptation training loop on synthetic StrategyQA-style data
# (the Dummy LLM backend, so no API key is required), and
# (iii) evaluates the resulting adapter, writing /output/metrics.json.
#
# A full reproduction (e.g. Table 2) needs a real OpenAI / Mixtral
# backend — set OPENAI_API_KEY and use:
#
#   python train.py --config configs/default.yaml --task gsm8k \
#                   --feedback combined --llm-backend openai
#
# Within the 24-hour PaperBench budget the dummy run is what we
# default to here.
# ----------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

echo "[reproduce] === BBox-Adapter reproduction ==="
echo "[reproduce] python: $(python --version 2>&1)"
echo "[reproduce] pwd: $(pwd)"

# -------------------------- 1. Install deps --------------------------
echo "[reproduce] installing requirements..."
python -m pip install --upgrade pip >/dev/null 2>&1 || true
python -m pip install -r requirements.txt || \
    echo "[reproduce] WARN: some packages failed to install; continuing"

# -------------------------- 2. Output dir ----------------------------
OUT_DIR="${OUTPUT_DIR:-/output}"
if ! mkdir -p "$OUT_DIR" 2>/dev/null; then
    OUT_DIR="./outputs"
    mkdir -p "$OUT_DIR"
fi
echo "[reproduce] output dir = $OUT_DIR"

# -------------------------- 3. Train --------------------------------
echo "[reproduce] === training (smoke run) ==="
python train.py \
    --config configs/default.yaml \
    --task strategyqa \
    --feedback combined \
    --llm-backend dummy \
    --output-dir "$OUT_DIR" \
    --smoke || {
        echo "[reproduce] ERROR: training failed"; exit 1;
    }

# -------------------------- 4. Evaluate ------------------------------
echo "[reproduce] === evaluation ==="
python eval.py \
    --config configs/default.yaml \
    --task strategyqa \
    --llm-backend dummy \
    --output-dir "$OUT_DIR" \
    --smoke || {
        echo "[reproduce] ERROR: evaluation failed"; exit 1;
    }

echo "[reproduce] === done ==="
echo "[reproduce] metrics file:"
cat "$OUT_DIR/metrics.json" || true
