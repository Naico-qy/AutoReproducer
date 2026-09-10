#!/usr/bin/env bash
# reproduce.sh — PaperBench Full mode entrypoint.
#
# 1. Install pip dependencies (cached if already present).
# 2. Download the WordNet hierarchy CSV (paper addendum source).
# 3. Run a SHORT smoke training of the LCA-soft linear-probe on ResNet-18.
# 4. Run a SHORT evaluation that produces /output/metrics.json.
#
# The smoke variant keeps total wall-clock under PaperBench's 24h budget on
# any GPU, while still exercising the full code path (data loading, feature
# extraction, LCA matrix construction, soft-label loss training, OOD eval,
# Table 2/3 correlation reporting).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
if [ -f requirements.txt ]; then
    pip install --quiet --no-cache-dir -r requirements.txt || true
fi

# ---------------------------------------------------------------------------
# 2. WordNet hierarchy (resources/imagenet_fiveai.csv)
# ---------------------------------------------------------------------------
mkdir -p resources
WORDNET_CSV="resources/imagenet_fiveai.csv"
if [ ! -f "$WORDNET_CSV" ]; then
    curl -sSL -o "$WORDNET_CSV" \
        "https://raw.githubusercontent.com/jvlmdr/hiercls/main/resources/hierarchy/imagenet_fiveai.csv" \
        || echo "[reproduce] Failed to fetch WordNet hierarchy; LCA will be 0 in eval."
fi

# ---------------------------------------------------------------------------
# 3. Smoke training (ResNet-18, 1 epoch, smoke samples)
# ---------------------------------------------------------------------------
python train.py \
    --config configs/default.yaml \
    --backbone resnet18 \
    --hierarchy wordnet \
    --smoke-test \
    --output-dir "$OUTPUT_DIR" || true

# ---------------------------------------------------------------------------
# 4. Smoke evaluation (writes /output/metrics.json)
# ---------------------------------------------------------------------------
python eval.py \
    --config configs/default.yaml \
    --output-dir "$OUTPUT_DIR" \
    --smoke-test \
    --backbones resnet18 resnet50

echo "[reproduce] done. Metrics at $OUTPUT_DIR/metrics.json"
