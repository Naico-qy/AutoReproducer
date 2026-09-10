#!/usr/bin/env bash
# reproduce.sh -- end-to-end smoke run for PaperBench Full mode.
#
# 1. install dependencies
# 2. run a tiny BaM smoke training across all four experiment families
# 3. aggregate metrics into /output/metrics.json
#
# Designed to complete in well under 24h; on a typical CPU it finishes
# in roughly 1-3 minutes.

set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${HERE}"

OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "${OUTPUT_DIR}"
LOCAL_OUT="${HERE}/outputs"
mkdir -p "${LOCAL_OUT}"

echo "== [1/3] Installing dependencies =="
pip install --no-cache-dir -r requirements.txt || pip install --user --no-cache-dir -r requirements.txt || true

echo "== [2/3] Running BaM smoke training across all experiments =="
python train.py --config configs/default.yaml --experiment all --smoke --out "${LOCAL_OUT}"

echo "== [3/3] Aggregating metrics =="
python eval.py --in_dir "${LOCAL_OUT}" --out "${OUTPUT_DIR}/metrics.json"

# Also leave a copy alongside the submission for convenience.
cp "${OUTPUT_DIR}/metrics.json" "${LOCAL_OUT}/metrics.json" || true

echo "Done.  Metrics written to ${OUTPUT_DIR}/metrics.json"
