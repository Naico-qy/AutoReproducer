#!/usr/bin/env bash
# PaperBench reproduction-mode entrypoint for
#   "Stochastic Interpolants with Data-Dependent Couplings" (ICML 2024).
#
# Behaviour:
#   1. Install pip dependencies.
#   2. Run a (possibly short) training pass for both tasks.
#   3. Run evaluation and write metrics to /output/metrics.json.
#
# Environment variables:
#   STEPS         number of gradient steps per task         (default: 200)
#   TASK          which task to run: superres|inpainting|both (default: both)
#   OUTPUT_DIR    where to write metrics                    (default: /output)
#   DEBUG         "1" → use synthetic data (works without ImageNet)
#                 (default: 1 — keep the smoke test self-contained)
#
# For a faithful 24-h GPU run set STEPS=200000 and DEBUG=0 (which will
# trigger HuggingFace ImageNet-1k download via `datasets`).

set -euo pipefail

STEPS="${STEPS:-200}"
TASK="${TASK:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
DEBUG="${DEBUG:-1}"

cd "$(dirname "$0")"

mkdir -p "$OUTPUT_DIR"

echo "[reproduce] installing dependencies"
python -m pip install --no-cache-dir -r requirements.txt || \
    python -m pip install --no-cache-dir -r requirements.txt --user

DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
  DEBUG_FLAG="--debug"
  echo "[reproduce] DEBUG=1 -> using synthetic dataset"
fi

run_task() {
    local task="$1"
    local cfg="configs/${task}.yaml"
    local run_dir="runs/${task}"
    mkdir -p "$run_dir"

    echo "[reproduce] === training ${task} for ${STEPS} steps ==="
    python train.py \
        --config "$cfg" \
        --task "$task" \
        --steps "$STEPS" \
        --output "$run_dir" \
        $DEBUG_FLAG

    echo "[reproduce] === evaluating ${task} ==="
    python eval.py \
        --config "$cfg" \
        --checkpoint "${run_dir}/last.pt" \
        --output "${OUTPUT_DIR}/metrics_${task}.json" \
        $DEBUG_FLAG
}

if [ "$TASK" = "both" ]; then
    run_task "superres"
    run_task "inpainting"
else
    run_task "$TASK"
fi

# Aggregate the two metric files into a single /output/metrics.json
python - <<'PY'
import json, os, glob
out_dir = os.environ.get("OUTPUT_DIR", "/output")
agg = {}
for f in sorted(glob.glob(os.path.join(out_dir, "metrics_*.json"))):
    key = os.path.basename(f).replace("metrics_", "").replace(".json", "")
    with open(f) as fh:
        agg[key] = json.load(fh)
with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
    json.dump(agg, fh, indent=2)
print(f"[reproduce] aggregated metrics -> {os.path.join(out_dir, 'metrics.json')}")
PY

echo "[reproduce] done."
