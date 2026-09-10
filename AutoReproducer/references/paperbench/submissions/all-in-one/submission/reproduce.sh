#!/usr/bin/env bash
# Reproduce a smoke-quality Simformer training + eval run.
# Output (metrics.json) is written to /output (PaperBench convention).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. Install dependencies (idempotent).
pip install --no-cache-dir -r requirements.txt || pip install --user -r requirements.txt

# 2. Decide the output directory. PaperBench mounts /output; if it does not
#    exist (e.g. local debugging), fall back to ./output.
OUTDIR="/output"
if [ ! -d "$OUTDIR" ] || [ ! -w "$OUTDIR" ]; then
    OUTDIR="$SCRIPT_DIR/output"
    mkdir -p "$OUTDIR"
fi
echo "[reproduce] Writing artifacts to $OUTDIR"

# 3. Smoke-quality Simformer training on the Two Moons benchmark.
#    The full paper used ~50k steps; we use a much smaller budget so the
#    24-hour PaperBench reproduction container completes well within budget.
python train.py \
    --config configs/default.yaml \
    --task two_moons \
    --num_simulations 5000 \
    --num_steps 2000 \
    --output_dir "$OUTDIR"

# 4. Evaluate.
python eval.py \
    --config configs/default.yaml \
    --checkpoint "$OUTDIR/ckpt_final.pt" \
    --output_dir "$OUTDIR" \
    --num_eval_obs 5

# 5. Sanity-check the metrics file exists.
if [ ! -f "$OUTDIR/metrics.json" ]; then
    echo "[reproduce] ERROR: metrics.json missing!" 1>&2
    exit 1
fi
echo "[reproduce] Done. metrics.json:"
cat "$OUTDIR/metrics.json"
