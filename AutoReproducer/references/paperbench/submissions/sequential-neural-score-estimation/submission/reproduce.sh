#!/usr/bin/env bash
# Reproduction entrypoint for PaperBench Full mode.
# Trains and evaluates TSNPSE on a small SBI benchmark task and writes
# the resulting metrics to /output/metrics.json (the judge reads this file).
#
# Reference: Sharrock, Simons, Liu & Beaumont,
# "Sequential Neural Score Estimation: Likelihood-Free Inference with
#  Conditional Score Based Diffusion Models", ICML 2024.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "${OUTPUT_DIR}"

echo "[reproduce] installing dependencies …"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || true

# Smoke-quality settings: small budget, few rounds, few iters, so this
# completes within reproduction time limits even on CPU. The full paper
# settings are R = 10 rounds × M = 100 simulations / round = 1000 budget
# minimum; we use 4 rounds × 250 sims = 1000 sims (matching the paper's
# smallest budget) for robust reproduction.
TASK="${TASK:-two_moons}"
METHOD="${METHOD:-tsnpse}"
SDE="${SDE:-ve}"
BUDGET="${BUDGET:-1000}"
ROUNDS="${ROUNDS:-4}"
MAX_ITERS="${MAX_ITERS:-300}"

echo "[reproduce] training task=${TASK}  method=${METHOD}  sde=${SDE}  budget=${BUDGET}  rounds=${ROUNDS}"
python train.py \
    --config configs/default.yaml \
    --task "${TASK}" \
    --method "${METHOD}" \
    --sde "${SDE}" \
    --budget "${BUDGET}" \
    --rounds "${ROUNDS}" \
    --max-iters "${MAX_ITERS}" \
    --output-dir "${OUTPUT_DIR}"

echo "[reproduce] evaluating checkpoint …"
python eval.py \
    --checkpoint "${OUTPUT_DIR}/score_net.pt" \
    --output-dir "${OUTPUT_DIR}" \
    --n-samples 1000 \
    --n-ref 1000 \
    --task "${TASK}"

echo "[reproduce] done. metrics → ${OUTPUT_DIR}/metrics.json"
cat "${OUTPUT_DIR}/metrics.json"
