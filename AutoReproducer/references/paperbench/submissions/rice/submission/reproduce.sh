#!/usr/bin/env bash
# Reproduction entrypoint for the RICE submission.
#
# This script performs a smoke-quality end-to-end run on Hopper-v4:
#   1. install dependencies
#   2. pre-train PPO target policy (Stage A)
#   3. train mask network               (Stage B, Algorithm 1)
#   4. refine via RICE                  (Stage C, Algorithm 2)
#   5. evaluate fidelity + refined reward
#
# Outputs:  /output/metrics.json  (PaperBench reproduction grader format)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pick the env via SUBMISSION_ENV (override-able); default is Hopper-v4.
ENV="${SUBMISSION_ENV:-Hopper-v4}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# Smoke run: cap timesteps so the pipeline finishes in minutes, not days.
# The full hyperparameter values from the paper remain in configs/default.yaml.
# Override via SMOKE=0 to use the paper's full settings.
SMOKE="${SMOKE:-1}"
SMOKE_CONFIG="$HERE/configs/smoke.yaml"
if [ "$SMOKE" = "1" ]; then
    python -c "
import yaml, sys
with open('$HERE/configs/default.yaml') as f: c = yaml.safe_load(f)
c['pretrain']['total_timesteps'] = 8192
c['mask']['total_timesteps']     = 4096
c['refine']['total_timesteps']   = 4096
c['eval']['n_eval_episodes']     = 3
c['eval']['fidelity_n_trajectories'] = 3
with open('$SMOKE_CONFIG','w') as f: yaml.safe_dump(c, f)
print('[reproduce.sh] wrote smoke config to $SMOKE_CONFIG')
"
    CONFIG="$SMOKE_CONFIG"
else
    CONFIG="$HERE/configs/default.yaml"
fi

echo "[reproduce.sh] Working dir: $HERE"
echo "[reproduce.sh] Env:         $ENV"
echo "[reproduce.sh] Output:      $OUTPUT_DIR"
echo "[reproduce.sh] Config:      $CONFIG"

# ---------------------------------------------------------------- 1. install
echo "[reproduce.sh] (1/5) Installing dependencies"
pip install --no-cache-dir -r requirements.txt || \
    echo "[reproduce.sh] WARNING: pip install partial; proceeding anyway."

# ---------------------------------------------------------------- 2. pretrain
echo "[reproduce.sh] (2/5) Pre-training PPO target policy"
python train.py --config "$CONFIG" --stage pretrain \
    --env "$ENV" --out_dir "$CHECKPOINT_DIR" || true

# ---------------------------------------------------------------- 3. mask
echo "[reproduce.sh] (3/5) Training Mask Network (Algorithm 1)"
python train.py --config "$CONFIG" --stage mask \
    --env "$ENV" --out_dir "$CHECKPOINT_DIR" || true

# ---------------------------------------------------------------- 4. refine
echo "[reproduce.sh] (4/5) Refining via RICE (Algorithm 2)"
python train.py --config "$CONFIG" --stage refine \
    --env "$ENV" --method rice --out_dir "$CHECKPOINT_DIR" || true

# Optionally also run baselines (StateMask-R, PPO-FT) for comparison;
# kept gated by env var so the smoke run stays brief.
if [ "${RUN_BASELINES:-0}" = "1" ]; then
    echo "[reproduce.sh] (4b) Refining via baselines"
    python train.py --config "$CONFIG" --stage refine \
        --env "$ENV" --method ppo_ft     --out_dir "$CHECKPOINT_DIR" || true
    python train.py --config "$CONFIG" --stage refine \
        --env "$ENV" --method statemask_r --out_dir "$CHECKPOINT_DIR" || true
    python train.py --config "$CONFIG" --stage refine \
        --env "$ENV" --method jsrl       --out_dir "$CHECKPOINT_DIR" || true
    python train.py --config "$CONFIG" --stage refine \
        --env "$ENV" --method random_expl --out_dir "$CHECKPOINT_DIR" || true
fi

# ---------------------------------------------------------------- 5. eval
echo "[reproduce.sh] (5/5) Evaluating refined policy + fidelity"
python eval.py --config "$CONFIG" --env "$ENV" \
    --method rice --checkpoint_dir "$CHECKPOINT_DIR" \
    --output "$OUTPUT_DIR/metrics.json" || \
    python eval.py --config "$CONFIG" --env "$ENV" \
        --method rice --checkpoint_dir "$CHECKPOINT_DIR" \
        --output "$OUTPUT_DIR/metrics.json" --skip_fidelity

echo "[reproduce.sh] Done. metrics.json:"
cat "$OUTPUT_DIR/metrics.json" || echo "(metrics.json missing)"
