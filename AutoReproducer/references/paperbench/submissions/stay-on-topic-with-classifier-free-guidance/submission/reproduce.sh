#!/usr/bin/env bash
# Convenience driver -- reproduces every IN-SCOPE experiment from
# "Stay on topic with Classifier-Free Guidance" (Sanchez et al., ICML 2024).
#
# Sections out of scope (per addendum) are skipped:
#   * §3.4 human study
#   * §4 VRAM analysis (kept §4.1 FLOPs/ANCOVA)
#   * Appendix D.1 GPT-J / CodeGen toy experiments
#   * LLaMA models
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results

echo "=========================================="
echo " §3.1 -- Zero-shot benchmarks (Table 5)"
echo "=========================================="
python eval.py --config configs/zero_shot.yaml --task zero_shot \
    2>&1 | tee logs/zero_shot.log

echo "=========================================="
echo " §3.2 -- Chain-of-Thought (Figure 2/17)"
echo "=========================================="
python eval.py --config configs/chain_of_thought.yaml --task cot \
    2>&1 | tee logs/cot.log

echo "=========================================="
echo " §3.3.1 -- HumanEval pass@k (Tables 2/3, Fig. 3)"
echo "=========================================="
python eval.py --config configs/humaneval.yaml --task humaneval \
    2>&1 | tee logs/humaneval.log

echo "=========================================="
echo " §4.1 -- FLOPs / ANCOVA cost analysis (Table 6)"
echo "=========================================="
python eval.py --config configs/flops.yaml --task flops \
    2>&1 | tee logs/flops.log

echo "=========================================="
echo " §5.1 -- Sampling-entropy comparison"
echo "=========================================="
python eval.py --config configs/entropy.yaml --task entropy \
    2>&1 | tee logs/entropy.log

echo "=========================================="
echo " §5.2 -- CFG vs. Instruction-Tuning PPL correlation (Fig. 5)"
echo "=========================================="
python eval.py --config configs/entropy.yaml --task ppl_corr \
    2>&1 | tee logs/ppl_corr.log

echo "=========================================="
echo " §5.3 -- Token re-ranking visualization (Table 3)"
echo "=========================================="
python eval.py --config configs/entropy.yaml --task visualize \
    2>&1 | tee logs/visualize.log

echo "All in-scope experiments completed.  Results -> ./results/"
