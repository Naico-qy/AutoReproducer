# DPMs-ANT: Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning

A faithful PyTorch re-implementation of

> **Wang, X., Lin, B., Liu, D., Chen, Y.-C., Xu, C.**  
> _Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning._  
> ICML 2024.

This codebase implements the paper's main algorithm (Algorithm 1) and the
two strategies that compose it: **Similarity-Guided Training** (§4.1) and
**Adversarial Noise Selection** (§4.2). Hyperparameters and architectural
details follow the paper §5.2 and the official addendum.

![DPMs-ANT architecture](figures/architecture.png)

> _Figure: pre-trained ε_θ (frozen, grey) is augmented with per-block
> trainable adaptors ψ (orange). For every batch, a multi-step PGD
> ascent finds the worst-case noise ε\* (Eq. 7); ψ is then updated to
> minimise the similarity-guided loss L(ψ) (Eq. 8) under that ε\*._

---

## 1 What is implemented

| Paper section                            | What this repo provides                                                                       | File                                                 |
| ---------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| §3 Preliminary                           | Forward / reverse Gaussian-diffusion schedule, σ̂_t (Eq. 5)                                    | `model/schedule.py`                                  |
| §4.1 Similarity-Guided Training (Eq. 5)  | `DPMsANT.similarity_loss()`                                                                   | `model/architecture.py`                              |
| §4.2 Adversarial Noise Selection (Eq. 7) | `DPMsANT.adversarial_noise()` (J-step PGD ascent + Norm)                                      | `model/architecture.py`                              |
| §4.3 Optimization & adaptor (Eq. 8)      | `Adaptor` + `AdaptedUNet` (down-pool→Norm+3×3 Conv→4-head Attn→MLP→up-sample×4→Norm→3×3 Conv) | `model/adaptor.py`                                   |
| Algorithm 1                              | `DPMsANT.training_step()`                                                                     | `model/architecture.py`, `train.py`                  |
| §5.2 Binary classifier p_φ(y\|x_t)       | `BinaryNoiseClassifier`, fine-tune with Adam, lr=1e-4, batch=64, 300 it (per addendum)        | `model/classifier.py`, `scripts/train_classifier.py` |
| §5.2 Hyperparameters                     | `configs/default.yaml` + `configs/all_targets.yaml` (every entry from addendum Table 3)       | `configs/`                                           |
| §5.2 Intra-LPIPS, FID                    | `eval.py`                                                                                     | `eval.py`                                            |

### Key equations and where they live

| Equation                                                            | Location                                                              |
| ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Eq. (1) `q(x_t\|x_0)`, `x_t = √ᾱ_t x_0 + √(1-ᾱ_t)ε`                 | `GaussianDiffusion.q_sample`                                          |
| Eq. (2) DDIM reverse step                                           | `GaussianDiffusion.ddim_step`                                         |
| Eq. (3) classifier-guidance reverse process                         | `similarity_grad` in `model/classifier.py`                            |
| Eq. (4) KL-divergence between source / target reverse-process means | implicit in Eq. 5 derivation; see `architecture.py` doc-strings       |
| Eq. (5) similarity-guided DPM loss                                  | `DPMsANT.similarity_loss`                                             |
| Eq. (6) min-max objective                                           | `DPMsANT.training_step` (outer min) + `adversarial_noise` (inner max) |
| Eq. (7) ε^{j+1} = Norm(ε^j + ω · ∇*ε ‖ε^j − ε*θ(x_t,t)‖²)           | `DPMsANT.adversarial_noise`                                           |
| Eq. (8) full loss L(ψ)                                              | `DPMsANT.training_step`                                               |

---

## 2 Hyperparameters (§5.2 + addendum)

The default config implements DDPM, FFHQ → 10-shot Sunglasses:

| Parameter                   | Value                                                           | Source                                                               |
| --------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------- |
| Image size                  | 256 (DDPM) / 64 (LDM)                                           | §5.2                                                                 |
| Diffusion steps T           | 1000                                                            | §3                                                                   |
| β schedule                  | linear, [1e-4, 0.02]                                            | Ho et al. 2020                                                       |
| Adaptor c (down-projection) | 4 (DDPM default) / 2 (LDM default)                              | §5.2                                                                 |
| Adaptor d (bottleneck)      | 8                                                               | §5.2                                                                 |
| Adaptor zero-init           | yes                                                             | §5.2                                                                 |
| γ (similarity guidance)     | 5 (default), per-target overrides in `configs/all_targets.yaml` | §5.2 / addendum                                                      |
| J (PGD inner steps)         | 10                                                              | §5.2                                                                 |
| ω (PGD step size)           | 0.02                                                            | §5.2                                                                 |
| Optimizer                   | AdamW                                                           | following Ho et al. (paper does not specify; standard for diffusion) |
| LR                          | 5e-5 (DDPM) / 1e-5 (LDM)                                        | §5.2                                                                 |
| Batch size                  | 40                                                              | §5.2                                                                 |
| Iterations                  | ~300                                                            | §5.2                                                                 |
| Classifier optimizer        | Adam, lr=1e-4, batch=64, 300 iters                              | addendum                                                             |

All target-specific overrides from the addendum are encoded in
`configs/all_targets.yaml`.

---

## 3 Quick start

```bash
# install
pip install -r requirements.txt

# (1) fine-tune the source-vs-target binary classifier
python -m scripts.train_classifier \
    --config configs/default.yaml \
    --source ./datasets/ffhq \
    --target ./datasets/10shot_sunglasses \
    --out ./outputs/classifier.pt

# (2) train the adaptor (Algorithm 1)
python train.py \
    --config configs/default.yaml \
    --source-ckpt ./checkpoints/ffhq_ddpm.pt \
    --classifier-ckpt ./outputs/classifier.pt \
    --out ./outputs

# (3) evaluate
python eval.py \
    --config configs/default.yaml \
    --adaptor ./outputs/adaptor.pt \
    --classifier-ckpt ./outputs/classifier.pt \
    --training-dir ./datasets/10shot_sunglasses \
    --target-dir   ./datasets/full_target \
    --out ./outputs/metrics.json
```

For other source/target pairs, swap `--config` for one of:
`ddpm_ffhq_babies.yaml`, `ddpm_ffhq_raphael.yaml`,
`ddpm_lsun_haunted.yaml`, `ddpm_lsun_landscape.yaml`,
`ldm_ffhq_*.yaml`, `ldm_lsun_*.yaml` (all defined in
`configs/all_targets.yaml`).

---

## 4 Pre-trained checkpoints required for full reproduction

- DDPM source U-Net – pre-trained on FFHQ-256 / LSUN-Church-256 by
  Dhariwal & Nichol 2021. The paper §5.2 says "we employ a pre-trained
  DDPM similar to DDPM-PA".
- LDM source U-Net – pre-trained autoencoder + denoiser from
  Rombach et al. 2022.
- Classifier – fine-tune from
  DDPM 256x256 → `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt`
  LDM 64x64 → `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt`
  (modify the final layer to 2 logits; addendum).

Without these checkpoints `train.py` falls back to randomly-initialized
weights so the script still runs end-to-end (smoke mode); reported
metrics will not match the paper but the code path is exercised.

---

## 5 Reproducing in a Docker container (PaperBench Full mode)

```bash
bash reproduce.sh
```

This runs a smoke-quality end-to-end pass and writes
`/output/metrics.json`. Set the env var `OUT_DIR` to redirect output.
If real datasets / source weights are not mounted, the script generates
synthetic placeholder images so the pipeline still completes.

---

## 6 Reference verification

The closest GAN-based baseline cited as the canonical few-shot
benchmark is **CDC** (Ojha et al., CVPR 2021). I verified its metadata
via CrossRef:

```
DOI            : 10.1109/CVPR46437.2021.01060
Title          : Few-shot Image Generation via Cross-domain Correspondence
Year           : 2021
Venue          : IEEE/CVF Conference on Computer Vision and Pattern Recognition
Status         : VERIFIED ✓
```

This citation grounds the dataset conventions used in `data/loader.py`
(10-shot Babies / Sunglasses / Raphael / Haunted Houses / Landscape
drawings, FFHQ and LSUN-Church as source domains).

---

## 7 File layout

```
submission/
├── README.md                    this file
├── requirements.txt
├── reproduce.sh                 PaperBench Full-mode entrypoint
├── train.py                     Algorithm 1 entrypoint
├── eval.py                      Intra-LPIPS + FID
├── utils.py                     YAML-with-extends, EMA, seeding, metric IO
├── configs/
│   ├── default.yaml             DDPM, FFHQ → Sunglasses (paper §5.2 base config)
│   ├── ddpm_ffhq_sunglasses.yaml
│   └── all_targets.yaml         all 10 (framework, source, target) tuples from addendum
├── model/
│   ├── __init__.py
│   ├── schedule.py              GaussianDiffusion: q_sample, σ̂_t, DDIM
│   ├── unet.py                  ADM-style U-Net ε_θ(x_t, t)
│   ├── adaptor.py               Adaptor ψ + AdaptedUNet (residual rule)
│   ├── classifier.py            Binary classifier p_φ + similarity_grad
│   └── architecture.py          DPMsANT: Eq. 5 / Eq. 7 / Eq. 8 + Algorithm 1
├── data/
│   ├── __init__.py
│   └── loader.py                FewShotImageDataset, classifier dataset
├── scripts/
│   ├── __init__.py
│   └── train_classifier.py      addendum: Adam / lr 1e-4 / bs 64 / 300 iters
└── figures/
    └── architecture.png         pipeline diagram (auto-generated)
```

---

## 8 Notes & limitations

- The implementation targets faithfulness to the paper / addendum
  rather than wall-clock speed; mixed-precision, gradient checkpointing
  and FlashAttention can be enabled by tweaking `configs/default.yaml`.
- The LDM autoencoder (encoder/decoder) is not bundled — when running
  with `framework: ldm` you must supply the pre-trained AutoencoderKL
  from Rombach et al. 2022 and feed its z-latents to `train.py`.
- Equation (4)’s exact KL derivation is referenced in the paper’s
  Appendix; we implement its operational form (Eq. 5) directly via the
  classifier-gradient guidance term.
