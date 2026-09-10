# Simformer — All-in-one Simulation-Based Inference

This is a faithful PyTorch reproduction of the **Simformer** model proposed in:

> Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, Jakob H. Macke.
> _All-in-one simulation-based inference._ ICML 2024 (PMLR 235).
> arXiv:2404.09636. Official JAX code: <https://github.com/mackelab/simformer>.

## What is implemented

The Simformer is a **probabilistic diffusion model** whose score
`s_φ(x̂_t, t)` is parameterized by a transformer over a sequence of tokens,
one per variable in the joint `x̂ = (θ, x)`. A single trained Simformer can
sample **arbitrary conditionals** of the joint distribution (posterior,
likelihood, or any other), and can incorporate domain knowledge about the
simulator's graphical structure via the transformer's attention mask
`M_E`.

This codebase implements every algorithmic ingredient described in the
paper and clarified in the addendum:

| Paper component                                     | File                                              |
| --------------------------------------------------- | ------------------------------------------------- |
| Tokenizer (id ⊕ value ⊕ cond, §3.1, addendum)       | `model/architecture.py::Tokenizer`                |
| Transformer with diffusion-time injection           | `model/architecture.py::TransformerBlock`         |
| Score model `s_φ^{M_E}`                             | `model/architecture.py::Simformer`                |
| Gaussian Fourier embedding of `t`                   | `model/architecture.py::GaussianFourierEmbedding` |
| VESDE / VPSDE (Song et al. 2021)                    | `model/sde.py`                                    |
| Masked denoising score-matching loss (Eq. 6, 7)     | `model/losses.py::simformer_loss`                 |
| Condition-mask mixture (joint/post/lik/Ber 0.3/0.7) | `model/losses.py::sample_condition_mask`          |
| Reverse-SDE conditional sampler (§3.3)              | `model/sampling.py::sample_conditional`           |
| Universal diffusion guidance (§3.4, Eq. 9)          | `model/sampling.py::guided_sample`                |
| Graph Inversion (Webb et al. 2018, addendum)        | `model/graph_inversion.py`                        |
| Benchmark tasks + structured masks (§4.1, addendum) | `tasks/`                                          |
| Random-Forest C2ST (100 trees, addendum)            | `utils/c2st.py`                                   |

The architecture is summarized in the diagram below
(generated for this README via `image_generate`):

![Simformer architecture](figures/architecture.png)

## Why these design choices

- **Variance Exploding SDE** is the default per §4.1 ("All results are
  obtained using the Variance Exploding SDE (VESDE)"); VPSDE is supported
  via the config.
- **Condition-mask mixture** uses equal weights {joint, posterior,
  likelihood, Ber(0.3), Ber(0.7)} — exactly the addendum specification.
- **Tokenizer** broadcasts the scalar value to the embedding dim (e.g.
  `[v, v, …, v]`) per addendum.md §"Tokenization".
- **Diffusion time** is embedded with a random Gaussian Fourier feature
  and added (via a learnable linear projection) to the output of every FFN
  block — addendum.md §"Training".
- **Optimizer**: AdamW + cosine LR with warm-up, gradient clipping at 1.0
  — these are the canonical defaults for ICML diffusion spotlights and
  match the "default sbi parameters" spirit of the addendum.
- **C2ST**: scikit-learn `RandomForestClassifier(n_estimators=100)` with a
  5-fold cross-validation, exactly as specified in addendum.md.

## Reference verification

The closest baseline that the paper compares against is **NPE** as
implemented in the `sbi` library (Tejero-Cantero et al. 2020). We
verified the paper's metadata on OpenAlex (DOI
`10.48550/arxiv.2404.09636`); CrossRef returned NOT FOUND for that DOI,
which is the expected behaviour for arXiv-only papers (arXiv mints DOIs
but does not deposit them with CrossRef). The OpenAlex record confirms
authors, title, year, and venue.

## Repository layout

```
submission/
├── README.md
├── requirements.txt
├── reproduce.sh                # smoke-quality train + eval entrypoint
├── train.py                    # CLI training driver
├── eval.py                     # CLI evaluation driver (writes metrics.json)
├── configs/default.yaml
├── model/
│   ├── __init__.py
│   ├── architecture.py        # Tokenizer + TransformerBlock + Simformer
│   ├── sde.py                 # VESDE / VPSDE
│   ├── losses.py              # masked DSM loss + mask mixture
│   ├── sampling.py            # reverse-SDE conditional sampler + guidance
│   └── graph_inversion.py     # Webb-2018 algorithm
├── tasks/                     # benchmark simulators (gaussian_linear, two_moons, slcp, …)
├── data/loader.py             # in-memory simulation dataset + normalization
├── utils/c2st.py              # Random-Forest C2ST
└── figures/architecture.png
```

## Running

Quick smoke training + eval (≈ a few minutes on CPU, seconds on a GPU):

```bash
bash reproduce.sh
```

Manual:

```bash
# Train Simformer on Two Moons with 10k simulations
python train.py --config configs/default.yaml --task two_moons --num_simulations 10000 --num_steps 50000

# Evaluate the final checkpoint, writing metrics.json
python eval.py --config configs/default.yaml --checkpoint output/ckpt_final.pt
```

## Notes on the _non-replication_ items (per addendum.md)

The following items are explicitly **not** required for this
replication and are intentionally not implemented in this codebase:

- §4.1 calibration / log-likelihood numbers,
- §4.3 calibration of `Simformer` for SIRD parameters,
- §4.4 additional details and results on guidance (Appendix A3.3).

We do, however, implement the **mechanism** for diffusion guidance
(`model/sampling.py::guided_sample`) so that the Hodgkin-Huxley
energy-interval experiment in §4.4 can be reproduced by an end-user
who supplies a custom constraint function `c(x̂)`.

## Citation

```bibtex
@inproceedings{gloeckler2024simformer,
  title     = {All-in-one simulation-based inference},
  author    = {Gloeckler, Manuel and Deistler, Michael and Weilbach, Christian and Wood, Frank and Macke, Jakob H.},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {235},
  year      = {2024}
}
```
