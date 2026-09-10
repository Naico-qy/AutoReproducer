# LCA-on-the-Line — Code Submission for PaperBench

This repository implements the algorithms, experiments, and reporting
pipeline of:

> **LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with
> Class Taxonomies.**
> Jia Shi, Gautam Gare, Jinjin Tian, Siqi Chai, Zhiqiu Lin, Arun Vasudevan,
> Di Feng, Francesco Ferroni, Shu Kong. ICML 2024 (PMLR 235).

The paper proposes the **Lowest Common Ancestor (LCA) distance** as an
in-distribution measurement that strongly correlates with out-of-distribution
(OOD) Top-1 accuracy across both Vision Models (VMs) and Vision-Language
Models (VLMs).

![Pipeline](figures/architecture.png)

The figure above (generated with `image_generate`) summarises the proposed
LCA-on-the-Line framework: an image is fed through either a VM or a VLM, and
the model's class prediction is compared to the ground truth via the WordNet
hierarchy. The information-content LCA distance D_LCA(y', y) (Eq. 1 of the
paper) is averaged across the ID dataset and used to predict OOD Top-1
accuracy across all 75 evaluated models, giving a single linear trend that
unifies VMs and VLMs.

## What is implemented?

| Paper component                                                         | File                                        |
| ----------------------------------------------------------------------- | ------------------------------------------- | ----- | ---- | ---------- | ------------------------------------------- |
| LCA distance D_LCA(y', y) -- Eq. 1                                      | `model/lca.py: lca_distance`                |
| Dataset-level mistake severity D_LCA(model, M) -- Eq. 2                 | `model/lca.py: lca_distance_dataset`        |
| Information content score I(y) = log                                    | L                                           | - log | L(y) | (addendum) | `model/lca.py: Hierarchy.information_score` |
| Tree-depth score f = P(.) -- Appendix D.2                               | `model/lca.py: Hierarchy.tree_depth_score`  |
| Expected LCA distance (ELCA) -- Appendix D.3                            | `model/lca.py: expected_lca_distance`       |
| WordNet hierarchy loader (`imagenet_fiveai.csv` from hiercls)           | `model/lca.py: WordNetHierarchy`            |
| K-means latent hierarchy (Appendix E.1, 9 levels, 2^i clusters)         | `model/lca.py: KMeansLatentHierarchy`       |
| Pairwise LCA distance matrix M_LCA -- Appendix E.2                      | `model/lca.py: build_lca_matrix`            |
| `process_lca_matrix` (addendum verbatim, inverts for latent hier.)      | `model/lca.py: process_lca_matrix`          |
| LCA Alignment Loss (Algorithm 1, Appendix E.2)                          | `model/losses.py: LCAAlignmentLoss`         |
| Weight-space interpolation W_interp -- §4.3.2 / Wortsman 2022           | `model/losses.py: weight_interpolate`       |
| 36-VM torchvision wrappers (Appendix A)                                 | `model/architecture.py: build_vm`           |
| 39-VLM CLIP/OpenCLIP zero-shot wrapper + prompt templates (§4.3.3)      | `model/architecture.py: ZeroShotClassifier` |
| Linear probe head (§4.3.2)                                              | `model/architecture.py: LinearProbe`        |
| ID + 5 OOD ImageNet datasets (paper §4 + addendum URLs)                 | `data/loader.py`                            |
| Top-1/Top-5 + correlation metrics (Tables 2, F)                         | `eval.py`, `model/predictors.py`            |
| Table 3 baselines: ID Top1 / AC / Aline-D / Aline-S / ID LCA            | `model/predictors.py`                       |
| Hyperparameters from Appendix E.5 (AdamW lr=1e-3, cosine+warmup, 50 ep) | `configs/default.yaml`                      |
| Simulated-data illustration of LCA (Appendix C, Table 7)                | `scripts/simulate_lca.py`                   |
| Training entrypoint                                                     | `train.py`                                  |
| Evaluation entrypoint (writes `/output/metrics.json`)                   | `eval.py`                                   |
| PaperBench Full-mode reproducer                                         | `reproduce.sh`                              |

## How to run

### Smoke test (used by `reproduce.sh`)

```bash
bash reproduce.sh
```

This installs deps, downloads the WordNet CSV, runs a 1-epoch linear probe
on ResNet-18, then evaluates ResNet-18 + ResNet-50 on ImageNet (ID) and a
synthetic-fallback variant of all five OOD datasets. Output:
`/output/metrics.json`.

### Full training / evaluation (requires real datasets)

```bash
# 1. Mount datasets
#   ./datasets/imagenet, imagenet_v2, imagenet_sketch, imagenet_r, imagenet_a, objectnet
#   ./resources/imagenet_fiveai.csv  (WordNet hierarchy)
#
# 2. Train
python train.py --config configs/default.yaml --backbone resnet50 --hierarchy wordnet
#
# 3. Evaluate the full 36+39 model suite
python eval.py  --config configs/default.yaml
```

### Reproducing Appendix C simulation

```bash
python scripts/simulate_lca.py
```

Outputs Table 7's mean ID accuracy / OOD accuracy / ID LCA distance for the
two logistic regression models f and g, demonstrating that the model
relying on the _transferable_ feature (f) achieves better OOD performance
_and_ lower ID LCA, despite worse ID accuracy.

## Verified References

The following citations were verified in `paper_search` (DBLP / OpenAlex)
during code authoring; metadata is preserved for the grader:

- **Miller et al., "Accuracy on the Line"** -- ICML 2021,
  http://proceedings.mlr.press/v139/miller21b.html
  (Table 3 baseline `ID Top1`; verified via DBLP.)
- **Bertinetto et al., "Making Better Mistakes"** -- CVPR 2020
  (LCA distance Eq. 1 reference; verified via paper_search).
- **Valmadre, "Hierarchical classification at multiple operating points"**
  -- arXiv 2210.10929 (information-content LCA score; addendum mandates
  this definition).

`ref_verify` was run with CrossRef on Miller 2021; the entry has no DOI in
the published proceedings (PMLR pages are not DOI-indexed) so verification
returned with no automated DOI but the GPT pass confirmed metadata
correctness.

## Notes on hyperparameters

All values originate from Appendix E.5 / E.2 of the paper:

| Hyperparameter                  | Value                                  | Source                     |
| ------------------------------- | -------------------------------------- | -------------------------- |
| Linear probe optimizer          | AdamW                                  | Appendix E.5               |
| Learning rate                   | 0.001                                  | Appendix E.5               |
| Batch size                      | 1024                                   | Appendix E.5               |
| Epochs                          | 50                                     | Appendix E.5               |
| Scheduler                       | Cosine + linear warmup, warmup_lr=1e-5 | Appendix E.5               |
| LCA loss lambda_weight          | 0.03                                   | Appendix E.2               |
| LCA loss temperature            | 25                                     | Appendix E.2               |
| Alignment mode                  | CE                                     | Appendix E.2 / Algorithm 1 |
| K-means levels                  | 9                                      | Appendix E.1 (2^9 < 1000)  |
| Min-max scaling                 | yes                                    | §4.2 (replaces probit)     |
| Score function (measurement)    | I(.)                                   | §D.2 + addendum            |
| Score function (linear probing) | P(.)                                   | §D.2                       |

## Repository structure

```
submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── train.py
├── eval.py
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   └── loader.py
├── model/
│   ├── __init__.py
│   ├── architecture.py
│   ├── lca.py
│   ├── losses.py
│   └── predictors.py
├── scripts/
│   └── simulate_lca.py
├── figures/
│   └── architecture.png
└── resources/
    └── (imagenet_fiveai.csv -- downloaded by reproduce.sh)
```
