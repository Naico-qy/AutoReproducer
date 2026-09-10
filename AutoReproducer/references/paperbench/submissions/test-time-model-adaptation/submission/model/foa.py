"""Forward-Optimization Adaptation (FOA): the main algorithm.

This module implements Algorithm 1 of:

    Niu et al., "Test-Time Model Adaptation with Only Forward Passes",
    ICML 2024.

Pipeline per test batch X_t:
    1. Sample K candidate prompts {p_k} ~ m + tau * N(0, Sigma)        (Eqn. 6)
    2. For each k, forward(X_t with p_k) -> per-layer CLS, logits
    3. Compute fitness v_k per Eqn. (5):
         L = sum_x sum_c -y_hat_c log y_hat_c
             + lambda * sum_i ||mu_i(X_t) - mu_i^S||_2 + ||sigma_i(X_t) - sigma_i^S||_2
    4. Tell CMA-ES the fitness values; it updates m, tau, Sigma
    5. Pick prediction y_hat_t with the best v_k as the final batch prediction
    6. Apply back-to-source activation shifting on top of the final CLS feat
       (Section 3.2; we apply it inside the candidate evaluation as well so the
        fitness reflects the deployed prediction path)

CMA-ES backend: per addendum.md, the original implementation uses the
``cmaes`` Python package (https://github.com/CyberAgentAILab/cmaes).  We use
that library if available; otherwise we fall back to a minimal
sigma-step ES that is sufficient for the smoke test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .activation_shift import ActivationShifter
from .architecture import PromptedViT


# ----------------------------------------------------------------------
# CMA-ES wrapper
# ----------------------------------------------------------------------
class _CMAES:
    """Thin wrapper that uses the `cmaes` package if available, with a tiny
    fallback ES for environments where that package is not installed.

    Per the paper, K = 28 = 4 + 3 * log(prompt_dim) following Hansen (2016).
    The mean is initialized to 0 and the covariance to I (Algorithm 1 line:
    "Initialize m^(0)=0, Sigma^(0)=I, tau^(0)=1 in Eqn. (6)").
    """

    def __init__(self, dim: int, popsize: int, sigma0: float = 1.0, seed: int = 0):
        self.dim = int(dim)
        self.popsize = int(popsize)
        self.sigma0 = float(sigma0)
        self.seed = int(seed)
        self._impl_kind: str
        try:
            from cmaes import CMA  # type: ignore

            self._impl = CMA(
                mean=np.zeros(self.dim, dtype=np.float64),
                sigma=self.sigma0,
                population_size=self.popsize,
                seed=self.seed,
            )
            self._impl_kind = "cmaes"
        except Exception:  # pragma: no cover - fallback path
            self._impl = None
            self._rng = np.random.default_rng(self.seed)
            self._mean = np.zeros(self.dim, dtype=np.float64)
            self._sigma = self.sigma0
            self._impl_kind = "fallback"

    def ask(self) -> List[np.ndarray]:
        if self._impl_kind == "cmaes":
            return [self._impl.ask() for _ in range(self.popsize)]
        # fallback: isotropic Gaussian around current mean
        return [
            self._mean + self._sigma * self._rng.standard_normal(self.dim)
            for _ in range(self.popsize)
        ]

    def tell(self, solutions: List[np.ndarray], values: List[float]) -> None:
        if self._impl_kind == "cmaes":
            self._impl.tell(list(zip(solutions, values)))
            return
        # fallback: update mean toward best half (truncated mean ES)
        order = np.argsort(values)  # minimize fitness
        elites = [solutions[i] for i in order[: max(1, self.popsize // 2)]]
        new_mean = np.mean(np.stack(elites, axis=0), axis=0)
        diff = np.linalg.norm(new_mean - self._mean) + 1e-9
        self._mean = new_mean
        # adaptive sigma: shrink slowly
        self._sigma *= math.exp(
            0.1 * (np.log(diff + 1e-9) - np.log(self._sigma + 1e-9))
        )
        self._sigma = float(np.clip(self._sigma, 1e-3, 5.0))

    @property
    def mean(self) -> np.ndarray:
        if self._impl_kind == "cmaes":
            return self._impl.mean.copy()
        return self._mean.copy()


# ----------------------------------------------------------------------
# Source-statistics helper
# ----------------------------------------------------------------------
@dataclass
class SourceStats:
    """Container for {mu_i^S, sigma_i^S}_{i=0..N} computed once before TTA.

    Following Section 3.1 ("Statistics calculation"), we feed Q ID samples
    through the *unprompted* model and collect the CLS token at every layer.
    Per Figure 2 (c), Q = 32 is sufficient for ImageNet.

    Index 0 corresponds to the input embedding's CLS token; index 1..N
    correspond to layers L_1..L_N (depth = 12 for ViT-Base).
    """

    mu: List[torch.Tensor]  # length N (we drop the input-layer CLS for stability)
    sigma: List[torch.Tensor]  # length N
    mu_final: torch.Tensor  # mu_N^S (== mu[-1]); used by ActivationShifter

    @classmethod
    @torch.no_grad()
    def collect(
        cls,
        model: PromptedViT,
        loader,
        device: torch.device,
        max_samples: int = 32,
    ) -> "SourceStats":
        """Compute {mu_i^S, sigma_i^S} from up to `max_samples` ID images.

        We disable the prompts during collection by passing zero prompts -- the
        statistics describe the source domain when no prompt is applied, which
        matches Algorithm 1's notation (the prompt is the *learnable* signal).
        """
        feats_per_layer: List[List[torch.Tensor]] = []
        seen = 0
        zero_prompt = torch.zeros(model.num_prompts, model.embed_dim, device=device)
        model.eval()
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch["image"]
            x = x.to(device, non_blocking=True)
            n_take = min(x.shape[0], max_samples - seen)
            if n_take <= 0:
                break
            x = x[:n_take]
            _logits, _final, cls_list = model(
                x, prompt_override=zero_prompt, return_all_cls=True
            )
            if not feats_per_layer:
                feats_per_layer = [[] for _ in cls_list]
            for i, f in enumerate(cls_list):
                feats_per_layer[i].append(f.detach().float().cpu())
            seen += n_take
            if seen >= max_samples:
                break
        if seen == 0:
            raise RuntimeError("SourceStats.collect: empty loader.")
        mus, sigmas = [], []
        for layer_feats in feats_per_layer:
            f = torch.cat(layer_feats, dim=0)  # (Q, D)
            mus.append(f.mean(dim=0))  # (D,)
            sigmas.append(f.std(dim=0, unbiased=False))
        return cls(mu=mus, sigma=sigmas, mu_final=mus[-1].clone())


# ----------------------------------------------------------------------
# FOA itself
# ----------------------------------------------------------------------
class FOA:
    """Test-time Forward-Optimization Adaptation (Algorithm 1).

    Parameters
    ----------
    model : PromptedViT
    source_stats : SourceStats
    popsize : int
        Population size K.  Paper default = 28 = 4 + 3*log(prompt_dim).
    lam : float
        Trade-off lambda in Eqn. (5).  Paper sets
        lambda = 0.4 * BS/64 on ImageNet-C/V2/Sketch and 0.2 * BS/64 on
        ImageNet-R.
    activation_shift : bool
        Whether to apply the back-to-source activation shifting (Section 3.2).
    gamma : float
        Step size in Eqn. (7); default 1.0.
    alpha : float
        EMA factor in Eqn. (9); default 0.1.
    """

    def __init__(
        self,
        model: PromptedViT,
        source_stats: SourceStats,
        popsize: int = 28,
        lam: float = 0.4,
        activation_shift: bool = True,
        gamma: float = 1.0,
        alpha: float = 0.1,
        sigma0: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.stats = source_stats
        self.popsize = int(popsize)
        self.lam = float(lam)
        self.use_shift = bool(activation_shift)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

        prompt_dim = model.get_prompt_dim()
        # Sanity check vs. paper formula for K
        # K = 4 + 3 * log(prompt_dim)  (Hansen 2016 default, cited in paper)
        # We do not overwrite popsize, only log it for the user.
        self._recommended_K = int(4 + 3 * math.log(max(prompt_dim, 2)))

        self.cma = _CMAES(
            dim=prompt_dim, popsize=self.popsize, sigma0=sigma0, seed=seed
        )
        self.shifter: Optional[ActivationShifter] = None
        if self.use_shift:
            self.shifter = ActivationShifter(
                mu_source=self.stats.mu_final, gamma=self.gamma, alpha=self.alpha
            )

    # ------------------------------------------------------------------
    # Fitness function (Eqn. 5)
    # ------------------------------------------------------------------
    def _fitness(
        self,
        logits: torch.Tensor,  # (B, C)
        cls_per_layer: List[torch.Tensor],  # list of (B, D), length = N
    ) -> float:
        """Compute fitness in Eqn. (5).

        L = sum_x H(y_hat) + lambda * sum_i ||mu_i(X_t)-mu_i^S||_2
                                          + ||sigma_i(X_t)-sigma_i^S||_2
        """
        # (1) entropy term
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Sum over class -> per-sample entropy; sum over batch as in paper
        # (the paper writes "sum_x sum_c -y_c log y_c" so we sum over the batch).
        ent = -(probs * log_probs).sum(dim=-1).sum()

        # (2) activation discrepancy term
        disc = torch.zeros((), device=logits.device, dtype=torch.float32)
        for f, mu_s, sig_s in zip(cls_per_layer, self.stats.mu, self.stats.sigma):
            mu_t = f.float().mean(dim=0)
            sig_t = f.float().std(dim=0, unbiased=False)
            disc = (
                disc
                + torch.linalg.norm(mu_t - mu_s.to(mu_t))
                + torch.linalg.norm(sig_t - sig_s.to(sig_t))
            )

        return float((ent + self.lam * disc).item())

    # ------------------------------------------------------------------
    # Step on a single test batch  (Algorithm 1, body of the for loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Adapt to test batch X_t and return the chosen prediction.

        Returns
        -------
        y_hat : (B,) long tensor of predicted class indices.
        info : dict with keys {"best_fitness", "best_logits", "popsize"}.
        """
        device = next(self.model.parameters()).device
        x = x.to(device, non_blocking=True)

        # 1. Sample K candidate prompts in flat-vector form
        candidates = self.cma.ask()
        prompt_dim = self.model.get_prompt_dim()
        assert all(c.shape == (prompt_dim,) for c in candidates)

        # 2. Evaluate fitness for each candidate
        fitnesses: List[float] = []
        logits_per_k: List[torch.Tensor] = []
        cls_final_per_k: List[torch.Tensor] = []

        # Compute the offset *once* per batch using the unprompted features:
        # the shift is supposed to act after prompt selection, but for fitness
        # purposes we share the same EMA target across candidates so they are
        # ranked consistently.
        offset = None
        if self.use_shift and self.shifter is not None:
            zero_prompt = torch.zeros(
                self.model.num_prompts, self.model.embed_dim, device=device
            )
            _logits0, final0, _ = self.model(
                x, prompt_override=zero_prompt, return_all_cls=True
            )
            offset = self.shifter.update_and_get_offset(final0)

        for c in candidates:
            p = torch.from_numpy(c.astype(np.float32)).to(device)
            logits, cls_final, cls_list = self.model(
                x, prompt_override=p, return_all_cls=True, cls_offset=offset
            )
            v = self._fitness(logits, cls_list)
            fitnesses.append(v)
            logits_per_k.append(logits)
            cls_final_per_k.append(cls_final)

        # 3. Tell CMA-ES the fitness scores (it minimizes)
        self.cma.tell(candidates, fitnesses)

        # 4. Pick prediction with best (lowest) fitness, per Algorithm 1 line:
        #    "Select final Y_t from {Y_t^k} with best v_k."
        best_idx = int(np.argmin(np.asarray(fitnesses)))
        best_logits = logits_per_k[best_idx]
        # Persist the CMA mean as the model's "current prompt" for inspection.
        self.model.set_prompt(torch.from_numpy(self.cma.mean.astype(np.float32)))

        y_hat = best_logits.argmax(dim=-1)
        return y_hat, {
            "best_fitness": float(fitnesses[best_idx]),
            "best_logits": best_logits.detach(),
            "popsize": self.popsize,
            "recommended_K": self._recommended_K,
        }

    # ------------------------------------------------------------------
    # Convenience: stream over a DataLoader and report top-1 accuracy.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def adapt_and_evaluate(self, loader, device: torch.device) -> Dict[str, float]:
        """Stream over a DataLoader, run FOA per batch, report acc + ECE.

        ECE is computed with 15 equal-width confidence bins (paper convention).
        """
        n_correct = 0
        n_total = 0
        # bins: [count, sum_conf, sum_correct] x 15
        bins = np.zeros((15, 3), dtype=np.float64)
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["image"], batch["label"]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds, info = self.step(x)
            n_correct += int((preds == y).sum().item())
            n_total += int(y.numel())
            probs = F.softmax(info["best_logits"], dim=-1)
            conf, _ = probs.max(dim=-1)
            for c, p, t in zip(
                conf.cpu().numpy(), preds.cpu().numpy(), y.cpu().numpy()
            ):
                b = min(int(float(c) * 15), 14)
                bins[b, 0] += 1.0
                bins[b, 1] += float(c)
                bins[b, 2] += float(p == t)
        acc = 100.0 * n_correct / max(n_total, 1)
        ece = 0.0
        if n_total > 0:
            for b in range(15):
                if bins[b, 0] == 0:
                    continue
                avg_conf = bins[b, 1] / bins[b, 0]
                avg_acc = bins[b, 2] / bins[b, 0]
                ece += (bins[b, 0] / n_total) * abs(avg_conf - avg_acc)
            ece *= 100.0
        return {"accuracy": acc, "ece": ece, "n_total": float(n_total)}
