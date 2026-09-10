"""CompoNet architecture: Self-Composing Policies Network.

Implements the architecture described in Section 4 of
Malagon, M., Ceberio, J., & Lozano, J. A. (2024).
"Self-Composing Policies for Scalable Continual Reinforcement Learning."
Proceedings of the 41st International Conference on Machine Learning (ICML).

This file provides:
  - SelfComposingPolicyModule (Section 4.2)
      * OutputAttentionHead
      * InputAttentionHead
      * InternalPolicy
  - CompoNet container (cascading graph of modules, Section 4)
  - CompoNetActor: a thin actor wrapper for SAC / PPO use
  - PositionalEncoding (Vaswani et al. 2017, used in both attention heads)
  - AtariEncoder: CNN encoder for visual sequences (Appendix E.1)
  - ProgressiveNet baseline (Rusu et al. 2016) -- used for comparison
  - PackNet baseline (Mallya & Lazebnik 2018) -- used for comparison
  - BaselineActor: a simple MLP actor

Citation verification (CrossRef / paper_search; see README.md):
  - Progressive Neural Networks: Rusu et al., arXiv:1606.04671 (2016)
  - Meta-World benchmark: Yu et al., CoRL 2020
  - PPO: Schulman et al., arXiv:1707.06347
  - Vaswani et al. NeurIPS 2017 (positional encoding & attention).

Equations are referenced inline by section number from the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding (Vaswani et al., 2017) -- Section 4.2 specifies that
# CompoNet uses cosine positional encoding to mark module ordering.
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. (2017).

    The encoding is added to the keys matrix of both attention heads to make
    module order observable. It has the same shape as Phi^{k;s}, namely
    (n_prev, |A|) for the output head, and (n_prev + 1, |A|) for the input
    head (because the row of v from the output head is prepended).
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        # Build a (max_len, d_model) sinusoidal matrix once.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        # Register as non-trainable buffer.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, n: int) -> torch.Tensor:
        """Return positional encoding of shape (n, d_model)."""
        return self.pe[:n]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Output attention head (Section 4.2, paragraph "Output Attention Head").
# Generates a tentative output v as a linear combination of preceding
# policies' outputs (rows of Phi).  Values matrix V = Phi (no projection).
# ---------------------------------------------------------------------------
class OutputAttentionHead(nn.Module):
    """Output Attention Head as defined in Section 4.2.

    q  = h_s  W^Q_out          ,  W^Q_out in R^{d_enc x d_model}
    K  = (Phi + E_out) W^K_out ,  W^K_out in R^{|A|   x d_model}
    V  = Phi
    out = softmax( q K^T / sqrt(d_model) ) V                  (Eq. 1)
    """

    def __init__(
        self, d_enc: int, d_model: int, n_actions: int, max_modules: int = 1024
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        # Linear transformations (no bias by default, mirroring Vaswani 2017).
        self.W_Q = nn.Linear(d_enc, d_model, bias=False)
        self.W_K = nn.Linear(n_actions, d_model, bias=False)
        # Positional encoding has the same shape as Phi (rows = #prev modules,
        # columns = |A|).  Build a (max_modules, n_actions) PE table.
        self.pos_enc = PositionalEncoding(max_modules, n_actions)

    def forward(
        self, h_s: torch.Tensor, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute v and the attention weights.

        Args:
            h_s : (B, d_enc) state representation
            phi : (B, n_prev, |A|) outputs of preceding policy modules

        Returns:
            v        : (B, |A|) tentative output vector
            attn_w   : (B, n_prev) attention weights over previous modules
        """
        bsz, n_prev, n_a = phi.shape
        if n_prev == 0:
            # No previous modules -- v is zero, attention weights empty.
            return phi.new_zeros(bsz, self.n_actions), phi.new_zeros(bsz, 0)

        # q : (B, d_model)
        q = self.W_Q(h_s)
        # K : (B, n_prev, d_model). Add positional encoding to phi BEFORE proj.
        pe = self.pos_enc(n_prev).to(phi.dtype).to(phi.device)  # (n_prev, |A|)
        K = self.W_K(phi + pe.unsqueeze(0))  # (B, n_prev, d_model)
        # Scaled dot-product attention, scaling by sqrt(d_model) per Eq. 1.
        scores = torch.einsum("bd,bnd->bn", q, K) / math.sqrt(self.d_model)
        attn_w = F.softmax(scores, dim=-1)  # (B, n_prev)
        # V = Phi (no linear transformation).  v = sum_j attn_j * Phi_j.
        v = torch.einsum("bn,bna->ba", attn_w, phi)  # (B, |A|)
        return v, attn_w


# ---------------------------------------------------------------------------
# Input attention head (Section 4.2, paragraph "Input Attention Head").
# Retrieves relevant info from previous policies *and* the tentative v.
# ---------------------------------------------------------------------------
class InputAttentionHead(nn.Module):
    """Input Attention Head.

    q  = h_s        W^Q_in
    K  = (P + E_in) W^K_in     where P = vstack(v, Phi)         (rows = n_prev+1)
    V  = P          W^V_in
    out = softmax( q K^T / sqrt(d_model) ) V
    """

    def __init__(
        self, d_enc: int, d_model: int, n_actions: int, max_modules: int = 1024
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_enc, d_model, bias=False)
        self.W_K = nn.Linear(n_actions, d_model, bias=False)
        self.W_V = nn.Linear(n_actions, d_model, bias=False)
        self.pos_enc = PositionalEncoding(max_modules + 1, n_actions)

    def forward(
        self, h_s: torch.Tensor, phi: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the contextualized vector and attention weights.

        Args:
            h_s : (B, d_enc)
            phi : (B, n_prev, |A|)
            v   : (B, |A|)  -- tentative output from the output attention head

        Returns:
            ctx    : (B, d_model)  -- contextual representation
            attn_w : (B, n_prev+1) -- attention weights (v is row 0)
        """
        bsz, n_prev, n_a = phi.shape
        # Row-wise concatenation: P has shape (B, n_prev+1, |A|).  v is row 0.
        P = torch.cat([v.unsqueeze(1), phi], dim=1)
        n_rows = P.shape[1]
        pe = self.pos_enc(n_rows).to(P.dtype).to(P.device)  # (n_rows, |A|)
        # Linear projections.
        q = self.W_Q(h_s)  # (B, d_model)
        K = self.W_K(P + pe.unsqueeze(0))  # (B, n_rows, d_model)
        V = self.W_V(P)  # (B, n_rows, d_model)
        scores = torch.einsum("bd,bnd->bn", q, K) / math.sqrt(self.d_model)
        attn_w = F.softmax(scores, dim=-1)  # (B, n_rows)
        ctx = torch.einsum("bn,bnd->bd", attn_w, V)  # (B, d_model)
        return ctx, attn_w


# ---------------------------------------------------------------------------
# Internal policy (Section 4.2, paragraph "Internal Policy").
# A small MLP that takes [h_s ; ctx] and outputs an |A|-dimensional vector
# that is *added* to the tentative v.
# ---------------------------------------------------------------------------
class InternalPolicy(nn.Module):
    """MLP internal policy (sometimes called residual head)."""

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        n_actions: int,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers: List[nn.Module] = []
        in_dim = d_enc + d_model
        # First hidden layer.
        layers.append(nn.Linear(in_dim, d_model))
        layers.append(act())
        # Intermediate layers.
        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(act())
        # Output layer.
        layers.append(nn.Linear(d_model, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, h_s: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_s, ctx], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# A single self-composing policy module (Figure 2).
# ---------------------------------------------------------------------------
class SelfComposingPolicyModule(nn.Module):
    """Single self-composing policy module (Section 4.2 of the paper).

    Forward:
        out = normalize( v + InternalPolicy(h_s, ctx) )

    where the optional normalization depends on the action space:
      - discrete (categorical): apply softmax (returns probabilities)
      - continuous (Gaussian mean): leave unnormalized (or tanh-bound)
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        n_actions: int,
        internal_layers: int = 2,
        action_space: str = "discrete",
        max_modules: int = 1024,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        assert action_space in ("discrete", "continuous")
        self.action_space = action_space
        self.n_actions = n_actions
        self.out_attn = OutputAttentionHead(d_enc, d_model, n_actions, max_modules)
        self.in_attn = InputAttentionHead(d_enc, d_model, n_actions, max_modules)
        self.internal = InternalPolicy(
            d_enc=d_enc,
            d_model=d_model,
            n_actions=n_actions,
            num_layers=internal_layers,
            activation=activation,
        )

    def forward(self, h_s: torch.Tensor, phi: torch.Tensor, return_aux: bool = False):
        """Compute the module output.

        Args:
            h_s : (B, d_enc) -- encoded state
            phi : (B, n_prev, |A|) outputs of preceding modules; n_prev >= 0

        Returns:
            out : (B, |A|)  module output (probabilities if discrete)
            aux : optional dict with intermediate tensors (for analysis)
        """
        # Output attention head -- proposes tentative v.
        v, attn_out = self.out_attn(h_s, phi)
        # Input attention head -- contextualizes v together with phi.
        ctx, attn_in = self.in_attn(h_s, phi, v)
        # Internal policy (residual on top of v).
        residual = self.internal(h_s, ctx)
        raw = v + residual
        # Optional normalization (Section 4.2 last paragraph).
        if self.action_space == "discrete":
            out = F.softmax(raw, dim=-1)
        else:
            out = raw  # continuous: returned as Gaussian mean
        if return_aux:
            return out, {
                "v": v,
                "residual": residual,
                "out_attn": attn_out,
                "in_attn": attn_in,
                "raw": raw,
            }
        return out


# ---------------------------------------------------------------------------
# Atari / ALE encoder (Appendix E.1).  3 conv layers + dense.
# ---------------------------------------------------------------------------
class AtariEncoder(nn.Module):
    """CNN encoder used for ALE environments (Appendix E.1).

    Architecture: (Nature DQN-style, as in CleanRL [Huang et al., 2022])
        Conv(32, 8, stride=4) -> ReLU
        Conv(64, 4, stride=2) -> ReLU
        Conv(64, 3, stride=1) -> ReLU
        Flatten -> Linear(d_enc=512) -> ReLU
    """

    def __init__(
        self, in_channels: int = 4, d_enc: int = 512, image_size: int = 84
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        # Compute conv output size by a dry run.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            n_flat = self.conv(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, d_enc),
            nn.ReLU(inplace=True),
        )
        self.d_enc = d_enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x in [0, 255] float or uint8.  Normalize to [0, 1].
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        h = self.conv(x).reshape(x.shape[0], -1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# CompoNet container -- the cascading graph of self-composing modules
# (Section 4, Figure 1).  Adds a new module per task; freezes previous ones.
# ---------------------------------------------------------------------------
class CompoNet(nn.Module):
    """The full Self-Composing Policies Network.

    Maintains a list of `SelfComposingPolicyModule`s.  When `current_module`
    indexes the trainable module, all preceding modules are frozen and used
    only to produce the matrix Phi^{k;s}.
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        n_actions: int,
        internal_layers: int = 2,
        action_space: str = "discrete",
        max_modules: int = 1024,
    ) -> None:
        super().__init__()
        self.d_enc = d_enc
        self.d_model = d_model
        self.n_actions = n_actions
        self.action_space = action_space
        self.internal_layers = internal_layers
        self.max_modules = max_modules
        self.modules_list = nn.ModuleList()
        # Add the initial trainable module.
        self.add_new_module(initial=True)

    # -- module management ---------------------------------------------------
    def add_new_module(self, initial: bool = False) -> None:
        """Freeze the current trainable module and add a fresh one.

        Called between tasks.  In the very first call (initial=True) we just
        create the first module (nothing to freeze).
        """
        if not initial:
            # Freeze parameters of the previously trainable module.
            for p in self.modules_list[-1].parameters():
                p.requires_grad_(False)
            self.modules_list[-1].eval()
        new_mod = SelfComposingPolicyModule(
            d_enc=self.d_enc,
            d_model=self.d_model,
            n_actions=self.n_actions,
            internal_layers=self.internal_layers,
            action_space=self.action_space,
            max_modules=self.max_modules,
        )
        self.modules_list.append(new_mod)

    @property
    def current_module(self) -> SelfComposingPolicyModule:
        return self.modules_list[-1]

    @property
    def num_tasks(self) -> int:
        return len(self.modules_list)

    # -- forward -------------------------------------------------------------
    def compute_phi(self, h_s: torch.Tensor) -> torch.Tensor:
        """Compute Phi^{k;s} -- the (B, k-1, |A|) matrix of previous outputs.

        For k=1 (i.e. the very first task) returns a (B, 0, |A|) tensor.
        """
        bsz = h_s.shape[0]
        if len(self.modules_list) == 1:
            return h_s.new_zeros(bsz, 0, self.n_actions)
        outs: List[torch.Tensor] = []
        # Recursively compute previous modules' outputs.  Each previous module
        # only sees Phi over the modules that came before *it* (so we recurse).
        with torch.no_grad():
            for j in range(len(self.modules_list) - 1):
                phi_j = self._phi_up_to(j, h_s)
                outs.append(self.modules_list[j](h_s, phi_j))
        # Stack: (B, k-1, |A|)
        return torch.stack(outs, dim=1)

    def _phi_up_to(self, j: int, h_s: torch.Tensor) -> torch.Tensor:
        """Phi for module j: outputs of modules 0..j-1, shape (B, j, |A|)."""
        bsz = h_s.shape[0]
        if j == 0:
            return h_s.new_zeros(bsz, 0, self.n_actions)
        outs: List[torch.Tensor] = []
        for i in range(j):
            phi_i = self._phi_up_to(i, h_s)
            outs.append(self.modules_list[i](h_s, phi_i))
        return torch.stack(outs, dim=1)

    def forward(self, h_s: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        phi = self.compute_phi(h_s)
        return self.current_module(h_s, phi, return_aux=return_aux)


# ---------------------------------------------------------------------------
# Actor wrapper combining encoder + CompoNet (Section 5.2 / Appendix E).
# Used for both PPO (discrete) and SAC (continuous, returns mean & log_std).
# ---------------------------------------------------------------------------
class CompoNetActor(nn.Module):
    """Actor module: encoder + CompoNet + (optional) log_std head."""

    def __init__(
        self,
        encoder: Optional[nn.Module],
        d_enc: int,
        d_model: int,
        n_actions: int,
        action_space: str = "discrete",
        internal_layers: int = 2,
        max_modules: int = 1024,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.action_space = action_space
        self.componet = CompoNet(
            d_enc=d_enc,
            d_model=d_model,
            n_actions=n_actions,
            internal_layers=internal_layers,
            action_space=action_space,
            max_modules=max_modules,
        )
        # For continuous (SAC) actions we predict a log-std for each dim.
        if action_space == "continuous":
            self.log_std = nn.Sequential(
                nn.Linear(d_enc, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, n_actions),
            )
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor):
        h = self.encode(obs)
        out = self.componet(h)
        if self.action_space == "continuous":
            log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
            return out, log_std
        # discrete -> probabilities
        return out

    def add_new_task(self) -> None:
        self.componet.add_new_module()
        # Re-initialize log_std head per task as in Wolczyk et al. 2021.
        if self.action_space == "continuous":
            for m in self.log_std:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Baselines used in the experimental comparison (Section 5.2).
# ---------------------------------------------------------------------------
class BaselineActor(nn.Module):
    """A simple MLP actor for the BASELINE / FT-1 / FT-N methods.

    The same MLP architecture is used for ProgressiveNet columns.
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        n_actions: int,
        num_layers: int = 3,
        action_space: str = "discrete",
    ) -> None:
        super().__init__()
        self.action_space = action_space
        layers: List[nn.Module] = [nn.Linear(d_enc, d_model), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, n_actions)
        if action_space == "continuous":
            self.log_std_head = nn.Linear(d_model, n_actions)

    def forward(self, h: torch.Tensor):
        z = self.trunk(h)
        out = self.head(z)
        if self.action_space == "discrete":
            return F.softmax(out, dim=-1)
        log_std = self.log_std_head(z).clamp(-20.0, 2.0)
        return out, log_std


# ---------------------------------------------------------------------------
# Progressive Neural Network baseline (Rusu et al. 2016).
# Adds a new "column" per task and lateral connections from frozen columns
# to every layer of the new column.  Memory grows quadratically.
# ---------------------------------------------------------------------------
class ProgressiveColumn(nn.Module):
    def __init__(
        self, d_in: int, d_hidden: int, num_layers: int, n_prev_columns: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        # Main weights for this column (per layer).
        self.W = nn.ModuleList()
        # Lateral adapters: one Linear from each previous column at each layer.
        self.U = nn.ModuleList()
        cur_in = d_in
        for li in range(num_layers):
            self.W.append(nn.Linear(cur_in, d_hidden))
            for _ in range(n_prev_columns):
                self.U.append(nn.Linear(d_hidden, d_hidden))
            cur_in = d_hidden
        self.n_prev = n_prev_columns

    def forward(
        self, x: torch.Tensor, prev_acts: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Returns list of activations per layer.

        prev_acts[c][l] is layer-l activation of previous column c.
        """
        acts: List[torch.Tensor] = []
        h = x
        idx = 0  # running index into self.U
        for li in range(self.num_layers):
            z = self.W[li](h)
            # add lateral contributions
            for c in range(self.n_prev):
                lateral = prev_acts[c][li]
                z = z + self.U[idx](lateral)
                idx += 1
            h = F.relu(z)
            acts.append(h)
        return acts


class ProgressiveNet(nn.Module):
    """ProgressiveNet baseline (Rusu et al. 2016).

    Reference: Rusu, A. A. et al. "Progressive Neural Networks."
    arXiv:1606.04671 (2016). Verified via paper_search; the paper has no
    DOI on CrossRef (arXiv-only), so verification fell through to manual.
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        n_actions: int,
        num_layers: int = 3,
        action_space: str = "discrete",
    ) -> None:
        super().__init__()
        self.d_enc = d_enc
        self.d_model = d_model
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.action_space = action_space
        self.columns = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.add_new_task(initial=True)

    def add_new_task(self, initial: bool = False) -> None:
        if not initial:
            for p in self.columns[-1].parameters():
                p.requires_grad_(False)
            self.columns[-1].eval()
        col = ProgressiveColumn(
            d_in=self.d_enc,
            d_hidden=self.d_model,
            num_layers=self.num_layers,
            n_prev_columns=len(self.columns),
        )
        self.columns.append(col)
        self.heads.append(nn.Linear(self.d_model, self.n_actions))

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        # Compute activations of all previous columns first (no grad).
        prev_acts: List[List[torch.Tensor]] = []
        with torch.no_grad():
            for c, col in enumerate(self.columns[:-1]):
                pa = []
                for cc in range(c):
                    pa.append(prev_acts[cc])
                prev_acts.append(col(h_s, pa))
        # Forward the current column with lateral connections.
        cur_acts = self.columns[-1](h_s, prev_acts)
        logits = self.heads[-1](cur_acts[-1])
        if self.action_space == "discrete":
            return F.softmax(logits, dim=-1)
        return logits


# ---------------------------------------------------------------------------
# PackNet baseline (Mallya & Lazebnik 2018).
# Single-network method that prunes parameters and freezes the top
# `1/N` fraction per task, where N = total tasks (Appendix E.2).
# ---------------------------------------------------------------------------
class PackNet(nn.Module):
    """PackNet baseline.

    Reference: Mallya, A. & Lazebnik, S. "PackNet: Adding Multiple Tasks
    to a Single Network by Iterative Pruning." CVPR 2018.

    Implements the prune-and-freeze logic.  The retraining phase length
    (20% of timestep budget) is enforced by the trainer, not here.
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        n_actions: int,
        total_tasks: int,
        num_layers: int = 3,
        action_space: str = "discrete",
    ) -> None:
        super().__init__()
        self.total_tasks = total_tasks
        self.action_space = action_space
        self.actor = BaselineActor(d_enc, d_model, n_actions, num_layers, action_space)
        self.heads: List[nn.Linear] = [self.actor.head]
        # Per-task masks (boolean) for each linear layer in the trunk.
        self.task_id = 0
        self.masks: List[List[torch.Tensor]] = [
            [torch.ones_like(p, dtype=torch.bool) for p in self._trunk_weights()]
        ]

    def _trunk_weights(self) -> List[torch.Tensor]:
        return [m.weight for m in self.actor.trunk if isinstance(m, nn.Linear)]

    def prune(self, fraction: float) -> None:
        """Prune the smallest-magnitude `fraction` of *available* weights."""
        with torch.no_grad():
            for w in self._trunk_weights():
                avail = self.masks[self.task_id][self._idx_of(w)]
                vals = w.abs()[avail]
                if vals.numel() == 0:
                    continue
                k = int(fraction * vals.numel())
                if k <= 0:
                    continue
                thresh = vals.kthvalue(k).values
                small = (w.abs() <= thresh) & avail
                w[small] = 0.0
                # Update mask: the kept weights become this task's frozen set.
                self.masks[self.task_id][self._idx_of(w)] = avail & ~small

    def _idx_of(self, w: torch.Tensor) -> int:
        for i, ww in enumerate(self._trunk_weights()):
            if ww is w:
                return i
        raise RuntimeError("weight not in trunk")

    def add_new_task(self) -> None:
        # Save head, instantiate new one, init mask for new task.
        new_head = nn.Linear(self.actor.head.in_features, self.actor.head.out_features)
        self.heads.append(new_head)
        self.actor.head = new_head
        self.task_id += 1
        self.masks.append(
            [torch.ones_like(w, dtype=torch.bool) for w in self._trunk_weights()]
        )

    def forward(self, h: torch.Tensor):
        return self.actor(h)


# ---------------------------------------------------------------------------
# Convenience dataclass for hyperparameters (used by trainer; see configs/).
# ---------------------------------------------------------------------------
@dataclass
class CompoNetConfig:
    d_enc: int = 39  # Meta-World state dim by default
    d_model: int = 256  # Hidden dim (Table E.1)
    n_actions: int = 4  # Meta-World action dim by default
    internal_layers: int = 2
    action_space: str = "continuous"
    max_modules: int = 1024
