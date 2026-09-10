"""Functional Reward Encoding (FRE) -- core neural network architectures.

Implements:
    * FREEncoder   -- permutation-invariant transformer producing p_theta(z | L_eta^e)
                       (Equation 6, Section 4.1 of the paper)
    * FREDecoder   -- feed-forward q_theta(eta(s_d) | s_d, z)
    * FRE          -- joint encoder/decoder with ELBO loss
                       L = -E[ sum_k log q(eta(s_d) | s_d, z) ] + beta * KL( p(z|.) || N(0,I) )
    * Actor / Critic / VNet -- IQL components conditioned on the FRE latent z
    * FREAgent     -- end-to-end training wrapper (encoder + IQL policy)

Mirrors the paper's variable names where possible:
    K       == K_encode        (32)
    K'      == K_decode        (8)
    z       == latent (128-dim)
    eta     == reward function
    s_e     == encoder state
    s_d     == decoder state

Verified reference for IQL (the chosen RL backbone):
    Kostrikov, Nair, Levine. "Offline Reinforcement Learning with Implicit
    Q-Learning." arXiv:2110.06169, 2021.  Authors verified via Semantic
    Scholar; arXiv DOIs are not indexed in CrossRef so ref_verify reports
    NOT FOUND -- this is expected for preprint-only references.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mlp(
    sizes: Sequence[int],
    activation: type = nn.ReLU,
    out_activation: type | None = None,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Construct a feed-forward MLP. `sizes` includes both input and output dims."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# FRE encoder
# ---------------------------------------------------------------------------
class FREEncoder(nn.Module):
    """Permutation-invariant transformer encoder p_theta(z | {(s_k, eta(s_k))}).

    Architecture (per addendum to the paper):
      - Each scalar reward eta(s_k) is rescaled to [0,1] then floor-discretized
        into one of `n_reward_bins` (= 32) buckets and looked up in a learned
        embedding table of width `reward_embed_dim` (= 64).
      - Each state s_k goes through a linear projection of width
        `state_embed_dim` (= 64).
      - The two embeddings are concatenated to form a 128-dim token.
      - K = 32 such tokens are processed by a transformer **without** positional
        encodings or causal masking -- so the operation is permutation-invariant.
        The transformer has `num_heads=4`; the residual/attention activations
        are 128-dim and each MLP block expands to 256 then back to 128 (the
        "Encoder Layers" list [256,256,256,256] in Table 3 specifies the MLP
        block dim per transformer layer).
      - The final-layer tokens are mean-pooled and projected to (mu, log_sigma)
        of a 128-dim diagonal Gaussian.
    """

    def __init__(
        self,
        state_dim: int,
        n_reward_bins: int = 32,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        token_dim: int = 128,
        z_dim: int = 128,
        mlp_dims: Sequence[int] = (256, 256, 256, 256),
        num_heads: int = 4,
        reward_min: float = -1.0,
        reward_max: float = 0.0,
    ):
        super().__init__()
        assert state_embed_dim + reward_embed_dim == token_dim, (
            "addendum: state_embed (64) + reward_embed (64) must equal token_dim (128)"
        )

        self.n_bins = n_reward_bins
        self.reward_min = reward_min
        self.reward_max = reward_max

        # learned reward embedding table (32 bins -> 64-dim each)
        self.reward_embed = nn.Embedding(n_reward_bins, reward_embed_dim)
        # linear state projection (state_dim -> 64)
        self.state_proj = nn.Linear(state_dim, state_embed_dim)

        # stack of TransformerEncoderLayers, one per "Encoder Layer" entry
        layers = []
        for hidden in mlp_dims:
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.transformer = nn.ModuleList(layers)

        self.mu_head = nn.Linear(token_dim, z_dim)
        self.logstd_head = nn.Linear(token_dim, z_dim)

    # ------------------------------------------------------------------
    def discretize(self, r: torch.Tensor) -> torch.Tensor:
        """Per addendum: rescale to [0,1], multiply by n_bins, floor."""
        r = (r - self.reward_min) / max(self.reward_max - self.reward_min, 1e-8)
        r = torch.clamp(r, 0.0, 1.0 - 1e-6)
        return torch.floor(r * self.n_bins).long()

    # ------------------------------------------------------------------
    def forward(self, s_e: torch.Tensor, r_e: torch.Tensor):
        """
        Args:
            s_e : (B, K, state_dim) encoder states
            r_e : (B, K)             scalar rewards eta(s_e)
        Returns:
            mu, logstd : (B, z_dim) each
        """
        b, k, _ = s_e.shape
        bin_ids = self.discretize(r_e)  # (B, K)
        r_emb = self.reward_embed(bin_ids)  # (B, K, reward_embed)
        s_emb = self.state_proj(s_e)  # (B, K, state_embed)
        tokens = torch.cat([s_emb, r_emb], dim=-1)  # (B, K, token_dim)

        h = tokens
        for layer in self.transformer:
            h = layer(h)
        pooled = h.mean(dim=1)  # permutation-invariant (mean over set)

        mu = self.mu_head(pooled)
        logstd = self.logstd_head(pooled).clamp(-5.0, 2.0)
        return mu, logstd

    # ------------------------------------------------------------------
    def sample(self, s_e: torch.Tensor, r_e: torch.Tensor):
        mu, logstd = self.forward(s_e, r_e)
        std = logstd.exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        # KL( N(mu, sigma^2) || N(0, I) )
        kl = 0.5 * (mu.pow(2) + std.pow(2) - 1.0 - 2.0 * logstd).sum(-1)
        return z, mu, logstd, kl


# ---------------------------------------------------------------------------
# FRE decoder
# ---------------------------------------------------------------------------
class FREDecoder(nn.Module):
    """q_theta(eta(s_d) | s_d, z) -- raw state and z concatenated (addendum).

    Per addendum: there is no embedding step on the decoder state -- the raw
    s_d and the latent z are concatenated and passed through a 3-layer MLP
    [512,512,512].
    """

    def __init__(
        self, state_dim: int, z_dim: int = 128, hidden: Sequence[int] = (512, 512, 512)
    ):
        super().__init__()
        self.net = mlp([state_dim + z_dim, *hidden, 1])

    def forward(self, s_d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """s_d: (B, K', state_dim);  z: (B, z_dim) -> (B, K')."""
        b, kp, _ = s_d.shape
        z_rep = z.unsqueeze(1).expand(b, kp, z.shape[-1])
        x = torch.cat([s_d, z_rep], dim=-1)  # (B, K', state+z)
        return self.net(x).squeeze(-1)  # (B, K')


# ---------------------------------------------------------------------------
# Combined FRE
# ---------------------------------------------------------------------------
class FRE(nn.Module):
    """Encoder + decoder, trained jointly with the variational ELBO of Eq. 6.

    Loss:
        L_FRE = MSE(q_theta(s_d, z),  eta(s_d))   +   beta * KL(p(z|.) || N(0,I))
    """

    def __init__(
        self,
        state_dim: int,
        *,
        n_reward_bins: int = 32,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        token_dim: int = 128,
        z_dim: int = 128,
        mlp_dims: Sequence[int] = (256, 256, 256, 256),
        decoder_hidden: Sequence[int] = (512, 512, 512),
        num_heads: int = 4,
        beta_kl: float = 0.01,
        reward_min: float = -1.0,
        reward_max: float = 0.0,
    ):
        super().__init__()
        self.encoder = FREEncoder(
            state_dim,
            n_reward_bins,
            state_embed_dim,
            reward_embed_dim,
            token_dim,
            z_dim,
            mlp_dims,
            num_heads,
            reward_min,
            reward_max,
        )
        self.decoder = FREDecoder(state_dim, z_dim, decoder_hidden)
        self.beta = beta_kl
        self.z_dim = z_dim

    # ------------------------------------------------------------------
    def loss(
        self, s_e: torch.Tensor, r_e: torch.Tensor, s_d: torch.Tensor, r_d: torch.Tensor
    ):
        """Computes the FRE ELBO loss.

        Args:
            s_e, r_e : (B, K, state_dim), (B, K)   encoder context
            s_d, r_d : (B, K', state_dim), (B, K') decoder targets
        Returns:
            total_loss, dict of metrics
        """
        z, mu, logstd, kl = self.encoder.sample(s_e, r_e)
        r_pred = self.decoder(s_d, z)
        recon_mse = F.mse_loss(r_pred, r_d, reduction="none").sum(-1).mean()
        kl_mean = kl.mean()
        total = recon_mse + self.beta * kl_mean
        return total, {
            "recon_mse": recon_mse.item(),
            "kl": kl_mean.item(),
            "loss": total.item(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, s_e: torch.Tensor, r_e: torch.Tensor) -> torch.Tensor:
        """Inference: encode K state-reward pairs into the posterior mean z."""
        mu, _ = self.encoder(s_e, r_e)
        return mu


# ---------------------------------------------------------------------------
# IQL components (Section 4.3 -- conditioned on z)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """Pi(a | s, z) -- Gaussian policy with z concatenated to s."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int,
        hidden: Sequence[int] = (512, 512, 512),
        log_std_min: float = -5.0,
    ):
        super().__init__()
        self.trunk = mlp([state_dim + z_dim, *hidden], layer_norm=True)
        self.mu = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Linear(hidden[-1], action_dim)
        self.log_std_min = log_std_min

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        h = self.trunk(torch.cat([s, z], dim=-1))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, 2.0)
        return mu, log_std

    def log_prob(self, s, z, a):
        mu, log_std = self.forward(s, z)
        std = log_std.exp()
        return (
            -0.5 * ((a - mu) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        ).sum(-1)

    def sample(self, s, z, deterministic: bool = False):
        mu, log_std = self.forward(s, z)
        if deterministic:
            return torch.tanh(mu)
        eps = torch.randn_like(mu)
        return torch.tanh(mu + log_std.exp() * eps)


class Critic(nn.Module):
    """Twin Q(s, a, z) -- two heads for clipped double-Q (IQL)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int,
        hidden: Sequence[int] = (512, 512, 512),
    ):
        super().__init__()
        in_dim = state_dim + action_dim + z_dim
        self.q1 = mlp([in_dim, *hidden, 1], layer_norm=True)
        self.q2 = mlp([in_dim, *hidden, 1], layer_norm=True)

    def forward(self, s, a, z):
        x = torch.cat([s, a, z], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class VNet(nn.Module):
    """V(s, z) -- expectile-regressed value network from IQL."""

    def __init__(
        self, state_dim: int, z_dim: int, hidden: Sequence[int] = (512, 512, 512)
    ):
        super().__init__()
        self.v = mlp([state_dim + z_dim, *hidden, 1], layer_norm=True)

    def forward(self, s, z):
        return self.v(torch.cat([s, z], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Full FRE-IQL agent
# ---------------------------------------------------------------------------
class FREAgent(nn.Module):
    """Encoder + IQL policy/critic/value, trained per Algorithm 1 of the paper.

    Use `compute_fre_loss` during the encoder pretraining phase, and
    `compute_iql_loss` after freezing the encoder.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        z_dim: int = 128,
        n_reward_bins: int = 32,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        token_dim: int = 128,
        enc_mlp_dims: Sequence[int] = (256, 256, 256, 256),
        dec_hidden: Sequence[int] = (512, 512, 512),
        rl_hidden: Sequence[int] = (512, 512, 512),
        num_heads: int = 4,
        beta_kl: float = 0.01,
        discount: float = 0.88,
        expectile: float = 0.8,
        awr_temperature: float = 3.0,
        target_tau: float = 0.001,
    ):
        super().__init__()
        self.fre = FRE(
            state_dim,
            n_reward_bins=n_reward_bins,
            state_embed_dim=state_embed_dim,
            reward_embed_dim=reward_embed_dim,
            token_dim=token_dim,
            z_dim=z_dim,
            mlp_dims=enc_mlp_dims,
            decoder_hidden=dec_hidden,
            num_heads=num_heads,
            beta_kl=beta_kl,
        )

        self.actor = Actor(state_dim, action_dim, z_dim, rl_hidden)
        self.critic = Critic(state_dim, action_dim, z_dim, rl_hidden)
        self.critic_target = Critic(state_dim, action_dim, z_dim, rl_hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.value = VNet(state_dim, z_dim, rl_hidden)

        self.gamma = discount
        self.tau_iql = expectile
        self.beta_awr = awr_temperature
        self.target_tau = target_tau
        self.z_dim = z_dim

    # ------------------------------------------------------------------
    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """L_2^tau(diff) = | tau - 1{diff<0} | * diff^2  (IQL Eq. 6)."""
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()

    # ------------------------------------------------------------------
    def compute_iql_loss(self, batch: dict, z: torch.Tensor):
        """One IQL update step, with z (latent task) concatenated everywhere.

        batch keys: s, a, r, s_next, done    (all already-tensor)
        """
        s, a, r, s_next, done = (
            batch["s"],
            batch["a"],
            batch["r"],
            batch["s_next"],
            batch["done"],
        )

        # ----- value loss: V(s,z) -> expectile of min(Q_target) ----------
        with torch.no_grad():
            q1_t, q2_t = self.critic_target(s, a, z)
            q_t = torch.minimum(q1_t, q2_t)
        v = self.value(s, z)
        v_loss = self.expectile_loss(q_t - v, self.tau_iql)

        # ----- critic loss: Q(s,a,z) <- r + gamma * V(s', z) -------------
        with torch.no_grad():
            v_next = self.value(s_next, z)
            target_q = r + self.gamma * (1.0 - done) * v_next
        q1, q2 = self.critic(s, a, z)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # ----- actor loss: AWR with exp(beta * advantage) ----------------
        with torch.no_grad():
            adv = q_t - v.detach()
            weight = torch.clamp(torch.exp(self.beta_awr * adv), max=100.0)
        log_prob = self.actor.log_prob(s, z, a)
        a_loss = -(weight * log_prob).mean()

        return (
            v_loss,
            q_loss,
            a_loss,
            {
                "v_loss": v_loss.item(),
                "q_loss": q_loss.item(),
                "a_loss": a_loss.item(),
                "adv_mean": adv.mean().item(),
            },
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1.0 - self.target_tau).add_(self.target_tau * p.data)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, s: torch.Tensor, z: torch.Tensor, deterministic: bool = True):
        return self.actor.sample(s, z, deterministic=deterministic)
