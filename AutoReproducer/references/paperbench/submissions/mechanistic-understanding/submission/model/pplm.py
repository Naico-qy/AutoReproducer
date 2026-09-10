"""PPLM-based toxic continuation generator (Section 4.2).

This implements Plug-and-Play Language Models (Dathathri et al. 2019, ICLR)
specialized to the case where the attribute classifier is the linear toxic
probe W_toxic[:, 1] trained in Section 3.1.

We use it to produce the *negative* (toxic) continuation y_- of each
preference pair; the *positive* (non-toxic) continuation y_+ is just
greedy decoding from the unmodified GPT-2.  Together this gives the
24,576 (prompt, y+, y-) triples described in Section 4.2.

Implementation notes
--------------------
* PPLM perturbs the past key/value cache via gradients of an attribute
  classifier so that p(a | y) is increased.  Here a = "toxic" and the
  classifier is the linear W_toxic, applied to the *mean-pooled last-layer
  hidden state*, exactly the same way the probe was trained.
* We apply the standard PPLM mixing rule:
        p(y | a)  proportional to  p(y) * p(a | y)^gamma
  -- realised by mixing the perturbed and unperturbed logits with
  ``gm_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PPLMConfig:
    step_size: float = 0.04
    num_iterations: int = 3
    window_length: int = 5
    horizon_length: int = 1
    kl_scale: float = 0.01
    gm_scale: float = 0.95
    gamma: float = 1.5
    grad_norm_clip: float = 1.0
    target_class: int = 1  # toxic = column 1 of W_toxic


def _to_var(t):
    return t.detach().clone().requires_grad_(True)


def perturb_past(
    past,
    last_hidden,
    classifier,
    cfg: PPLMConfig,
):
    """Run PPLM's attribute-driven perturbation of the past KV cache.

    Parameters
    ----------
    past         : tuple of (key, value) tensors, the GPT-2 cache.
    last_hidden  : [B, T, d] -- the last-layer residual stream that fed the
                   classifier.  Mean-pooled across T inside this fn.
    classifier   : a callable mapping mean-pooled hidden -> logits [B, 2].
    cfg          : PPLMConfig
    """
    # Build per-layer accumulators of the same shape as past
    accum_grads = [(torch.zeros_like(k), torch.zeros_like(v)) for (k, v) in past]

    for _ in range(cfg.num_iterations):
        perturbed = []
        for (k, v), (gk, gv) in zip(past, accum_grads):
            perturbed.append((_to_var(k + gk), _to_var(v + gv)))

        pooled = last_hidden.mean(dim=1)  # [B, d]
        logits = classifier(pooled)  # [B, 2]
        logp_target = F.log_softmax(logits, dim=-1)[:, cfg.target_class].sum()

        # Gradient w.r.t. the perturbations
        grads = torch.autograd.grad(
            outputs=logp_target,
            inputs=[t for kv in perturbed for t in kv],
            allow_unused=True,
            retain_graph=False,
        )

        new_accum = []
        for i, (gk, gv) in enumerate(accum_grads):
            dk, dv = grads[2 * i], grads[2 * i + 1]
            if dk is None:
                dk = torch.zeros_like(gk)
            if dv is None:
                dv = torch.zeros_like(gv)

            # Normalize per-layer gradients (PPLM trick)
            denom_k = dk.norm() + 1e-8
            denom_v = dv.norm() + 1e-8
            dk = dk / denom_k
            dv = dv / denom_v

            gk = gk + cfg.step_size * dk
            gv = gv + cfg.step_size * dv
            # KL "anchor": shrink toward 0 so we stay close to the original past
            gk = gk * (1.0 - cfg.kl_scale)
            gv = gv * (1.0 - cfg.kl_scale)
            new_accum.append((gk.detach(), gv.detach()))
        accum_grads = new_accum

    perturbed_past = tuple(
        (k + gk, v + gv) for (k, v), (gk, gv) in zip(past, accum_grads)
    )
    return perturbed_past


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    """Plain greedy decoding from GPT-2.  Used to produce y_+ (non-toxic)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.lm.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)


def generate_pplm_toxic(
    model,
    tokenizer,
    classifier,
    prompt: str,
    max_new_tokens: int,
    device,
    cfg: PPLMConfig,
) -> str:
    """PPLM-steered toxic continuation.  Used to produce y_- (toxic).

    Uses the standard PPLM trick of mixing perturbed and unperturbed logits:
        p_final = softmax(logits_perturbed)^gamma * softmax(logits_unperturbed)^(1-gamma)
    realized by ``gm_scale`` weighting on the *probability* level.
    """
    model.lm.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = ids
    past = None

    for _ in range(max_new_tokens):
        # 1. forward to get current `past` and last hidden state
        out = model.lm(
            input_ids=generated if past is None else generated[:, -1:],
            past_key_values=past,
            output_hidden_states=True,
            use_cache=True,
        )
        past = out.past_key_values
        last_hidden = out.hidden_states[-1]  # [B, T, d]

        # 2. PPLM perturbation toward toxic direction
        if past is not None and len(past) > 0:
            past = perturb_past(past, last_hidden, classifier, cfg)

        # 3. unperturbed logits (re-run forward on last token with new past)
        out2 = model.lm(
            input_ids=generated[:, -1:],
            past_key_values=past,
            use_cache=True,
        )
        logits_perturbed = out2.logits[:, -1, :]
        # Mix with unperturbed logits via gm_scale
        logits_unpert = out.logits[:, -1, :]
        probs = F.softmax(logits_perturbed, dim=-1) ** cfg.gm_scale * F.softmax(
            logits_unpert, dim=-1
        ) ** (1.0 - cfg.gm_scale)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_id = probs.argmax(dim=-1, keepdim=True)  # [B, 1]
        generated = torch.cat([generated, next_id], dim=-1)
        past = out2.past_key_values

    return tokenizer.decode(generated[0][ids.shape[1] :], skip_special_tokens=True)
