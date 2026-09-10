"""Fine-tune the binary source-vs-target classifier p_phi(y | x_t).

Per addendum.md:
    "These pre-trained models were fine-tuned by modifying the last layer
     to output two classes to classify whether images where coming from
     the source or the target dataset. To fine-tune the model the authors
     used Adam as the optimizer with a learning rate of 1e-4, a batch
     size of 64, and trained for 300 iterations."

The classifier is conditioned on a uniformly-sampled diffusion timestep t
and trained on noised images x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε.

Usage:
    python -m scripts.train_classifier \
        --config configs/default.yaml \
        --source ./datasets/ffhq \
        --target ./datasets/10shot_sunglasses \
        --out ./outputs/classifier.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn.functional as F

# allow running both as module and as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, set_seed
from data.loader import build_classifier_loader
from model.classifier import BinaryNoiseClassifier
from model.schedule import GaussianDiffusion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train binary source-vs-target classifier")
    p.add_argument("--config", required=True)
    p.add_argument("--source", required=True, help="path to source images")
    p.add_argument("--target", required=True, help="path to target images")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(
        args.device or cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    image_size = cfg["image_size"]
    diff = GaussianDiffusion(
        T=cfg["diffusion"]["num_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=device,
    )

    classifier = BinaryNoiseClassifier(
        in_channels=cfg["image_channels"],
        image_size=image_size,
        num_classes=cfg["classifier"]["num_classes"],
    ).to(device)

    loader = build_classifier_loader(
        source_root=args.source,
        target_root=args.target,
        batch_size=cfg["classifier"]["batch_size"],
        image_size=image_size,
        num_workers=cfg["data"]["num_workers"],
    )

    opt = torch.optim.Adam(classifier.parameters(), lr=cfg["classifier"]["lr"])

    iters = cfg["classifier"]["iterations"]
    classifier.train()

    step = 0
    while step < iters:
        for x0, y in loader:
            x0 = x0.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.randint(
                0,
                cfg["diffusion"]["num_timesteps"],
                (x0.size(0),),
                device=device,
                dtype=torch.long,
            )
            x_t = diff.q_sample(x0, t)

            logits = classifier(x_t, t)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                print(
                    f"[classifier] iter {step:04d}  loss={loss.item():.4f}  acc={acc:.3f}",
                    flush=True,
                )
            step += 1
            if step >= iters:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {"state_dict": classifier.state_dict(), "config": cfg["classifier"]}, args.out
    )
    print(f"[classifier] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
