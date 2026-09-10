"""Model wrappers used in LCA-on-the-Line.

Per the paper's Appendix A and the addendum, vision-only models (VMs) are
loaded via torchvision and vision-language models (VLMs) via the OpenCLIP
and OpenAI CLIP modules. Each wrapper exposes:

    forward_features(x) -> (B, D)   # penultimate features (M(X))
    forward(x)          -> (B, K)   # logits over K = 1000 ImageNet classes

Following the addendum, "M(X)" is the last hidden layer immediately before
the linear classifier (FC) is applied — this matches the convention used in
the linear-probing experiments of §4.3.2.

Verified reference (CrossRef via paper_search):
    Miller et al., "Accuracy on the Line", ICML 2021. (Table 3 baseline.)
"""

from __future__ import annotations

import importlib
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vision Models (torchvision) — paper §A lists 36 backbones
# ---------------------------------------------------------------------------


_VM_FACTORY = {
    "alexnet": "alexnet",
    "convnext_tiny": "convnext_tiny",
    "densenet121": "densenet121",
    "densenet161": "densenet161",
    "densenet169": "densenet169",
    "densenet201": "densenet201",
    "efficientnet_b0": "efficientnet_b0",
    "googlenet": "googlenet",
    "inception_v3": "inception_v3",
    "mnasnet0_5": "mnasnet0_5",
    "mnasnet0_75": "mnasnet0_75",
    "mnasnet1_0": "mnasnet1_0",
    "mnasnet1_3": "mnasnet1_3",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "regnet_y_1_6gf": "regnet_y_1_6gf",
    "wide_resnet101_2": "wide_resnet101_2",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
    "shufflenet_v2_x2_0": "shufflenet_v2_x2_0",
    "squeezenet1_0": "squeezenet1_0",
    "squeezenet1_1": "squeezenet1_1",
    "swin_b": "swin_b",
    "vgg11": "vgg11",
    "vgg13": "vgg13",
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "vgg11_bn": "vgg11_bn",
    "vgg13_bn": "vgg13_bn",
    "vgg16_bn": "vgg16_bn",
    "vgg19_bn": "vgg19_bn",
    "vit_b_32": "vit_b_32",
    "vit_l_32": "vit_l_32",
}


def _strip_classifier(model: nn.Module, name: str) -> Tuple[nn.Module, nn.Module, int]:
    """Replace the final classification layer with `nn.Identity`, and return
    (penultimate_extractor, original_classifier, feature_dim).

    This is implemented per torchvision's documented head names. We keep the
    original classifier so callers can reconstruct the full forward pass when
    they only need logits.
    """
    # torchvision conventions: ResNets / RegNet / GoogLeNet -> .fc
    #                          DenseNet / MNASNet / MobileNetV3 / VGG /
    #                          ConvNeXt / Swin / EfficientNet / SqueezeNet ->
    #                              .classifier (sometimes a Sequential)
    #                          ViT (vit_b_32, vit_l_32) -> .heads.head
    fc_attrs = ["fc", "classifier", "heads"]
    for attr in fc_attrs:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if attr == "fc":
                feat_dim = head.in_features
                setattr(model, attr, nn.Identity())
                return model, head, feat_dim
            if attr == "classifier":
                # For VGG/AlexNet/ConvNeXt/etc. the classifier is a Sequential
                # whose final Linear layer is the FC head.
                if isinstance(head, nn.Sequential):
                    last_linear_idx = max(
                        i for i, m in enumerate(head) if isinstance(m, nn.Linear)
                    )
                    last_linear = head[last_linear_idx]
                    feat_dim = last_linear.in_features
                    new_seq = nn.Sequential(*list(head.children())[:last_linear_idx])
                    setattr(model, attr, new_seq)
                    return model, last_linear, feat_dim
                if isinstance(head, nn.Linear):
                    feat_dim = head.in_features
                    setattr(model, attr, nn.Identity())
                    return model, head, feat_dim
            if attr == "heads":
                head_layer = getattr(head, "head", head)
                if isinstance(head_layer, nn.Linear):
                    feat_dim = head_layer.in_features
                    head.head = nn.Identity()
                    return model, head_layer, feat_dim
    raise ValueError(f"Could not strip classifier from torchvision model {name}")


class VMWrapper(nn.Module):
    """torchvision VM wrapper exposing penultimate features.

    Following the addendum: M(X) is the last hidden layer before the FC.
    """

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        if name not in _VM_FACTORY:
            raise KeyError(f"Unknown torchvision VM: {name}")
        models_mod = importlib.import_module("torchvision.models")
        ctor = getattr(models_mod, _VM_FACTORY[name])
        try:
            base = ctor(weights="DEFAULT" if pretrained else None)
        except TypeError:
            base = ctor(pretrained=pretrained)
        self.backbone, self.classifier_head, self.feature_dim = _strip_classifier(
            base, name
        )
        self.name = name

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier_head(feats)


def build_vm(name: str, pretrained: bool = True) -> VMWrapper:
    return VMWrapper(name=name, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Vision-Language Models — CLIP (OpenAI) and OpenCLIP (paper §A: 39 VLMs)
# ---------------------------------------------------------------------------


class ZeroShotClassifier(nn.Module):
    """Zero-shot classifier for VLMs (CLIP / OpenCLIP).

    Builds a class-name embedding matrix W in R^{K x D} from text prompts and
    classifies images by computing logits = scale * image_feat @ W.T.

    Prompt-engineering ablations from §4.3.3 are supported through the
    `prompt_template` argument:
        - default:        "a photo of a {classname}."
        - hierarchy:      "{classname}, which is a type of {parent}, which is
                           a type of {grandparent}."
        - shuffle_parent: same template but parent/grandparent randomly
                           sampled (control from §4.3.3).
    """

    def __init__(
        self,
        backbone: str,
        pretrained: str = "openai",
        provider: str = "open_clip",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.provider = provider
        self.device = device
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.text_features: Optional[torch.Tensor] = None
        self.logit_scale: Optional[torch.Tensor] = None
        self._load()

    def _load(self) -> None:
        if self.provider == "open_clip":
            oc = importlib.import_module("open_clip")
            self.model, _, self.preprocess = oc.create_model_and_transforms(
                self.backbone_name, pretrained=self.pretrained, device=self.device
            )
            self.tokenizer = oc.get_tokenizer(self.backbone_name)
            self.logit_scale = self.model.logit_scale.exp().detach()
        elif self.provider == "clip":
            clip = importlib.import_module("clip")
            self.model, self.preprocess = clip.load(
                self.backbone_name, device=self.device
            )
            self.tokenizer = clip.tokenize
            self.logit_scale = self.model.logit_scale.exp().detach()
        else:
            raise ValueError(f"Unknown VLM provider: {self.provider}")

    @torch.no_grad()
    def build_classifier(
        self,
        class_names: Sequence[str],
        prompt_template: str = "a photo of a {classname}.",
        parents: Optional[Sequence[str]] = None,
        grandparents: Optional[Sequence[str]] = None,
    ) -> None:
        """Build W by averaging text embeddings of prompts per class.

        For §4.3.3 'hierarchy' prompts, supply per-class parent/grandparent
        names; the template should contain {classname}, {parent}, {grandparent}.
        """
        prompts: List[str] = []
        for i, c in enumerate(class_names):
            kw = {"classname": c}
            if parents is not None:
                kw["parent"] = parents[i]
            if grandparents is not None:
                kw["grandparent"] = grandparents[i]
            prompts.append(prompt_template.format(**kw))
        toks = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        self.text_features = feats

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.text_features is None:
            raise RuntimeError("Call build_classifier(class_names) before forward.")
        img_feat = self.forward_features(images)
        # Logits for cross-entropy / softmax — temperature-scaled cosine sim.
        return self.logit_scale * img_feat @ self.text_features.t()


def build_vlm(spec, device: str = "cuda") -> ZeroShotClassifier:
    """Build a VLM either from a CLIP architecture name (str) or an OpenCLIP
    spec dict {arch, pretrained}."""
    if isinstance(spec, str):
        return ZeroShotClassifier(backbone=spec, provider="clip", device=device)
    if isinstance(spec, dict):
        return ZeroShotClassifier(
            backbone=spec["arch"],
            pretrained=spec.get("pretrained", "openai"),
            provider="open_clip",
            device=device,
        )
    raise TypeError(f"Unsupported VLM spec: {spec}")


# ---------------------------------------------------------------------------
# Linear-probe head used in §4.3.2 experiments
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer trained on frozen features.

    Used in §4.3.2: fed the penultimate features from a frozen VM backbone
    (as defined in the addendum: M(X) = features before FC) and trained for
    50 epochs with AdamW + cosine schedule + warmup (paper Appendix E.5).
    """

    def __init__(self, in_dim: int, num_classes: int = 1000) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
