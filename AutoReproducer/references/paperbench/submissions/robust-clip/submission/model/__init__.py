"""Model package for FARE: Robust CLIP via unsupervised adversarial fine-tuning.

References
----------
Schlarmann, Singh, Croce, Hein.
"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for
Robust Large Vision-Language Models." ICML 2024.

The trainable component is exactly the CLIP vision tower (image encoder).
The text tower is unused for FARE training because the loss in Eq. (3) only
involves image embeddings: L_FARE(x) = max_{||z-x||<=eps} ||phi_FT(z) - phi_Org(x)||_2^2.
"""

from .architecture import FAREModel, fare_loss
from .clip_loader import load_clip_vision, CLIPVisionWrapper

__all__ = ["FAREModel", "fare_loss", "load_clip_vision", "CLIPVisionWrapper"]
