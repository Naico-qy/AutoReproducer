"""Adversarial attack implementations for FARE training and evaluation.

* pgd.py    : 10-step PGD used during FARE training (Sec. B.1).
* apgd.py   : APGD (Croce & Hein, 2020) for evaluation (Sec. B.10).
              The implementation is structurally based on the reference at
              https://github.com/fra31/robust-finetuning (per addendum).
"""

from .pgd import pgd_fare, pgd_tecoa
from .apgd import apgd_attack

__all__ = ["pgd_fare", "pgd_tecoa", "apgd_attack"]
