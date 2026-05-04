# -*- coding: utf-8 -*-
"""
drophet_utils — small shared helpers used across the training scripts.

Currently exposes:
- seed_everything(seed): deterministic-ish seeding for python/numpy/torch.
- pair_key(a, b): canonical, order-invariant key for a drug pair (used for
  group-aware train/test splits so the same pair never leaks across splits).

Kept as a single flat module on purpose so the existing numbered scripts can
import it without restructuring the project.
"""

from __future__ import annotations

import os
import random
from typing import Iterable, Optional


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and (if available) torch RNGs.

    Why this exists: the GNN scripts previously relied only on
    ``train_test_split(random_state=42)`` which leaves model init,
    DataLoader shuffling, and CUDA ops nondeterministic. Re-running training
    produced different weights every time.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism. cuDNN benchmark off slows training but
        # gives reproducible kernels for small models like ours.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def pair_key(a: Optional[str], b: Optional[str]) -> str:
    """Order-invariant key for a (Drug_A, Drug_B) pair.

    "Aspirin + Warfarin" and "Warfarin + Aspirin" must hash to the same key
    so they cannot land in different splits. Empty / NaN entries are folded
    to the empty string for monotherapy rows.
    """
    sa = "" if a is None else str(a).strip().lower()
    sb = "" if b is None else str(b).strip().lower()
    if sa == "nan":
        sa = ""
    if sb == "nan":
        sb = ""
    lo, hi = sorted([sa, sb])
    return f"{lo}||{hi}"


def pair_keys(drug1: Iterable, drug2: Iterable):
    return [pair_key(a, b) for a, b in zip(drug1, drug2)]
