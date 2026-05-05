# -*- coding: utf-8 -*-
"""Tests for drophet_utils — these run without torch/rdkit installed."""

import pandas as pd
import pytest

from drophet_utils import pair_key, pair_keys, seed_everything


class TestPairKey:
    def test_order_invariance(self):
        assert pair_key("Aspirin", "Warfarin") == pair_key("Warfarin", "Aspirin")

    def test_case_insensitive(self):
        assert pair_key("Aspirin", "Warfarin") == pair_key("aspirin", "warfarin")
        assert pair_key("Aspirin", "Warfarin") == pair_key("ASPIRIN", "WARFARIN")

    def test_whitespace_stripped(self):
        assert pair_key(" Aspirin ", "Warfarin") == pair_key("Aspirin", "Warfarin")

    def test_monotherapy_empty_drug2(self):
        # Different drugs alone should still produce different keys
        assert pair_key("Aspirin", "") != pair_key("Warfarin", "")
        # And a mono "X + ''" must equal "'' + X"
        assert pair_key("Aspirin", "") == pair_key("", "Aspirin")

    def test_nan_handled_like_empty(self):
        assert pair_key("Aspirin", float("nan")) == pair_key("Aspirin", "")
        assert pair_key("Aspirin", None) == pair_key("Aspirin", "")

    def test_pair_keys_vectorized(self):
        df = pd.DataFrame({"a": ["Aspirin", "Warfarin"], "b": ["Warfarin", "Aspirin"]})
        keys = pair_keys(df["a"], df["b"])
        assert keys[0] == keys[1]


class TestSeedEverything:
    def test_python_random_deterministic(self):
        import random
        seed_everything(42)
        seq1 = [random.random() for _ in range(5)]
        seed_everything(42)
        seq2 = [random.random() for _ in range(5)]
        assert seq1 == seq2

    def test_numpy_deterministic(self):
        np = pytest.importorskip("numpy")
        seed_everything(42)
        a = np.random.rand(5).tolist()
        seed_everything(42)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_torch_deterministic_if_available(self):
        torch = pytest.importorskip("torch")
        seed_everything(42)
        a = torch.randn(5).tolist()
        seed_everything(42)
        b = torch.randn(5).tolist()
        assert a == b
