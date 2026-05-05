# -*- coding: utf-8 -*-
"""Tests for the GNN's smiles_to_graph helper. Skipped if torch / rdkit / pyg
are unavailable in the test environment."""

import importlib.util
from pathlib import Path

import pytest

# Heavy deps may be absent in CI's lightweight job — skip rather than fail.
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("rdkit")


def _load_gnn_module():
    spec = importlib.util.spec_from_file_location(
        "gnn", Path(__file__).resolve().parent.parent / "019_train_gnn_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gnn():
    return _load_gnn_module()


class TestSmilesToGraph:
    def test_valid_smiles(self, gnn):
        g = gnn.smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        assert g.x.shape[0] > 0
        assert g.x.shape[1] == 6
        assert g.edge_index.shape[0] == 2

    def test_empty_smiles_returns_dummy(self, gnn):
        # Empty SMILES is the monotherapy case — must return a 1-node dummy
        # so PyG batching still works (vs raising and crashing the loop).
        g = gnn.smiles_to_graph("")
        assert g.num_nodes == 1
        assert g.x.shape == (1, 6)

    def test_whitespace_only_returns_dummy(self, gnn):
        g = gnn.smiles_to_graph("   ")
        assert g.num_nodes == 1

    def test_nan_returns_dummy(self, gnn):
        import pandas as pd
        g = gnn.smiles_to_graph(float("nan"))
        assert g.num_nodes == 1

    def test_invalid_smiles_returns_dummy(self, gnn):
        g = gnn.smiles_to_graph("not_a_valid_smiles_!!!")
        assert g.num_nodes == 1


class TestGNNModelOutputDim:
    def test_default_output_dim_is_one(self, gnn):
        m = gnn.GNNModel()
        assert m.out.out_features == 1

    def test_multitask_output_dim(self, gnn):
        m = gnn.GNNModel(n_outputs=7)
        assert m.out.out_features == 7

    def test_dropout_layers_present(self, gnn):
        import torch
        m = gnn.GNNModel()
        dropout_modules = [mod for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]
        assert len(dropout_modules) >= 2, "Expected at least 2 Dropout layers for MC-Dropout"
