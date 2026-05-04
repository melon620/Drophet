# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 4: GNN Self-Supervised Pretraining

This script performs "Property-Guided Pretraining" on a larger diverse set
of drug-like molecules to build a robust structural backbone for the GNN.
Task: Predict 5-7 fundamental RDKit descriptors simultaneously.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader  # Updated to resolve deprecation warning
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, LayerNorm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import os
import copy

from drophet_utils import seed_everything

seed_everything(42)

# --- 1. Data Generation / Fetching ---

def get_extensive_descriptors(smiles):
    """Calculates a wide range of descriptors for the pretraining task."""
    if not isinstance(smiles, str): return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol)
    ]

def smiles_to_graph(smiles):
    """Standard SMILES to PyG Graph conversion."""
    if not isinstance(smiles, str): return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    xs = [[atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
           float(atom.GetIsAromatic()), float(atom.GetHybridization()),
           atom.GetNumRadicalElectrons()] for atom in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j]); edge_indices.append([j, i])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=x.size(0))

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list):
        self.data_list = []
        self.targets = []
        print(f"⌛ Processing {len(smiles_list)} molecules for pretraining...")

        for s in smiles_list:
            g = smiles_to_graph(s)
            d = get_extensive_descriptors(s)
            if g and d:
                self.data_list.append(g)
                self.targets.append(d)

        if not self.targets:
            raise ValueError("No valid molecules found for pretraining. Check your SMILES data.")

        self.targets = np.array(self.targets)
        self.scaler = StandardScaler()
        self.targets_scaled = self.scaler.fit_transform(self.targets)

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx], torch.tensor(self.targets_scaled[idx], dtype=torch.float)

# --- 2. GIN Backbone Architecture ---

class GINBackbone(torch.nn.Module):
    def __init__(self, node_features=6, hidden_channels=64):
        super(GINBackbone, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(node_features, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        self.ln1 = LayerNorm(hidden_channels)

        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.ln2 = LayerNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.ln1(self.conv1(x, edge_index), batch))
        x = F.relu(self.ln2(self.conv2(x, edge_index), batch))
        return global_add_pool(x, batch) + global_mean_pool(x, batch)

class PretrainModel(torch.nn.Module):
    def __init__(self, backbone, hidden_channels=64, output_dim=7):
        super(PretrainModel, self).__init__()
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, data):
        emb = self.backbone(data.x, data.edge_index, data.batch)
        return self.head(emb)

# --- 3. Pretraining Loop ---

def main():
    print("--- Phase 4: GNN Backbone Pretraining ---")

    # 3.1 Load a diverse set of drug SMILES
    if not os.path.exists('training_matrix_refined_for_gnn.csv'):
        print("❌ Error: Need base SMILES list. Run Script 018.")
        return

    base_df = pd.read_csv('training_matrix_refined_for_gnn.csv')

    # FIX: Filter out NaNs (floats) from the combined list
    all_smiles = base_df['SMILES_1'].tolist() + base_df['SMILES_2'].tolist()
    unique_smiles = [s for s in set(all_smiles) if isinstance(s, str) and s.strip() != ""]

    print(f"✅ Found {len(unique_smiles)} unique valid SMILES strings.")

    # Note: For actual pretraining, you want thousands of SMILES.
    # Here we use the unique set and encourage the model to learn their intrinsic properties.
    dataset = PretrainDataset(unique_smiles)
    # Using the updated DataLoader from torch_geometric.loader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    backbone = GINBackbone(node_features=6, hidden_channels=64)
    model = PretrainModel(backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    print(f"Training Backbone on {len(unique_smiles)} unique structures...")
    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for data, target in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"   Epoch {epoch:03d} | Pretrain Loss: {total_loss/len(loader):.4f}")

    # Save ONLY the backbone weights
    torch.save(backbone.state_dict(), 'gnn_pretrained_backbone.pth')
    print("✅ Pretrained backbone saved to gnn_pretrained_backbone.pth")

if __name__ == "__main__":
    main()
