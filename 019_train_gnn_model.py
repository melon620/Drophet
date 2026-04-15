# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 5: Advanced GNN Fine-tuning (Pretrained GIN Version - Robust Loading)

This script implements a Siamese Graph Isomorphism Network (GIN) that:
1. Loads a Pretrained Backbone: Inherits chemical knowledge from Phase 4 (020).
2. Performs Fine-tuning: Adapts pretrained weights to clinical DDI toxicity.
3. Maintains Clinical Tooling: Includes inference engine with PubChem API lookup.
4. Robust Weight Loading: Handles prefix mismatches and structural shifts.

Requirement: training_matrix_refined_for_gnn.csv, gnn_pretrained_backbone.pth
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, LayerNorm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import copy
import pickle
import requests
import urllib.parse

# --- 1. Graph & Descriptor Engine ---

def get_descriptors(smiles):
    """Calculates global descriptors for fusion with graph embeddings."""
    if pd.isna(smiles) or smiles == "":
        return [0.0] * 5
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0] * 5
    return [
        Descriptors.MolWt(mol) / 1000.0,
        Descriptors.MolLogP(mol) / 10.0,
        Descriptors.TPSA(mol) / 200.0,
        float(Descriptors.NumHDonors(mol)) / 10.0,
        float(Descriptors.NumHAcceptors(mol)) / 15.0
    ]

def smiles_to_graph(smiles):
    """Converts SMILES to Graph with robust node features."""
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    # Features: [AtomicNum, Degree, Charge, IsAromatic, Hybridization, RadicalElectrons]
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

class DDIPairDataset(torch.utils.data.Dataset):
    def __init__(self, df, target_name, target_scaler=None, augment=False):
        self.smiles1 = df['SMILES_1'].values
        self.smiles2 = df['SMILES_2'].values
        self.augment = augment
        self.desc1 = np.array([get_descriptors(s) for s in self.smiles1])
        self.desc2 = np.array([get_descriptors(s) for s in self.smiles2])
        y = df[target_name].values.reshape(-1, 1)
        self.targets = target_scaler.transform(y) if target_scaler else y

    def __len__(self):
        return len(self.targets) * 2 if self.augment else len(self.targets)

    def __getitem__(self, idx):
        meta_idx = idx if idx < len(self.targets) else idx - len(self.targets)
        is_swapped = idx >= len(self.targets)
        s1, s2 = (self.smiles1[meta_idx], self.smiles2[meta_idx]) if not is_swapped else (self.smiles2[meta_idx], self.smiles1[meta_idx])
        d1, d2 = (self.desc1[meta_idx], self.desc2[meta_idx]) if not is_swapped else (self.desc2[meta_idx], self.desc1[meta_idx])
        g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
        if g1 is None: g1 = Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
        if g2 is None: g2 = Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
        return g1, g2, torch.tensor(d1, dtype=torch.float), torch.tensor(d2, dtype=torch.float), torch.tensor(self.targets[meta_idx], dtype=torch.float)

# --- 2. Advanced GIN Architecture (Restored with Dropout for Compatibility) ---

class GINBackbone(torch.nn.Module):
    def __init__(self, node_features=6, hidden_channels=64):
        super(GINBackbone, self).__init__()
        # Restored to 4 layers (Linear, ReLU, Dropout, Linear) to match existing weights
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2), # Index 2
            torch.nn.Linear(hidden_channels, hidden_channels) # Index 3
        )
        self.conv1 = GINConv(nn1)
        self.ln1 = LayerNorm(hidden_channels)

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
        self.ln2 = LayerNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.ln1(self.conv1(x, edge_index), batch))
        x = F.relu(self.ln2(self.conv2(x, edge_index), batch))
        return global_add_pool(x, batch) + global_mean_pool(x, batch)

class GNNModel(torch.nn.Module):
    def __init__(self, node_features=6, desc_features=5, hidden_channels=64):
        super(GNNModel, self).__init__()
        self.backbone = GINBackbone(node_features, hidden_channels)
        input_dim = (hidden_channels * 2) + (desc_features * 2)
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)

    def forward(self, g1, g2, d1, d2):
        emb1 = self.backbone(g1.x, g1.edge_index, g1.batch)
        emb2 = self.backbone(g2.x, g2.edge_index, g2.batch)
        combined = torch.cat([emb1, emb2, d1, d2], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- 3. Clinical Inference Tool with Robust Loading ---

class DDIInferenceTool:
    def __init__(self, model_path='ddi_gnn_best_model.pth', scaler_path='target_scaler.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNNModel().to(self.device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle prefix mapping (if saved as flat but model expects 'backbone.')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('conv') or k.startswith('ln'):
                    new_state_dict[f'backbone.{k}'] = v
                else:
                    new_state_dict[k] = v

            try:
                self.model.load_state_dict(new_state_dict)
            except RuntimeError:
                # Fallback to loading whatever fits if architecture is slightly different
                self.model.load_state_dict(new_state_dict, strict=False)
                print("⚠️ Warning: Loaded state dict with strict=False due to minor architecture differences.")

        self.model.eval()
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

    def fetch_smiles(self, drug_name):
        try:
            name = drug_name.strip()
            encoded = urllib.parse.quote(name)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return res.text.strip()
        except: pass
        return None

    def predict_from_names(self, name1, name2):
        s1 = self.fetch_smiles(name1)
        s2 = self.fetch_smiles(name2)
        if not s1 or not s2: return {"error": "Missing structures."}

        g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
        d1 = torch.tensor([get_descriptors(s1)], dtype=torch.float).to(self.device)
        d2 = torch.tensor([get_descriptors(s2)], dtype=torch.float).to(self.device)
        bg1, bg2 = Batch.from_data_list([g1]).to(self.device), Batch.from_data_list([g2]).to(self.device)

        with torch.no_grad():
            output = self.model(bg1, bg2, d1, d2)
            inc = self.scaler.inverse_transform(output.cpu().numpy())[0][0]
            inc = max(0.0, inc)

        tier = "🟢 Low Risk" if inc < 5 else "🟡 Moderate Risk" if inc < 20 else "🔴 High Risk"
        return {"incidence": f"{inc:.2f}%", "tier": tier, "s1": s1, "s2": s2}

# --- 4. Training Pipeline ---

def pair_collate(batch):
    g1, g2, d1, d2, t = zip(*batch)
    return Batch.from_data_list(g1), Batch.from_data_list(g2), torch.stack(d1), torch.stack(d2), torch.stack(t)

def train_pipeline():
    print("--- Phase 5: GNN Fine-tuning Pipeline (Transfer Learning) ---")
    input_file = 'training_matrix_refined_for_gnn.csv'
    pretrain_file = 'gnn_pretrained_backbone.pth'

    if not os.path.exists(input_file): return
    df = pd.read_csv(input_file)
    target_name = [c for c in df.columns if c.startswith('Target_')][0]

    target_scaler = StandardScaler()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    target_scaler.fit(train_df[target_name].values.reshape(-1, 1))
    with open('target_scaler.pkl', 'wb') as f: pickle.dump(target_scaler, f)

    train_loader = DataLoader(DDIPairDataset(train_df, target_name, target_scaler, augment=True), batch_size=8, shuffle=True, collate_fn=pair_collate)
    test_loader = DataLoader(DDIPairDataset(test_df, target_name, target_scaler, augment=False), batch_size=8, collate_fn=pair_collate)

    model = GNNModel().to('cpu')

    if os.path.exists(pretrain_file):
        print(f"💎 Initializing with pretrained backbone: {pretrain_file}")
        pre_state = torch.load(pretrain_file)
        # Fix prefix if necessary for backbone loading
        fixed_pre_state = {k.replace('backbone.', ''): v for k, v in pre_state.items()}
        model.backbone.load_state_dict(fixed_pre_state, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    criterion = torch.nn.HuberLoss()
    best_mae, best_state = float('inf'), None

    for epoch in range(1, 201):
        model.train()
        for g1, g2, d1, d2, target in train_loader:
            optimizer.zero_grad(); out = model(g1, g2, d1, d2); loss = criterion(out, target); loss.backward(); optimizer.step()

        model.eval()
        total_mae = 0
        with torch.no_grad():
            for g1, g2, d1, d2, target in test_loader:
                out = model(g1, g2, d1, d2)
                p = target_scaler.inverse_transform(out.numpy()).flatten()
                a = target_scaler.inverse_transform(target.numpy()).flatten()
                total_mae += mean_absolute_error(a, np.maximum(p, 0))

        avg_mae = total_mae / len(test_loader)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_state = copy.deepcopy(model.state_dict())
            if epoch % 50 == 0: print(f"   Epoch {epoch:03d} | Best Val MAE: {best_mae:.4f}")

    torch.save(best_state, 'ddi_gnn_best_model.pth')
    print("✅ Fine-tuning complete.")

if __name__ == "__main__":
    if not (os.path.exists('ddi_gnn_best_model.pth') and os.path.exists('target_scaler.pkl')):
        train_pipeline()

    print("\n" + "="*50)
    print("🏥 PROJECT DROPHET: CLINICAL DDI SCREENING TOOL")
    print("Type 'exit' to quit.")
    print("="*50)

    try:
        tool = DDIInferenceTool()
        while True:
            n1 = input("\nEnter Drug 1: ").strip()
            if n1.lower() == 'exit': break
            n2 = input("Enter Drug 2: ").strip()
            if n2.lower() == 'exit': break
            res = tool.predict_from_names(n1, n2)
            if "error" in res: print(f"❌ {res['error']}")
            else: print(f"📊 Result for {n1} + {n2}: {res['incidence']} | {res['tier']}\n")
    except KeyboardInterrupt: pass
    print("\n👋 Goodbye!")
