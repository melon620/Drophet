# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 3: Advanced GNN Modeling (Graph Neural Networks - Clinical Research Edition)

This script implements a Siamese Graph Isomorphism Network (GIN) with:
1. Training Pipeline: Symmetrized GIN with early stopping.
2. Inference Engine: Class-based predictor with AUTOMATIC SMILES lookup.
3. Interactive Mode: Continuous loop for drug-pair screening.
4. Scaler Persistence: Using pickle to ensure consistent percentage outputs.
Requirement: training_matrix_refined_for_gnn.csv
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
import sys

# --- 1. Enhanced Graph & Descriptor Engine ---

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

    xs = []
    for atom in mol.GetAtoms():
        x = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
            float(atom.GetHybridization()),
            atom.GetNumRadicalElectrons(),
        ]
        xs.append(x)
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

# --- 2. Advanced GIN Architecture ---

class GNNModel(torch.nn.Module):
    def __init__(self, node_features=6, desc_features=5, hidden_channels=64):
        super(GNNModel, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(node_features, hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        self.ln1 = LayerNorm(hidden_channels)
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.ln2 = LayerNorm(hidden_channels)
        self.fc1 = torch.nn.Linear((hidden_channels * 2) + (desc_features * 2), 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)

    def forward_one(self, g):
        x, edge_index, batch = g.x, g.edge_index, g.batch
        x = F.relu(self.ln1(self.conv1(x, edge_index), batch))
        x = F.relu(self.ln2(self.conv2(x, edge_index), batch))
        return global_add_pool(x, batch) + global_mean_pool(x, batch)

    def forward(self, g1, g2, d1, d2):
        emb1 = self.forward_one(g1); emb2 = self.forward_one(g2)
        combined = torch.cat([emb1, emb2, d1, d2], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- 3. Clinical Inference Tool with Name Lookup ---

class DDIInferenceTool:
    """A wrapper for using the trained model with automatic drug name to SMILES lookup."""
    def __init__(self, model_path='ddi_gnn_best_model.pth', scaler_path='target_scaler.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNNModel().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        self.model.eval()

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler pkl not found at {scaler_path}")

    def fetch_smiles(self, drug_name):
        """Fetches canonical SMILES from PubChem API for a given drug name."""
        try:
            name = drug_name.strip()
            encoded = urllib.parse.quote(name)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                smiles = res.text.strip()
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not find SMILES for '{drug_name}': {e}")
        return None

    def get_risk_tier(self, incidence):
        if incidence < 5.0: return "🟢 Low Risk"
        if incidence < 20.0: return "🟡 Moderate Risk"
        return "🔴 High Risk"

    def predict(self, smiles1, smiles2):
        """Internal prediction using SMILES."""
        g1 = smiles_to_graph(smiles1)
        g2 = smiles_to_graph(smiles2)
        d1 = torch.tensor([get_descriptors(smiles1)], dtype=torch.float).to(self.device)
        d2 = torch.tensor([get_descriptors(smiles2)], dtype=torch.float).to(self.device)

        bg1 = Batch.from_data_list([g1]).to(self.device) if g1 else None
        bg2 = Batch.from_data_list([g2]).to(self.device) if g2 else None

        with torch.no_grad():
            output = self.model(bg1, bg2, d1, d2)
            incidence = self.scaler.inverse_transform(output.cpu().numpy())[0][0]
            incidence = max(0.0, incidence)

        return {
            "predicted_incidence": f"{incidence:.2f}%",
            "risk_tier": self.get_risk_tier(incidence)
        }

    def predict_from_names(self, name1, name2):
        """The primary user-facing method: takes drug names, fetches SMILES, and predicts."""
        s1 = self.fetch_smiles(name1)
        s2 = self.fetch_smiles(name2)

        if not s1 or not s2:
            missing = []
            if not s1: missing.append(name1)
            if not s2: missing.append(name2)
            return {"error": f"Could not find valid chemical structures for: {', '.join(missing)}"}

        results = self.predict(s1, s2)
        results["drug1_smiles"] = s1
        results["drug2_smiles"] = s2
        return results

# --- 4. Main Workflow ---

def pair_collate(batch):
    g1, g2, d1, d2, t = zip(*batch)
    return Batch.from_data_list(g1), Batch.from_data_list(g2), torch.stack(d1), torch.stack(d2), torch.stack(t)

def train_pipeline():
    print("--- Phase 5: GIN Training Pipeline Started ---")
    input_file = 'training_matrix_refined_for_gnn.csv'
    if not os.path.exists(input_file):
        print("❌ Error: Refined matrix not found. Run Script 018 first.")
        return

    df = pd.read_csv(input_file)
    target_name = [c for c in df.columns if c.startswith('Target_')][0]

    target_scaler = StandardScaler()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    target_scaler.fit(train_df[target_name].values.reshape(-1, 1))

    with open('target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    print("📁 Target scaler saved to target_scaler.pkl")

    train_loader = DataLoader(DDIPairDataset(train_df, target_name, target_scaler, augment=True), batch_size=8, shuffle=True, collate_fn=pair_collate)
    test_loader = DataLoader(DDIPairDataset(test_df, target_name, target_scaler, augment=False), batch_size=8, collate_fn=pair_collate)

    model = GNNModel().to('cpu')
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
            if epoch % 50 == 0: print(f"   Epoch {epoch} | Best MAE: {best_mae:.4f}")

    torch.save(best_state, 'ddi_gnn_best_model.pth')
    print("✅ Training complete. Model weights saved.")

if __name__ == "__main__":
    # Artifact Check
    model_exists = os.path.exists('ddi_gnn_best_model.pth')
    scaler_exists = os.path.exists('target_scaler.pkl')

    if not (model_exists and scaler_exists):
        print("🔍 Missing artifacts. Starting training...")
        train_pipeline()
    else:
        print("💎 Found existing model and scaler.")

    print("\n" + "="*50)
    print("🏥 CLINICAL DDI RISK PREDICTION TOOL (Interactive)")
    print("Type 'exit' or 'quit' to stop, or press Ctrl+C.")
    print("="*50)

    try:
        tool = DDIInferenceTool()

        while True:
            print("\n" + "-"*30)
            name_a = input("Enter Drug 1 Name (or 'exit'): ").strip()
            if name_a.lower() in ['exit', 'quit']: break

            name_b = input("Enter Drug 2 Name: ").strip()
            if name_b.lower() in ['exit', 'quit']: break

            if not name_a or not name_b:
                print(" Please enter both drug names.")
                continue

            print(f"🔍 Analyzing: {name_a} + {name_b}...")
            result = tool.predict_from_names(name_a, name_b)

            if "error" in result:
                print(f" Error: {result['error']}")
            else:
                print(f" SMILES 1: {result['drug1_smiles']}")
                print(f" SMILES 2: {result['drug2_smiles']}")
                print(f" Predicted Incidence: {result['predicted_incidence']}")
                print(f"  Risk Tier: {result['risk_tier']}")
            print("-"*30)

    except KeyboardInterrupt:
        print("\n\n Tool terminated by user.")
    except Exception as e:
        print(f"🚨 Inference Error: {e}")
