# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Project Drophet)
Phase 5: Advanced GNN Fine-tuning (Size-Invariant Pure Regression)

This version surgically fixes the Out-of-Distribution (OOD) size bug and monotherapy crashes:
1. Feature Normalization: Prevents large molecules (like Ketoconazole) from causing
   massive negative logits via global_add_pool by standardizing the concatenated vector.
2. Monotherapy Support: Explicitly handles empty Drug 2 inputs for baseline risk assessment.
3. MLP Capacity: Restored to 128->64 to handle normalized feature complexity.
4. Pure Linear Output & MSE Loss: Maintained for continuous, mathematically sound gradients.

Requirements: training_matrix_augmented.csv, gnn_pretrained_backbone.pth
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import os
import copy
import requests
import urllib.parse
import warnings

from drophet_utils import seed_everything, pair_keys

warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

SEED = 42
seed_everything(SEED)

# --- 0. Control Flags ---
FORCE_RETRAIN = True

# --- 1. Graph & Descriptor Engine ---

def get_descriptors(smiles):
    if pd.isna(smiles) or smiles.strip() == "": return [0.0] * 5
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [0.0] * 5
    return [
        Descriptors.MolWt(mol) / 1000.0,
        Descriptors.MolLogP(mol) / 10.0,
        Descriptors.TPSA(mol) / 200.0,
        float(Descriptors.NumHDonors(mol)) / 10.0,
        float(Descriptors.NumHAcceptors(mol)) / 15.0
    ]

def smiles_to_graph(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        # Return a robust dummy graph for empty inputs (Monotherapy support)
        return Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)

    mol = Chem.MolFromSmiles(smiles)
    if not mol: return Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)

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
    def __init__(self, df, target_cols=None):
        self.smiles1 = df['SMILES_1'].values
        self.smiles2 = df['SMILES_2'].values
        self.desc1 = np.array([get_descriptors(s) for s in self.smiles1])
        self.desc2 = np.array([get_descriptors(s) for s in self.smiles2])

        if target_cols is None:
            target_cols = sorted([c for c in df.columns if c.startswith('Target_')])
        if not target_cols:
            raise ValueError("No Target_ columns found in dataset.")
        self.target_cols = list(target_cols)

        # Multi-task: keep one target per AE category instead of collapsing
        # them with df.max(axis=1). Max-aggregation throws away the structure
        # the model is trying to learn — every pair has the same "primary"
        # signal regardless of which AE actually drove the % rate.
        self.targets = (df[self.target_cols].values.astype(np.float32)) / 100.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        s1, s2 = self.smiles1[idx], self.smiles2[idx]
        d1, d2 = self.desc1[idx], self.desc2[idx]
        g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
        return g1, g2, torch.tensor(d1, dtype=torch.float), torch.tensor(d2, dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)

# --- 2. Advanced GIN Architecture (Size-Invariant Restored) ---

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

class GNNModel(torch.nn.Module):
    def __init__(self, node_features=6, desc_features=5, hidden_channels=64, n_outputs=1):
        super(GNNModel, self).__init__()
        self.backbone = GINBackbone(node_features, hidden_channels)
        input_dim = (hidden_channels * 2) + (desc_features * 2)

        # [CRITICAL FIX] LayerNorm neutralizes the magnitude explosion from global_add_pool for large molecules
        self.norm = torch.nn.LayerNorm(input_dim)

        # Restored capacity to learn complex synergies
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, g1, g2, d1, d2):
        emb1 = self.backbone(g1.x, g1.edge_index, g1.batch)
        emb2 = self.backbone(g2.x, g2.edge_index, g2.batch)

        emb_add = emb1 + emb2
        emb_diff = torch.abs(emb1 - emb2)
        d_add = d1 + d2
        d_diff = torch.abs(d1 - d2)

        combined = torch.cat([emb_add, emb_diff, d_add, d_diff], dim=1)

        # Standardize features before MLP to ensure size-invariance
        x = self.norm(combined)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.out(x)

# --- 3. Production-Ready Inference Tool ---

class DDIInferenceTool:
    def __init__(self, model_path='ddi_gnn_best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_ready = False
        self.model = None
        self.target_cols = None

        if not os.path.exists(model_path):
            print("⚠️ Inference Tool initialization failed: Missing model artifacts.")
            return

        loaded = torch.load(model_path, map_location=self.device)

        if isinstance(loaded, dict) and 'model_state' in loaded:
            # Multi-task format saved by train_pipeline (this file).
            state_dict = loaded['model_state']
            self.target_cols = loaded.get('target_cols')
            n_outputs = int(loaded.get('n_outputs') or
                            (len(self.target_cols) if self.target_cols else 1))
        else:
            # Legacy format: raw state_dict, single output head.
            state_dict = loaded
            n_outputs = 1
            if 'out.weight' in state_dict:
                n_outputs = state_dict['out.weight'].shape[0]

        self.model = GNNModel(n_outputs=n_outputs).to(self.device)

        is_modern = any(k.startswith('backbone.') for k in state_dict.keys())
        if is_modern:
            new_state_dict = state_dict
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('conv') or (k.startswith('ln') and len(v.shape) > 0 and v.shape[0] == 64):
                    new_state_dict[f'backbone.{k}'] = v
                else:
                    new_state_dict[k] = v

        try: self.model.load_state_dict(new_state_dict)
        except RuntimeError: self.model.load_state_dict(new_state_dict, strict=False)

        self.model.eval()
        self.is_ready = True

    def fetch_smiles(self, drug_name):
        try:
            name = drug_name.strip()
            if name == "": return "" # Fast exit for empty inputs
            encoded = urllib.parse.quote(name)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
            res = requests.get(url, timeout=10)
            if res.status_code == 200: return res.text.strip()
        except Exception: pass
        return None

    def predict_from_names(self, name1, name2):
        if not self.is_ready: return {"error": "Model not loaded properly."}

        # [CRITICAL FIX] Handle empty inputs elegantly for Monotherapy
        s1 = "" if name1.strip() == "" else self.fetch_smiles(name1)
        s2 = "" if name2.strip() == "" else self.fetch_smiles(name2)

        if s1 is None or s2 is None:
            return {"error": f"Missing structures via PubChem. Please check spelling."}

        if s1 == "" and s2 == "":
            return {"error": "Both drug inputs cannot be empty."}

        g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
        d1 = torch.tensor([get_descriptors(s1)], dtype=torch.float).to(self.device)
        d2 = torch.tensor([get_descriptors(s2)], dtype=torch.float).to(self.device)
        bg1, bg2 = Batch.from_data_list([g1]).to(self.device), Batch.from_data_list([g2]).to(self.device)

        with torch.no_grad():
            output = self.model(bg1, bg2, d1, d2)
            arr = np.clip(output.cpu().numpy().flatten() * 100.0, 0.0, 100.0)

        # Top-line incidence is the worst per-category prediction. Same coarse
        # semantics as the prior single-head max-aggregated model so existing
        # callers (Gradio app, CLI prompt) keep working, but we now also
        # surface the per-category numbers so users can see *which* AE drives
        # the risk tier instead of one opaque %.
        inc = float(arr.max()) if arr.size else 0.0

        breakdown = []
        if self.target_cols and len(self.target_cols) == arr.size:
            breakdown = sorted(
                [(c, float(p)) for c, p in zip(self.target_cols, arr)],
                key=lambda t: t[1], reverse=True,
            )

        tier = "🟢 Low Risk" if inc < 5 else "🟡 Moderate Risk" if inc < 20 else "🔴 High Risk"
        return {
            "incidence": f"{inc:.2f}%",
            "tier": tier,
            "s1": s1,
            "s2": s2,
            "breakdown": breakdown,
        }

# --- 4. Training Pipeline ---

def pair_collate(batch):
    g1, g2, d1, d2, t = zip(*batch)
    return Batch.from_data_list(g1), Batch.from_data_list(g2), torch.stack(d1), torch.stack(d2), torch.stack(t)

def train_pipeline():
    print("--- Phase 5: GNN Fine-tuning (Size-Invariant Pure Regression) ---")
    input_file = 'training_matrix_augmented.csv'
    if not os.path.exists(input_file): input_file = 'training_matrix_refined_for_gnn.csv'
    if not os.path.exists(input_file):
        print(f"❌ Error: Training dataset '{input_file}' missing.")
        return

    df = pd.read_csv(input_file)
    print(f"📊 Loading dataset with {len(df)} drug pairs...")

    # --- Pair-aware split ---
    # Previously: a plain random split could put "Aspirin+Warfarin" in train
    # and "Warfarin+Aspirin" (or a duplicate) in test, leaking labels.
    # We now group by an order-invariant pair key so the same pair never
    # appears on both sides of the split.
    if not {'Drug_1', 'Drug_2'}.issubset(df.columns):
        raise ValueError("Expected 'Drug_1' and 'Drug_2' columns for pair-aware split.")

    groups = pair_keys(df['Drug_1'].values, df['Drug_2'].values)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df, test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    # Sanity check: zero overlap of canonical pair keys between splits.
    train_keys = set(pair_keys(train_df['Drug_1'].values, train_df['Drug_2'].values))
    test_keys = set(pair_keys(test_df['Drug_1'].values, test_df['Drug_2'].values))
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"Pair leakage between train/test: {len(overlap)} overlapping keys.")
    print(f"   Split: {len(train_df)} train / {len(test_df)} test, no pair overlap.")

    target_cols = sorted([c for c in df.columns if c.startswith('Target_')])
    if not target_cols:
        raise ValueError("No Target_ columns found in training dataset.")
    n_outputs = len(target_cols)
    print(f"🎯 Multi-task heads: {n_outputs} AE categories ({', '.join(target_cols[:3])}{'...' if n_outputs > 3 else ''})")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(DDIPairDataset(train_df, target_cols), batch_size=8, shuffle=True,
                              collate_fn=pair_collate, generator=g)
    test_loader = DataLoader(DDIPairDataset(test_df, target_cols), batch_size=8, collate_fn=pair_collate)

    model = GNNModel(n_outputs=n_outputs).to('cpu')
    pretrain_file = 'gnn_pretrained_backbone.pth'
    if os.path.exists(pretrain_file):
        print(f"💎 Loading pretrained chemical backbone...")
        pre_state = torch.load(pretrain_file)
        model.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in pre_state.items()}, strict=False)

    # Adding a tiny weight_decay (1e-4) to prevent overfitting on the small dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # L1 (MAE) for both the training objective and the selection metric, so the
    # loss the optimiser minimises matches the metric we use to pick the best
    # epoch. Previously: trained on MSE, selected on MAE — those don't agree,
    # which means the epoch checkpoint we kept wasn't optimal for either.
    criterion = torch.nn.L1Loss()

    best_mae, best_state = float('inf'), None

    for epoch in range(1, 201):
        model.train()
        train_loss = 0.0
        for g1, g2, d1, d2, target in train_loader:
            optimizer.zero_grad()
            out = model(g1, g2, d1, d2)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        total_mae = 0.0
        n_batches = 0
        with torch.no_grad():
            for g1, g2, d1, d2, target in test_loader:
                out = model(g1, g2, d1, d2)
                p = np.maximum(out.numpy(), 0.0) * 100.0
                a = target.numpy() * 100.0
                total_mae += float(np.abs(p - a).mean())
                n_batches += 1

        avg_mae = total_mae / max(n_batches, 1)

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == 1:
            train_avg = train_loss / max(len(train_loader), 1)
            print(f"   Epoch {epoch:03d} | Train L1: {train_avg:.4f} | Val MAE: {avg_mae:.4f}% (Best: {best_mae:.4f}%)")

    # Save state + metadata so DDIInferenceTool knows what each output column
    # means and how many outputs the head produced. The legacy raw-state-dict
    # format is still read for backwards compatibility.
    torch.save({
        'model_state': best_state,
        'target_cols': target_cols,
        'n_outputs': n_outputs,
    }, 'ddi_gnn_best_model.pth')
    print("✅ Training complete. Artifacts saved successfully.")

if __name__ == "__main__":
    needs_training = FORCE_RETRAIN or not os.path.exists('ddi_gnn_best_model.pth')

    if needs_training:
        if FORCE_RETRAIN:
            print("\n🚀 [FORCE_RETRAIN] Wiping existing weights...")
            for f in ['ddi_gnn_best_model.pth', 'target_scaler.pkl']:
                if os.path.exists(f): os.remove(f)
        train_pipeline()

    print("\n" + "="*50)
    print("🏥 PROJECT DROPHET: CLINICAL DDI SCREENING ENGINE (v1.0)")
    print("="*50)

    try:
        tool = DDIInferenceTool()
        if not tool.is_ready:
            print("🚨 Exiting interactive mode due to initialization failure.")
            exit()

        while True:
            n1 = input("\nDrug 1 (or 'exit'): ").strip()
            if n1.lower() == 'exit': break
            if n1.lower() == 'retrain':
                train_pipeline()
                tool = DDIInferenceTool()
                continue

            n2 = input("Drug 2: ").strip()
            if n2.lower() == 'exit': break

            print(f"🔍 Fetching SMILES and analyzing {n1} + {n2}...")
            res = tool.predict_from_names(n1, n2)

            if "error" in res:
                print(f"❌ {res['error']}")
            else:
                print(f"🧪 SMILES 1: {res['s1']}")
                print(f"🧪 SMILES 2: {res['s2']}")
                print(f"📊 Result:   {res['incidence']} | {res['tier']}")
                breakdown = res.get('breakdown') or []
                if breakdown:
                    print("📋 Top categories (per-AE %):")
                    for col, pct in breakdown[:5]:
                        print(f"     - {col}: {pct:.2f}%")

    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user.")
    except Exception as e:
        print(f"\n🚨 Unexpected Execution Error: {e}")
    print("\n👋 Ready for deployment! Goodbye.")
