# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Project Drophet)
Phase 6: N-Side Drug Interaction Model

Extends the 2-drug GNN (Script 019) to handle arbitrary n-drug combinations:
- Each drug becomes a node encoded by the frozen GINBackbone
- A complete interaction graph connects every drug pair with bidirectional edges
- Two rounds of graph attention let drugs communicate via their pairwise features
- Global pooling aggregates all drug representations into a single risk score

Edge attention weights identify which drug pair drives the combined risk.

Requirements: training_matrix_refined_for_gnn.csv, gnn_pretrained_backbone.pth
"""

import copy
import os
import time
import urllib.parse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GATConv, GINConv, LayerNorm, global_add_pool,
                                 global_mean_pool)

from drophet_utils import seed_everything

warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

SEED = 42
seed_everything(SEED)

FORCE_RETRAIN = True

BACKBONE_DIM = 64                                # global_add_pool(64) + global_mean_pool(64) — summed, not concatenated
DESC_DIM     = 5                                 # RDKit descriptors per drug
NODE_DIM     = BACKBONE_DIM + DESC_DIM           # 69
EDGE_DIM     = BACKBONE_DIM * 2 + DESC_DIM * 2  # 138
MODEL_PATH   = 'nside_ddi_model.pth'

# --- 1. Molecular Feature Engine (mirrors 019) ---

def get_descriptors(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == '':
        return [0.0] * DESC_DIM
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0] * DESC_DIM
    return [
        Descriptors.MolWt(mol)                / 1000.0,
        Descriptors.MolLogP(mol)              /   10.0,
        Descriptors.TPSA(mol)                 /  200.0,
        float(Descriptors.NumHDonors(mol))    /   10.0,
        float(Descriptors.NumHAcceptors(mol)) /   15.0,
    ]

def smiles_to_graph(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == '':
        return Data(x=torch.zeros((1, 6)),
                    edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return Data(x=torch.zeros((1, 6)),
                    edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
    xs = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
           float(a.GetIsAromatic()), float(a.GetHybridization()),
           a.GetNumRadicalElectrons()] for a in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    return Data(x=x, edge_index=edge_index, num_nodes=x.size(0))

# --- 2. GIN Backbone (identical to 019/020 — loaded frozen) ---

class GINBackbone(torch.nn.Module):
    def __init__(self, node_features=6, hidden_channels=64):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(node_features, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        self.ln1   = LayerNorm(hidden_channels)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.ln2   = LayerNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.ln1(self.conv1(x, edge_index), batch))
        x = F.relu(self.ln2(self.conv2(x, edge_index), batch))
        return global_add_pool(x, batch) + global_mean_pool(x, batch)

def load_frozen_backbone(path='gnn_pretrained_backbone.pth'):
    backbone = GINBackbone()
    if os.path.exists(path):
        backbone.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"   Loaded backbone weights from {path}")
    else:
        print(f"   Warning: {path} not found — backbone uses random weights")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone

@torch.no_grad()
def encode_smiles_list(smiles_list, backbone):
    """Encode a list of SMILES → (n, BACKBONE_DIM) tensor."""
    embeddings = []
    for smi in smiles_list:
        g     = smiles_to_graph(smi)
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        embeddings.append(backbone(g.x, g.edge_index, batch).squeeze(0))
    return torch.stack(embeddings)

# --- 3. Interaction Graph Builder ---

def build_interaction_graph(embeddings, descriptors):
    """
    Build a complete drug interaction graph from precomputed embeddings.

    embeddings:  (n, 128) — GINBackbone output per drug
    descriptors: (n,   5) — RDKit descriptor vector per drug

    Nodes: (n, 133) = backbone embedding + descriptors
    Edges: bidirectional complete graph + self-loops (one per node)
      - Cross-drug edge_attr (266) = [emb_add, emb_diff, d_add, d_diff]
      - Self-loop  edge_attr (266) = zeros
        (self-loops keep GAT neighbourhoods non-empty for monotherapy n=1,
         and prevent magnitude collapse in the first attention round)
    """
    n = embeddings.size(0)
    x = torch.cat([embeddings, descriptors], dim=1)  # (n, 133)

    edge_list = []
    attr_list = []

    for i in range(n):                          # self-loops with zero attributes
        edge_list.append([i, i])
        attr_list.append(torch.zeros(EDGE_DIM))

    for i, j in combinations(range(n), 2):     # bidirectional cross-drug edges
        e_add  = embeddings[i] + embeddings[j]
        e_diff = (embeddings[i] - embeddings[j]).abs()
        d_add  = descriptors[i] + descriptors[j]
        d_diff = (descriptors[i] - descriptors[j]).abs()
        feat   = torch.cat([e_add, e_diff, d_add, d_diff])  # (266,)
        edge_list += [[i, j], [j, i]]
        attr_list += [feat, feat]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.stack(attr_list)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)

# --- 4. Dataset ---

def combo_key(drugs):
    """Order-invariant canonical key for an n-drug combination (anti-leakage)."""
    parts = sorted(str(d).strip().lower()
                   for d in drugs if d and str(d).strip() not in ('', 'nan'))
    return '||'.join(parts) if parts else '__empty__'

class DDINSideDataset(torch.utils.data.Dataset):
    """
    Handles variable-size drug combinations (n >= 1).
    Backbone encoding is precomputed once at construction (frozen backbone).
    Fully backward-compatible with the existing 2-column Drug_N / SMILES_N schema.
    Automatically picks up Drug_3, Drug_4, ... if present in the CSV.
    """
    def __init__(self, df, backbone):
        self.smiles_cols = sorted(c for c in df.columns if c.startswith('SMILES_'))
        self.drug_cols   = sorted(c for c in df.columns if c.startswith('Drug_'))
        target_cols      = [c for c in df.columns if c.startswith('Target_')]

        if not target_cols:
            raise ValueError("No Target_ columns found in dataset.")

        # Build SMILES → embedding cache (backbone runs once per unique structure)
        all_smiles = []
        for col in self.smiles_cols:
            all_smiles.extend(str(s) for s in df[col].dropna() if str(s).strip())
        unique_smiles = list(set(all_smiles))

        print(f"   Pre-computing embeddings for {len(unique_smiles)} unique structures...")
        emb_cache = {}
        for smi in unique_smiles:
            g     = smiles_to_graph(smi)
            batch = torch.zeros(g.num_nodes, dtype=torch.long)
            with torch.no_grad():
                emb_cache[smi] = backbone(g.x, g.edge_index, batch).squeeze(0).detach()

        dummy_emb = torch.zeros(BACKBONE_DIM)

        self.graphs  = []
        self.targets = []
        self.combos  = []  # drug name lists (used externally for combo_key splitting)

        for _, row in df.iterrows():
            smiles_list = []
            drugs_list  = []
            for sc, dc in zip(self.smiles_cols, self.drug_cols):
                smi  = row.get(sc, '')
                drug = row.get(dc, '')
                if isinstance(smi, str) and smi.strip():
                    smiles_list.append(smi.strip())
                    drugs_list.append(str(drug).strip() if pd.notna(drug) else '')

            if not smiles_list:
                continue

            embeddings  = torch.stack([emb_cache.get(s, dummy_emb) for s in smiles_list])
            descriptors = torch.stack([
                torch.tensor(get_descriptors(s), dtype=torch.float)
                for s in smiles_list])

            self.graphs.append(build_interaction_graph(embeddings, descriptors))
            self.targets.append(
                float(max(row[c] for c in target_cols if pd.notna(row[c]))) / 100.0)
            self.combos.append(drugs_list)

        print(f"   Dataset ready: {len(self.graphs)} samples")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.targets[idx], dtype=torch.float32)

# --- 5. N-Side GAT Model ---

class NSideDDIModel(torch.nn.Module):
    """
    Two-round Graph Attention Network over the drug interaction graph.

    add_self_loops=False in GATConv because self-loops are added explicitly
    in build_interaction_graph with their own zero edge attributes, keeping
    edge_attr shapes consistent across all edges in the batch.
    """
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=64,
                 heads1=4, heads2=2):
        super().__init__()
        self.node_proj = torch.nn.Linear(node_dim, 128)

        # Round 1: 128 → hidden*heads1 = 256
        self.gat1  = GATConv(128, hidden, heads=heads1, edge_dim=edge_dim,
                             concat=True, add_self_loops=False)
        self.norm1 = torch.nn.LayerNorm(hidden * heads1)

        # Round 2: 256 → hidden*heads2 = 128
        self.gat2  = GATConv(hidden * heads1, hidden, heads=heads2, edge_dim=edge_dim,
                             concat=True, add_self_loops=False)
        self.norm2 = torch.nn.LayerNorm(hidden * heads2)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * heads2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_proj(x))
        x = F.relu(self.norm1(self.gat1(x, ei, edge_attr=ea)))
        x = F.relu(self.norm2(self.gat2(x, ei, edge_attr=ea)))

        graph_emb = global_add_pool(x, data.batch) + global_mean_pool(x, data.batch)
        return self.head(graph_emb).squeeze(-1)

# --- 6. Training Pipeline ---

def nside_collate(batch):
    graphs, targets = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(targets)

def train_pipeline():
    print("\n--- Phase 6: N-Side DDI Model Training ---")

    input_file = 'training_matrix_nside.csv'
    if not os.path.exists(input_file):
        input_file = 'training_matrix_refined_for_gnn.csv'
    if not os.path.exists(input_file):
        print(f"Error: dataset not found. Run script 023 first.")
        return

    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} samples from {input_file}")

    backbone  = load_frozen_backbone()
    drug_cols = sorted(c for c in df.columns if c.startswith('Drug_'))
    groups    = [combo_key(row[drug_cols].tolist()) for _, row in df.iterrows()]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    overlap = {groups[i] for i in train_idx} & {groups[i] for i in val_idx}
    if overlap:
        raise RuntimeError(f"Combo leakage detected: {len(overlap)} overlapping keys.")
    print(f"   Split: {len(train_df)} train / {len(val_df)} val — no overlap confirmed.")

    print("\nBuilding train dataset...")
    train_ds = DDINSideDataset(train_df, backbone)
    print("Building val dataset...")
    val_ds   = DDINSideDataset(val_df, backbone)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=nside_collate, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=8, collate_fn=nside_collate)

    model     = NSideDDIModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    best_mae, best_state = float('inf'), None

    for epoch in range(1, 201):
        model.train()
        train_loss = 0.0
        for batch_data, targets in train_loader:
            optimizer.zero_grad()
            out  = model(batch_data)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        preds_all, acts_all = [], []
        with torch.no_grad():
            for batch_data, targets in val_loader:
                out = model(batch_data)
                preds_all.extend(np.maximum(out.numpy().flatten(), 0.0) * 100.0)
                acts_all.extend(targets.numpy().flatten() * 100.0)

        avg_mae = mean_absolute_error(acts_all, preds_all)

        if avg_mae < best_mae:
            best_mae   = avg_mae
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == 1:
            avg_loss = train_loss / max(len(train_loader), 1)
            print(f"   Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Val MAE: {avg_mae:.4f}% (Best: {best_mae:.4f}%)")

    torch.save(best_state, MODEL_PATH)
    print(f"\n   Saved best model → {MODEL_PATH}  (Val MAE: {best_mae:.4f}%)")

# --- 7. Inference Tool ---

class NSideInferenceTool:
    def __init__(self, backbone_path='gnn_pretrained_backbone.pth',
                 model_path=MODEL_PATH):
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = load_frozen_backbone(backbone_path).to(self.device)
        self.model    = NSideDDIModel().to(self.device)
        self.is_ready = False

        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.is_ready = True
            print(f"   Model loaded from {model_path}")
        else:
            print(f"   Model not found at {model_path}. Run training first.")

    def fetch_smiles(self, drug_name):
        name = drug_name.strip()
        if not name:
            return ''
        try:
            url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                   f"{urllib.parse.quote(name)}/property/CanonicalSMILES/TXT")
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return res.text.strip()
        except Exception:
            pass
        return None

    def predict(self, drug_names):
        """Predict combined interaction risk for n drugs (n >= 1)."""
        if not self.is_ready:
            return {'error': 'Model not loaded. Run training first.'}
        if not drug_names:
            return {'error': 'No drugs provided.'}

        print(f"   Fetching SMILES for {len(drug_names)} drug(s)...")
        smiles_map = {}
        for name in drug_names:
            smi = self.fetch_smiles(name)
            if smi is None:
                return {'error': f"Could not find SMILES for '{name}' on PubChem."}
            smiles_map[name] = smi
            time.sleep(0.2)  # PubChem rate limit

        valid_pairs = [(n, s) for n, s in smiles_map.items() if s.strip()]
        if not valid_pairs:
            return {'error': 'No valid SMILES retrieved.'}

        valid_names  = [p[0] for p in valid_pairs]
        valid_smiles = [p[1] for p in valid_pairs]

        embeddings  = encode_smiles_list(valid_smiles, self.backbone.cpu())
        descriptors = torch.stack([
            torch.tensor(get_descriptors(s), dtype=torch.float)
            for s in valid_smiles])

        graph = build_interaction_graph(embeddings, descriptors)
        batch = Batch.from_data_list([graph]).to(self.device)

        with torch.no_grad():
            out = self.model(batch)
            inc = float(max(0.0, min(out.item() * 100.0, 100.0)))

        tier  = 'Low' if inc < 5 else 'Moderate' if inc < 20 else 'High'
        emoji = {'Low': '🟢', 'Moderate': '🟡', 'High': '🔴'}[tier]

        return {
            'incidence': f'{inc:.2f}%',
            'tier':      f'{emoji} {tier}',
            'n':         len(valid_names),
            'drugs':     valid_names,
            'smiles':    smiles_map,
        }

# --- 8. Entry Point ---

if __name__ == '__main__':
    needs_training = FORCE_RETRAIN or not os.path.exists(MODEL_PATH)

    if needs_training:
        if FORCE_RETRAIN and os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print("Removed existing model for retraining.")
        train_pipeline()

    print('\n' + '=' * 55)
    print('DROPHET — N-SIDE DDI PREDICTOR')
    print('=' * 55)
    print('Enter drugs one per line. Blank line to analyze.')
    print("Commands: 'retrain', 'exit'\n")

    tool = NSideInferenceTool()
    if not tool.is_ready:
        print('Training required. Set FORCE_RETRAIN = True and re-run.')
        exit()

    while True:
        print('Enter drugs (one per line, blank line when done):')
        drug_names = []
        try:
            while True:
                line = input(f'  Drug {len(drug_names) + 1}: ').strip()
                if line.lower() == 'exit':
                    print('Exiting.')
                    exit()
                if line.lower() == 'retrain':
                    train_pipeline()
                    tool = NSideInferenceTool()
                    break
                if not line:
                    break
                drug_names.append(line)
        except KeyboardInterrupt:
            print('\nInterrupted.')
            exit()

        if not drug_names:
            continue

        print(f'\nAnalyzing: {" + ".join(drug_names)}')
        res = tool.predict(drug_names)

        if 'error' in res:
            print(f'Error: {res["error"]}')
        else:
            print(f'\n  Drugs ({res["n"]}): {", ".join(res["drugs"])}')
            for name, smi in res['smiles'].items():
                print(f'  {name}: {smi or "(no interaction partner)"}')
            print(f'\n  Combined Risk: {res["incidence"]} | {res["tier"]}\n')
