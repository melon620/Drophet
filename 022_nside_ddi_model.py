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
import math
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
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GATConv, GINConv, LayerNorm, global_add_pool,
                                 global_mean_pool)

from drophet_utils import seed_everything

warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

SEED = 42
seed_everything(SEED)

FORCE_RETRAIN = False

BACKBONE_DIM = 64                                # global_add_pool(64) + global_mean_pool(64) — summed, not concatenated
DESC_DIM     = 5                                 # RDKit descriptors per drug
NODE_DIM     = BACKBONE_DIM + DESC_DIM           # 69
EDGE_DIM     = BACKBONE_DIM * 2 + DESC_DIM * 2  # 138
MODEL_PATH   = 'nside_ddi_model.pth'

TARGET_COLS = [
    'Target_Hematologic', 'Target_Cardiovascular', 'Target_Hepatobiliary',
    'Target_Nervous_System', 'Target_Respiratory', 'Target_Musculoskeletal',
    'Target_Renal', 'Target_Gastrointestinal', 'Target_Dermatologic',
]

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
            self.targets.append(torch.tensor(
                [float(row[c]) / 100.0 if pd.notna(row[c]) else 0.0
                 for c in target_cols], dtype=torch.float32))
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
                 heads1=4, heads2=2, n_targets=9):
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
            torch.nn.Linear(hidden, n_targets),
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_proj(x))
        x = F.relu(self.norm1(self.gat1(x, ei, edge_attr=ea)))
        x = F.relu(self.norm2(self.gat2(x, ei, edge_attr=ea)))

        graph_emb = global_add_pool(x, data.batch) + global_mean_pool(x, data.batch)
        return self.head(graph_emb)  # (batch, n_targets)

    def forward_with_attention(self, data):
        """Same as forward() but also returns per-edge attention averaged across heads and layers."""
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_proj(x))

        x1, (_, alpha1) = self.gat1(x,  ei, edge_attr=ea, return_attention_weights=True)
        x1 = F.relu(self.norm1(x1))

        x2, (_, alpha2) = self.gat2(x1, ei, edge_attr=ea, return_attention_weights=True)
        x2 = F.relu(self.norm2(x2))

        graph_emb = global_add_pool(x2, data.batch) + global_mean_pool(x2, data.batch)
        output    = self.head(graph_emb)  # (batch, n_targets)

        # Average across heads, then average the two layers → (num_edges,)
        combined_alpha = (alpha1.mean(dim=1) + alpha2.mean(dim=1)) / 2.0

        return output, ei, combined_alpha

# --- 6. Training Pipeline ---

def nside_collate(batch):
    graphs, targets = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(targets)

def train_pipeline():
    print("\n--- Phase 6: N-Side DDI Model Training ---")

    for candidate in ('training_matrix_real.csv',
                      'training_matrix_nside.csv',
                      'training_matrix_refined_for_gnn.csv'):
        if os.path.exists(candidate):
            input_file = candidate
            break
    else:
        print("Error: no training dataset found. Run script 024 first.")
        return

    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} samples from {input_file}")

    backbone    = load_frozen_backbone()
    drug_cols   = sorted(c for c in df.columns if c.startswith('Drug_'))
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    n_targets   = len(target_cols)
    print(f"   Target columns ({n_targets}): {', '.join(target_cols)}")
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

    model     = NSideDDIModel(n_targets=n_targets)
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
                preds_all.append(out.numpy() * 100.0)    # (batch, n_targets)
                acts_all.append(targets.numpy() * 100.0) # (batch, n_targets)

        preds_np = np.concatenate(preds_all, axis=0)
        acts_np  = np.concatenate(acts_all,  axis=0)
        avg_mae  = float(np.abs(preds_np - acts_np).mean())

        if avg_mae < best_mae:
            best_mae   = avg_mae
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == 1:
            avg_loss = train_loss / max(len(train_loader), 1)
            print(f"   Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Val MAE: {avg_mae:.4f}% (Best: {best_mae:.4f}%)")
        if epoch % 40 == 0:
            per_cat = np.abs(preds_np - acts_np).mean(axis=0)
            short = [c.replace('Target_', '') for c in target_cols]
            cat_str = '  '.join(f"{n}:{v:.1f}" for n, v in zip(short, per_cat))
            print(f"            Per-cat MAE: {cat_str}")

    torch.save({'state_dict': best_state, 'target_cols': target_cols,
                'n_targets': n_targets}, MODEL_PATH)
    print(f"\n   Saved best model → {MODEL_PATH}  (Val MAE: {best_mae:.4f}%)")
    print(f"   Categories: {', '.join(c.replace('Target_', '') for c in target_cols)}")

# --- 7. Inference Tool ---

class NSideInferenceTool:
    def __init__(self, backbone_path='gnn_pretrained_backbone.pth',
                 model_path=MODEL_PATH):
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = load_frozen_backbone(backbone_path).to(self.device)
        self.model    = NSideDDIModel().to(self.device)
        self.is_ready = False

        self.target_cols = TARGET_COLS  # fallback

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                n_targets = ckpt.get('n_targets', 9)
                self.target_cols = ckpt.get('target_cols', TARGET_COLS)
                self.model = NSideDDIModel(n_targets=n_targets).to(self.device)
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()
            self.is_ready = True
            print(f"   Model loaded from {model_path} "
                  f"({len(self.target_cols)} targets)")
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
            out  = self.model(batch).squeeze(0)  # (n_targets,)
            vals = [float(max(0.0, min(v * 100.0, 100.0))) for v in out.tolist()]

        per_cat = {col.replace('Target_', ''): v
                   for col, v in zip(self.target_cols, vals)}
        inc = max(vals)

        tier  = 'Low' if inc < 5 else 'Moderate' if inc < 20 else 'High'
        emoji = {'Low': '🟢', 'Moderate': '🟡', 'High': '🔴'}[tier]

        return {
            'incidence':    f'{inc:.2f}%',
            'tier':         f'{emoji} {tier}',
            'per_category': {k: f'{v:.2f}%' for k, v in per_cat.items()},
            'n':            len(valid_names),
            'drugs':        valid_names,
            'smiles':       smiles_map,
        }

    def get_attention(self, drug_names, smiles_map):
        """
        Extract GAT attention weights and map them back to named drug pairs.
        Returns list of (drug_i, drug_j, share) sorted by descending attention,
        where share is in [0,1] and all cross-drug shares sum to 1.
        Returns [] for monotherapy (n=1, no cross-drug edges).
        """
        if not self.is_ready:
            return []

        valid_pairs  = [(n, s) for n, s in smiles_map.items() if s and s.strip()]
        valid_names  = [p[0] for p in valid_pairs]
        valid_smiles = [p[1] for p in valid_pairs]
        n = len(valid_names)

        if n < 2:
            return []

        embeddings  = encode_smiles_list(valid_smiles, self.backbone.cpu())
        descriptors = torch.stack([
            torch.tensor(get_descriptors(s), dtype=torch.float)
            for s in valid_smiles])

        graph = build_interaction_graph(embeddings, descriptors)
        batch = Batch.from_data_list([graph]).to(self.device)

        with torch.no_grad():
            _, ei, alpha = self.model.forward_with_attention(batch)

        # Accumulate attention for each undirected cross-drug pair
        pair_alphas = {}
        for e in range(ei.shape[1]):
            src, dst = ei[0, e].item(), ei[1, e].item()
            if src == dst:
                continue  # skip self-loops
            key = tuple(sorted((src, dst)))
            pair_alphas.setdefault(key, []).append(alpha[e].item())

        # Average bidirectional attention per pair, then normalise to sum=1
        pair_scores = {k: float(np.mean(v)) for k, v in pair_alphas.items()}
        total = sum(pair_scores.values()) or 1.0
        pair_scores = {k: v / total for k, v in pair_scores.items()}

        return sorted(
            [(valid_names[i], valid_names[j], score)
             for (i, j), score in pair_scores.items()],
            key=lambda x: -x[2],
        )

# --- 8. Prediction Visualiser ---

def visualize_prediction(result, tool, attention=None):
    """
    Render a 3-panel dark-theme figure for a completed prediction:
      Top    — risk gauge (gradient bar, combined % marked)
      Bottom-left  — pairwise vs combined bar chart
      Bottom-right — drug interaction network (nodes = drugs, edges = pair risk)
    Pairwise risks are computed on-the-fly by running the n-side model for each
    2-drug sub-combination, so the same model is used throughout.
    """
    try:
        import matplotlib
        matplotlib.use('MacOSX')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as e:
        print(f"  (Visualisation unavailable: {e})")
        return

    drugs        = result['drugs']
    smiles_dict  = result['smiles']
    combined_pct = float(result['incidence'].replace('%', ''))
    n            = result['n']
    per_cat      = {k: float(v.replace('%', ''))
                    for k, v in result.get('per_category', {}).items()}

    def risk_color(r):
        if r < 5:  return '#27ae60'
        if r < 20: return '#f39c12'
        return '#e74c3c'

    # Pairwise risks still shown in the network panel (lightweight model calls)
    pair_data = []
    for i, j in combinations(range(n), 2):
        si, sj = smiles_dict.get(drugs[i], ''), smiles_dict.get(drugs[j], '')
        pair_risk = 0.0
        if si and sj:
            embs  = encode_smiles_list([si, sj], tool.backbone.cpu())
            descs = torch.stack([
                torch.tensor(get_descriptors(si), dtype=torch.float),
                torch.tensor(get_descriptors(sj), dtype=torch.float),
            ])
            g = build_interaction_graph(embs, descs)
            with torch.no_grad():
                out = tool.model(Batch.from_data_list([g]).to(tool.device))
                pair_risk = float(max(0.0, min(out.squeeze(0).max().item() * 100.0, 100.0)))
        pair_data.append((drugs[i], drugs[j], i, j, pair_risk))

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')

    if n > 1:
        ax_gauge = fig.add_axes([0.05, 0.58, 0.90, 0.32])
        ax_pairs = fig.add_axes([0.05, 0.07, 0.44, 0.42])
        ax_net   = fig.add_axes([0.55, 0.07, 0.40, 0.42])
    else:
        ax_gauge = fig.add_axes([0.05, 0.40, 0.90, 0.45])
        ax_pairs = None
        ax_net   = fig.add_axes([0.20, 0.05, 0.60, 0.30])

    # ── Panel 1: Risk gauge ────────────────────────────────────────────────────
    cmap_risk = LinearSegmentedColormap.from_list('risk', ['#27ae60', '#f39c12', '#e74c3c'])
    gradient  = np.linspace(0, 1, 300).reshape(1, -1)
    ax_gauge.imshow(gradient, aspect='auto', cmap=cmap_risk,
                    extent=[0, 100, 0, 1], alpha=0.30)
    ax_gauge.axvspan(0,   5,  alpha=0.08, color='#27ae60')
    ax_gauge.axvspan(5,  20,  alpha=0.08, color='#f39c12')
    ax_gauge.axvspan(20, 100, alpha=0.08, color='#e74c3c')

    # Needle + value label
    col = risk_color(combined_pct)
    ax_gauge.axvline(combined_pct, color=col, lw=3, alpha=0.95, zorder=4)
    ax_gauge.scatter([combined_pct], [0.50], color=col, s=220, zorder=5)
    ax_gauge.text(combined_pct, 0.82, f"{combined_pct:.1f}%",
                  ha='center', va='center', color='white',
                  fontsize=26, fontweight='bold', zorder=6)
    tier_label = result['tier'].split()[-1].upper()   # strip emoji, e.g. "HIGH"
    ax_gauge.text(combined_pct, 0.22, tier_label,
                  ha='center', va='center', color=col,
                  fontsize=12, fontweight='bold', zorder=6)

    # Zone labels
    for x, lbl in [(2.5, 'LOW'), (12.5, 'MODERATE'), (60, 'HIGH')]:
        ax_gauge.text(x, 0.06, lbl, ha='center', va='bottom',
                      color='#555577', fontsize=7.5, style='italic')

    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.set_xticks([0, 5, 20, 40, 60, 80, 100])
    ax_gauge.set_xticklabels(['0%','5%','20%','40%','60%','80%','100%'],
                              color='#888899', fontsize=9)
    ax_gauge.set_yticks([])
    ax_gauge.set_facecolor('#0d0d1a')
    ax_gauge.set_title(
        'Combined Risk — ' + ' + '.join(drugs),
        color='white', fontsize=12, fontweight='bold', pad=10)
    for sp in ax_gauge.spines.values():
        sp.set_edgecolor('#333355')

    # ── Panel 2: Per-category risk bar chart ──────────────────────────────────
    if ax_pairs is not None and per_cat:
        cat_names = list(per_cat.keys())
        cat_risks = list(per_cat.values())
        bar_colors = [risk_color(r) for r in cat_risks]
        y_pos = np.arange(len(cat_names))
        ax_pairs.barh(y_pos, cat_risks, color=bar_colors, alpha=0.85,
                      height=0.6, zorder=2)
        for idx, r in enumerate(cat_risks):
            ax_pairs.text(r + 0.4, idx, f"{r:.1f}%",
                          va='center', color='white', fontsize=9)
        ax_pairs.set_yticks(y_pos)
        ax_pairs.set_yticklabels(cat_names, color='#cccccc', fontsize=9)
        ax_pairs.set_xlim(0, max(cat_risks + [1]) * 1.35 + 3)
        ax_pairs.set_xlabel('Risk (%)', color='#888899', fontsize=9)
        ax_pairs.set_title('Risk by Organ System', color='white',
                           fontsize=10, fontweight='bold')
        ax_pairs.set_facecolor('#0d0d1a')
        ax_pairs.invert_yaxis()
        ax_pairs.tick_params(colors='#666688')
        for sp in ax_pairs.spines.values():
            sp.set_edgecolor('#333355')
        ax_pairs.axvline(5,  color='#27ae60', lw=0.8, ls=':', alpha=0.5, zorder=1)
        ax_pairs.axvline(20, color='#f39c12', lw=0.8, ls=':', alpha=0.5, zorder=1)

    # ── Panel 3: Drug network graph ────────────────────────────────────────────
    ax_net.set_facecolor('#0d0d1a')
    ax_net.set_aspect('equal')
    ax_net.set_xlim(-1.7, 1.7)
    ax_net.set_ylim(-1.7, 1.7)
    ax_net.axis('off')
    ax_net.set_title('Interaction Network', color='white',
                     fontsize=10, fontweight='bold')

    # Place nodes on a regular polygon (or centred for n=1)
    if n == 1:
        positions = [(0.0, 0.0)]
    else:
        positions = [
            (math.cos(2 * math.pi * i / n - math.pi / 2),
             math.sin(2 * math.pi * i / n - math.pi / 2))
            for i in range(n)
        ]

    # Build attention lookup: {frozenset({di,dj}): share}
    attn_map = {}
    if attention:
        for ai, aj, share in attention:
            attn_map[frozenset({ai, aj})] = share

    # Edges (drawn before nodes so nodes appear on top)
    for di, dj, i, j, pr in pair_data:
        xi, yi = positions[i]
        xj, yj = positions[j]
        ec    = risk_color(pr)
        share = attn_map.get(frozenset({di, dj}), None)
        # Thickness: attention-driven when available, else risk-driven
        lw    = (1.5 + share * 12.0) if share is not None else (1.5 + pr / 20.0)
        alpha_edge = 0.55 + (share * 0.45 if share is not None else 0.25)
        ax_net.plot([xi, xj], [yi, yj], color=ec, lw=lw,
                    alpha=alpha_edge, zorder=1)
        mx, my = (xi + xj) / 2 * 1.18, (yi + yj) / 2 * 1.18
        # Label: show both risk% and attention% if available
        lbl = f"{pr:.0f}%"
        if share is not None:
            lbl += f"\n{share*100:.0f}% attn"
        ax_net.text(mx, my, lbl,
                    ha='center', va='center', color=ec, fontsize=7.5,
                    fontweight='bold', zorder=3,
                    bbox=dict(facecolor='#1a1a2e', edgecolor='none',
                              alpha=0.75, pad=1.5))

    # Nodes — highlight the drug involved in the top-attention pair
    top_pair_drugs = set()
    if attention:
        top_pair_drugs = {attention[0][0], attention[0][1]}

    for i, (x, y) in enumerate(positions):
        is_top = drugs[i] in top_pair_drugs
        ec_node = '#f39c12' if is_top else '#4a90d9'
        lw_node = 2.8      if is_top else 2.0
        circ = mpatches.Circle((x, y), 0.24, color='#16213e',
                                ec=ec_node, lw=lw_node, zorder=4)
        ax_net.add_patch(circ)
        label = drugs[i] if len(drugs[i]) <= 13 else drugs[i][:12] + '…'
        ax_net.text(x, y, label, ha='center', va='center',
                    color='white', fontsize=7.5, fontweight='bold', zorder=5)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, 'DROPHET  ·  N-SIDE DDI ANALYSIS',
             ha='center', va='top', color='#4a90d9',
             fontsize=13, fontweight='bold')

    plt.show(block=False)
    plt.pause(0.001)


# --- 9. Entry Point ---

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
            print(f'\n  Combined Risk: {res["incidence"]} | {res["tier"]}')
            if 'per_category' in res:
                print('\n  Per-organ-system risks:')
                for cat, risk_str in res['per_category'].items():
                    r = float(risk_str.replace('%', ''))
                    bar = '█' * int(r / 3)
                    print(f'    {cat:<22} {risk_str:>8}  {bar}')

            attention = tool.get_attention(res['drugs'], res['smiles'])
            if attention:
                print('\n  Interaction drivers (GAT attention):')
                for di, dj, share in attention:
                    bar = '█' * int(share * 30)
                    print(f'    {di} + {dj:<20} {share*100:5.1f}%  {bar}')
            print()
            visualize_prediction(res, tool, attention=attention)
