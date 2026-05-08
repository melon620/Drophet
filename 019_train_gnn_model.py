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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from drophet_utils import seed_everything, pair_keys

warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

SEED = 42
seed_everything(SEED)

# --- 0. Control Flags ---
FORCE_RETRAIN = True
LIVE_PLOT = True  # real-time training dashboard

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
    def __init__(self, df):
        self.smiles1 = df['SMILES_1'].values
        self.smiles2 = df['SMILES_2'].values
        self.desc1 = np.array([get_descriptors(s) for s in self.smiles1])
        self.desc2 = np.array([get_descriptors(s) for s in self.smiles2])

        target_cols = [c for c in df.columns if c.startswith('Target_')]
        if not target_cols: raise ValueError("No Target_ columns found in dataset.")

        max_risk_values = df[target_cols].max(axis=1).values
        self.targets = (max_risk_values.reshape(-1, 1).astype(np.float32)) / 100.0

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
    def __init__(self, node_features=6, desc_features=5, hidden_channels=64):
        super(GNNModel, self).__init__()
        self.backbone = GINBackbone(node_features, hidden_channels)
        input_dim = (hidden_channels * 2) + (desc_features * 2)

        # [CRITICAL FIX] LayerNorm neutralizes the magnitude explosion from global_add_pool for large molecules
        self.norm = torch.nn.LayerNorm(input_dim)

        # Restored capacity to learn complex synergies
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)

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
        self.model = GNNModel().to(self.device)
        self.is_ready = False

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
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
        else:
            print("⚠️ Inference Tool initialization failed: Missing model artifacts.")

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
            inc = max(0.0, min(output.item() * 100.0, 100.0))

        tier = "🟢 Low Risk" if inc < 5 else "🟡 Moderate Risk" if inc < 20 else "🔴 High Risk"
        return {"incidence": f"{inc:.2f}%", "tier": tier, "s1": s1, "s2": s2}

# --- 4. Live Visualisation Helpers ---

_DARK, _PANEL, _TEXT, _MUTED = '#0d1117', '#161b22', '#c9d1d9', '#8b949e'
_BLUE, _RED, _GREEN, _BORDER  = '#58a6ff', '#ff7b72', '#3fb950', '#30363d'

# Known pharmacological anchors tracked live during training
_REF_PAIRS = [
    ("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
     "CC(=O)OC1=CC=CC=C1C(=O)O",
     "Warfarin", "Aspirin", 38.45),
    ("CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
     "CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OCC3COC(O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl",
     "Simvastatin", "Ketoconazole", 32.40),
]

def _infer_ref(model):
    model.eval()
    out = []
    with torch.no_grad():
        for s1, s2, n1, n2, actual in _REF_PAIRS:
            g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
            d1 = torch.tensor([get_descriptors(s1)], dtype=torch.float)
            d2 = torch.tensor([get_descriptors(s2)], dtype=torch.float)
            raw = model(Batch.from_data_list([g1]), Batch.from_data_list([g2]), d1, d2)
            out.append((n1, n2, actual, max(0.0, raw.item() * 100.0)))
    return out

def _setup_dashboard():
    plt.ion()
    fig = plt.figure(figsize=(16, 9), facecolor=_DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    axs = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in {
        'loss': (0,0), 'mae': (0,1), 'scat': (0,2),
        'm1':   (1,0), 'm2': (1,1), 'info': (1,2)}.items()}

    for ax in axs.values():
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_MUTED, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(_BORDER)

    # Loss curve
    axs['loss'].set_title('Train MSE Loss', color=_TEXT, fontsize=9, fontweight='bold')
    axs['loss'].set_xlabel('Epoch', color=_MUTED, fontsize=8)
    line_loss, = axs['loss'].plot([], [], color=_BLUE, lw=1.5)

    # MAE curve
    axs['mae'].set_title('Validation MAE (%)', color=_TEXT, fontsize=9, fontweight='bold')
    axs['mae'].set_xlabel('Epoch', color=_MUTED, fontsize=8)
    line_mae,  = axs['mae'].plot([], [], color=_RED,   lw=1.5, label='Val MAE')
    line_best, = axs['mae'].plot([], [], color=_GREEN, lw=1.2, ls='--', label='Best')
    axs['mae'].legend(fontsize=7, labelcolor=_TEXT, facecolor=_PANEL, edgecolor=_BORDER)

    # Scatter plot
    axs['scat'].set_title('Test: Actual vs Predicted', color=_TEXT, fontsize=9, fontweight='bold')
    axs['scat'].set_xlabel('Actual Risk (%)', color=_MUTED, fontsize=8)
    axs['scat'].set_ylabel('Predicted Risk (%)', color=_MUTED, fontsize=8)
    scat_pts = axs['scat'].scatter([], [], c=_BLUE, alpha=0.55, s=22, zorder=3)
    diag,    = axs['scat'].plot([], [], '--', color=_RED, alpha=0.35, lw=1)

    # Molecular structure panels (drawn once, static)
    try:
        from rdkit.Chem import Draw
        for ax_key, (s1, s2, n1, n2, actual) in zip(('m1', 'm2'), _REF_PAIRS):
            mol1, mol2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
            if mol1 and mol2:
                img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2,
                                           subImgSize=(220, 140), returnPNG=False)
                axs[ax_key].imshow(img)
            axs[ax_key].axis('off')
            axs[ax_key].set_title(f'{n1}  +  {n2}\nActual: {actual:.1f}%  |  Pred: —',
                                  color=_TEXT, fontsize=8, fontweight='bold')
    except Exception:
        for k in ('m1', 'm2'):
            axs[k].axis('off')
            axs[k].text(0.5, 0.5, 'RDKit draw unavailable',
                        transform=axs[k].transAxes, color=_MUTED,
                        ha='center', va='center', fontsize=8)

    # Info / metrics panel
    axs['info'].axis('off')
    info_txt = axs['info'].text(0.06, 0.96, 'Initialising…',
                                transform=axs['info'].transAxes,
                                color=_TEXT, fontsize=9, va='top',
                                fontfamily='monospace')

    fig.suptitle('Project Drophet — Live GNN Training', color=_TEXT,
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.pause(0.05)

    artists = dict(line_loss=line_loss, line_mae=line_mae, line_best=line_best,
                   scat_pts=scat_pts, diag=diag, info_txt=info_txt)
    return fig, axs, artists

def _update_dashboard(fig, axs, artists, epoch,
                      epochs_h, loss_h, mae_h, best_h,
                      all_a, all_p, ref_preds):
    ll, lm, lb  = artists['line_loss'], artists['line_mae'], artists['line_best']
    sp, dg, itx = artists['scat_pts'],  artists['diag'],     artists['info_txt']

    ll.set_data(epochs_h, loss_h)
    axs['loss'].relim(); axs['loss'].autoscale_view()

    lm.set_data(epochs_h, mae_h)
    lb.set_data(epochs_h, best_h)
    axs['mae'].relim(); axs['mae'].autoscale_view()

    if all_a:
        sp.set_offsets(np.column_stack([all_a, all_p]))
        lim = max(max(all_a), max(all_p), 1.0) * 1.08
        dg.set_data([0, lim], [0, lim])
        axs['scat'].set_xlim(0, lim); axs['scat'].set_ylim(0, lim)

    for ax_key, (n1, n2, actual, pred) in zip(('m1', 'm2'), ref_preds):
        col = _RED if pred > 20 else (_BLUE if pred > 5 else _GREEN)
        axs[ax_key].set_title(f'{n1}  +  {n2}\nActual: {actual:.1f}%  |  Pred: {pred:.1f}%',
                              color=col, fontsize=8, fontweight='bold')

    info_lines = [
        f"  Epoch    {epoch:>3d} / 200",
        f"  Val MAE  {mae_h[-1]:.3f}%",
        f"  Best MAE {best_h[-1]:.3f}%",
        f"  MSE Loss {loss_h[-1]:.5f}",
        "",
        "  Reference predictions:",
    ]
    for n1, n2, actual, pred in ref_preds:
        tier = "HIGH" if pred > 20 else ("MOD" if pred > 5 else "low")
        info_lines.append(f"  {n1[:9]}+{n2[:9]}: {pred:5.1f}% [{tier}]")
    itx.set_text('\n'.join(info_lines))

    fig.canvas.draw_idle()
    plt.pause(0.001)


# --- 5. Training Pipeline ---

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

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(DDIPairDataset(train_df), batch_size=8, shuffle=True,
                              collate_fn=pair_collate, generator=g)
    test_loader = DataLoader(DDIPairDataset(test_df), batch_size=8, collate_fn=pair_collate)

    # --- Live dashboard setup ---
    epochs_h, loss_h, mae_h, best_h = [], [], [], []
    _fig, _axs, _artists = (None, None, None)
    if LIVE_PLOT:
        try:
            _fig, _axs, _artists = _setup_dashboard()
        except Exception as e:
            print(f"⚠️  Live plot unavailable ({e}); continuing without visualisation.")

    model = GNNModel().to('cpu')
    pretrain_file = 'gnn_pretrained_backbone.pth'
    if os.path.exists(pretrain_file):
        print(f"💎 Loading pretrained chemical backbone...")
        pre_state = torch.load(pretrain_file)
        model.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in pre_state.items()}, strict=False)

    # Adding a tiny weight_decay (1e-4) to prevent overfitting on the small dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    best_mae, best_state = float('inf'), None

    for epoch in range(1, 201):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for g1, g2, d1, d2, target in train_loader:
            optimizer.zero_grad()
            out = model(g1, g2, d1, d2)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches  += 1
        train_loss_avg = train_loss_sum / max(train_batches, 1)

        model.eval()
        total_mae = 0
        all_a, all_p = [], []
        with torch.no_grad():
            for g1, g2, d1, d2, target in test_loader:
                out = model(g1, g2, d1, d2)
                p = np.maximum(out.numpy().flatten(), 0.0) * 100.0
                a = target.numpy().flatten() * 100.0
                total_mae += mean_absolute_error(a, p)
                if LIVE_PLOT and _fig is not None:
                    all_a.extend(a.tolist())
                    all_p.extend(p.tolist())

        avg_mae = total_mae / len(test_loader)

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == 1:
            print(f"   Epoch {epoch:03d} | Val MAE: {avg_mae:.4f}% (Best: {best_mae:.4f}%)")

        if LIVE_PLOT and _fig is not None:
            epochs_h.append(epoch)
            loss_h.append(train_loss_avg)
            mae_h.append(avg_mae)
            best_h.append(best_mae)
            try:
                _update_dashboard(_fig, _axs, _artists, epoch,
                                  epochs_h, loss_h, mae_h, best_h,
                                  all_a, all_p, _infer_ref(model))
            except Exception:
                pass

    torch.save(best_state, 'ddi_gnn_best_model.pth')
    print("✅ Training complete. Artifacts saved successfully.")

    if LIVE_PLOT and _fig is not None:
        plt.ioff()
        _fig.suptitle('Project Drophet — Training Complete', color=_TEXT,
                      fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        print("📊 Dashboard open — close the window or press Ctrl+C to continue.")

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

    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user.")
    except Exception as e:
        print(f"\n🚨 Unexpected Execution Error: {e}")
    print("\n👋 Ready for deployment! Goodbye.")
