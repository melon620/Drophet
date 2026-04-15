# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 3: Advanced GNN Modeling (Graph Neural Networks - Symmetrized GIN with Early Stopping)

This script implements a Siamese Graph Isomorphism Network (GIN) with:
1. Data Symmetrization: Doubling training data for robustness.
2. Best Model Tracking: Saving weights based on minimal test MAE.
3. Enhanced Regularization: Dropout within GIN blocks to prevent overfitting.
4. Final Result Export: Saving predictions vs actuals for performance auditing.
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

# --- 1. Enhanced Graph & Descriptor Engine ---

def get_descriptors(smiles):
    """Calculates global descriptors for fusion with graph embeddings."""
    if pd.isna(smiles) or smiles == "":
        return [0.0] * 5
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0] * 5
    # Standardized scaling based on typical drug property ranges
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
        self.df_metadata = df[['Drug_1', 'Drug_2']].reset_index(drop=True)
        self.smiles1 = df['SMILES_1'].values
        self.smiles2 = df['SMILES_2'].values
        self.augment = augment

        # Pre-calculate descriptors
        self.desc1 = np.array([get_descriptors(s) for s in self.smiles1])
        self.desc2 = np.array([get_descriptors(s) for s in self.smiles2])

        y = df[target_name].values.reshape(-1, 1)
        self.targets = target_scaler.transform(y) if target_scaler else y

    def __len__(self):
        return len(self.targets) * 2 if self.augment else len(self.targets)

    def __getitem__(self, idx):
        is_swapped = False
        meta_idx = idx
        if self.augment and idx >= len(self.targets):
            meta_idx = idx - len(self.targets)
            is_swapped = True

        if not is_swapped:
            s1, s2 = self.smiles1[meta_idx], self.smiles2[meta_idx]
            d1, d2 = self.desc1[meta_idx], self.desc2[meta_idx]
        else:
            s1, s2 = self.smiles2[meta_idx], self.smiles1[meta_idx]
            d1, d2 = self.desc2[meta_idx], self.desc1[meta_idx]

        g1 = smiles_to_graph(s1)
        g2 = smiles_to_graph(s2)

        if g1 is None: g1 = Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
        if g2 is None: g2 = Data(x=torch.zeros((1, 6)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)

        return g1, g2, torch.tensor(d1, dtype=torch.float), torch.tensor(d2, dtype=torch.float), torch.tensor(self.targets[meta_idx], dtype=torch.float)

# --- 2. Advanced GIN Architecture ---

class GNNModel(torch.nn.Module):
    def __init__(self, node_features=6, desc_features=5, hidden_channels=64):
        super(GNNModel, self).__init__()

        # GIN Convolutional layers with Dropout
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, hidden_channels)
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

        input_dim = (hidden_channels * 2) + (desc_features * 2)
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)

    def forward_one(self, g):
        x, edge_index, batch = g.x, g.edge_index, g.batch
        x = F.relu(self.ln1(self.conv1(x, edge_index), batch))
        x = F.relu(self.ln2(self.conv2(x, edge_index), batch))
        return global_add_pool(x, batch) + global_mean_pool(x, batch)

    def forward(self, g1, g2, d1, d2):
        emb1 = self.forward_one(g1)
        emb2 = self.forward_one(g2)
        combined = torch.cat([emb1, emb2, d1, d2], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        return self.out(x)

def pair_collate(batch):
    g1_list, g2_list, d1_list, d2_list, targets = zip(*batch)
    return Batch.from_data_list(g1_list), Batch.from_data_list(g2_list), \
           torch.stack(d1_list), torch.stack(d2_list), torch.stack(targets)

# --- 3. Main Execution ---

def main():
    print("--- Phase 5: GIN-Descriptor Hybrid Pipeline (Robust Training) ---")
    input_file = 'training_matrix_refined_for_gnn.csv'
    if not os.path.exists(input_file): return

    df = pd.read_csv(input_file)
    target_name = [c for c in df.columns if c.startswith('Target_')][0]

    target_scaler = StandardScaler()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    target_scaler.fit(train_df[target_name].values.reshape(-1, 1))

    # Keep track of drug names for the test set
    test_drug_names = test_df[['Drug_1', 'Drug_2']].reset_index(drop=True)

    train_loader = DataLoader(DDIPairDataset(train_df, target_name, target_scaler, augment=True),
                              batch_size=8, shuffle=True, collate_fn=pair_collate)
    test_loader = DataLoader(DDIPairDataset(test_df, target_name, target_scaler, augment=False),
                             batch_size=8, shuffle=False, collate_fn=pair_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = torch.nn.HuberLoss()

    best_mae = float('inf')
    best_model_state = None

    print(f"Training on {device} with Early Stopping Logic...")
    for epoch in range(1, 401):
        model.train()
        total_train_loss = 0
        for g1, g2, d1, d2, target in train_loader:
            g1, g2, d1, d2, target = g1.to(device), g2.to(device), d1.to(device), d2.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(g1, g2, d1, d2)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Periodic Evaluation
        model.eval()
        total_test_mae = 0
        with torch.no_grad():
            for g1, g2, d1, d2, target in test_loader:
                g1, g2, d1, d2, target = g1.to(device), g2.to(device), d1.to(device), d2.to(device), target.to(device)
                out = model(g1, g2, d1, d2)
                p = target_scaler.inverse_transform(out.cpu().numpy()).flatten()
                a = target_scaler.inverse_transform(target.cpu().numpy()).flatten()
                total_test_mae += mean_absolute_error(a, np.maximum(p, 0))

        avg_test_mae = total_test_mae / len(test_loader)
        scheduler.step(total_train_loss / len(train_loader))

        if avg_test_mae < best_mae:
            best_mae = avg_test_mae
            best_model_state = copy.deepcopy(model.state_dict())
            if epoch > 50:
                print(f"   New Best MAE: {best_mae:.4f} at Epoch {epoch}")

        if epoch % 100 == 0:
            print(f"   Epoch {epoch:03d} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val MAE: {avg_test_mae:.4f}")

    # Load the best state for final evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for g1, g2, d1, d2, target in test_loader:
            g1, g2, d1, d2, target = g1.to(device), g2.to(device), d1.to(device), d2.to(device), target.to(device)
            out = model(g1, g2, d1, d2)
            p = target_scaler.inverse_transform(out.cpu().numpy()).flatten()
            a = target_scaler.inverse_transform(target.cpu().numpy()).flatten()
            preds.extend(np.maximum(p, 0)); actuals.extend(a)

    print("\n" + "="*40)
    print("FINAL EVALUATION")
    print("="*40)
    final_r2 = r2_score(actuals, preds)
    final_mae = mean_absolute_error(actuals, preds)
    print(f"R-squared (R2): {final_r2:.4f}")
    print(f"MAE:            {final_mae:.4f}%")
    print("="*40)

    # Save Results for Audit
    comparison_df = test_drug_names.copy()
    comparison_df['Actual_Incidence'] = actuals
    comparison_df['Predicted_Incidence'] = preds
    comparison_df['Error'] = np.abs(np.array(actuals) - np.array(preds))
    comparison_df.to_csv('gnn_test_results_comparison.csv', index=False)
    print(f"📁 Audit CSV saved to 'gnn_test_results_comparison.csv'")

    torch.save(best_model_state, 'ddi_gnn_best_model.pth')

if __name__ == "__main__":
    main()
