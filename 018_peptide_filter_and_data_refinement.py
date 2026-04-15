# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 018: Peptide Filtering & Data Refinement

This script addresses the "High Error Outliers" identified in Phase 4.
It filters out large macromolecules/peptides (like Ghrelin) that bias
the small-molecule predictive engine.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

def is_peptide_or_macro(smiles):
    """
    Heuristic to identify peptides or large macromolecules.
    Peptides typically have high Nitrogen counts and high Molecular Weight.
    """
    if pd.isna(smiles) or smiles == "":
        return False

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return True # Treat invalid as noise

    mw = Descriptors.MolWt(mol)
    # Heuristic: Most small molecule drugs are < 800 Da.
    # Peptides like Ghrelin are > 3000 Da.
    if mw > 1200:
        return True

    # Count peptide bonds/nitrogens as a secondary check
    # Fixed: Changed getAtoms() to GetAtoms() (RDKit is case-sensitive)
    n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    if n_count > 15:
        return True

    return False

def main():
    print("--- Script 018: Refining Dataset for Deep Learning ---")
    input_file = 'training_matrix_cleaned.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run Phase 1 scripts first.")
        return

    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Initial dataset size: {initial_count} pairs.")

    # 1. Identify and Filter Peptides
    print("Filtering macromolecules and peptides (e.g., Ghrelin)...")
    df['is_macro_1'] = df['SMILES_1'].apply(is_peptide_or_macro)
    df['is_macro_2'] = df['SMILES_2'].apply(is_peptide_or_macro)

    df_refined = df[~(df['is_macro_1'] | df['is_macro_2'])].copy()

    peptide_count = initial_count - len(df_refined)
    print(f"Removed {peptide_count} peptide/macromolecule outliers.")

    # 2. Handle Monotherapy (Drug 2 is NaN)
    # For GNN processing, we ensure Drug 2 SMILES is an empty string rather than NaN
    df_refined['SMILES_2'] = df_refined['SMILES_2'].fillna("")

    # 3. Final Consistency Check
    # Ensure Target variable is not null
    target_cols = [c for c in df_refined.columns if c.startswith('Target_')]
    df_refined = df_refined.dropna(subset=target_cols)

    print(f"\n✅ SUCCESS: Refinement Complete.")
    print(f"📊 Final Rows for GNN: {len(df_refined)}")
    print(f"📈 Data Retention: {(len(df_refined)/initial_count*100):.2f}%")

    output_file = 'training_matrix_refined_for_gnn.csv'
    df_refined.drop(columns=['is_macro_1', 'is_macro_2']).to_csv(output_file, index=False)
    print(f"📁 Refined matrix saved to: {output_file}")

if __name__ == "__main__":
    main()
