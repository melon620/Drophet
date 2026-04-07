# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 2: Feature Engineering (Molecular Representation)

This script transforms SMILES strings from the cleaned matrix into numerical 
features: Morgan Fingerprints (ECFP4) and Physicochemical Descriptors.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os

def get_features_from_smiles(smiles, n_bits=1024):
    """
    Calculates Morgan Fingerprints and basic descriptors for a single SMILES.
    """
    if pd.isna(smiles) or smiles == "":
        return [0] * (n_bits + 5) # Return zeros if missing
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * (n_bits + 5)
    
    # 1. Morgan Fingerprint (Radius 2 = ECFP4)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    fp_list = [int(b) for b in fp.ToBitString()]
    
    # 2. Physicochemical Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    
    return fp_list + [mw, logp, tpsa, h_donors, h_acceptors]

def main():
    print("--- Phase 2: Generating Molecular Features ---")
    input_file = 'training_matrix_cleaned.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} cleaned drug pairs.")

    n_bits = 1024 # Standard bit size for balance between depth and performance
    
    feature_cols_1 = [f"D1_Bit_{i}" for i in range(n_bits)] + ["D1_MW", "D1_LogP", "D1_TPSA", "D1_HDonors", "D1_HAcceptors"]
    feature_cols_2 = [f"D2_Bit_{i}" for i in range(n_bits)] + ["D2_MW", "D2_LogP", "D2_TPSA", "D2_HDonors", "D2_HAcceptors"]

    print("Extracting features for Drug 1...")
    d1_features = df['Canonical_SMILES_1'].apply(lambda x: get_features_from_smiles(x, n_bits))
    d1_df = pd.DataFrame(d1_features.tolist(), columns=feature_cols_1)

    print("Extracting features for Drug 2...")
    d2_features = df['Canonical_SMILES_2'].apply(lambda x: get_features_from_smiles(x, n_bits))
    d2_df = pd.DataFrame(d2_features.tolist(), columns=feature_cols_2)

    # Combine metadata, targets, and new features
    # Keep NCTId, GroupId and Target columns
    target_cols = [col for col in df.columns if col.startswith('Target_AE_')]
    metadata_cols = ['NCTId', 'GroupId', 'Drug_1', 'Drug_2']
    
    final_df = pd.concat([df[metadata_cols], d1_df, d2_df, df[target_cols]], axis=1)

    output_filename = 'ddi_training_dataset_final.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ SUCCESS: Feature Engineering Complete.")
    print(f"📊 Final Dataset Shape: {final_df.shape}")
    print(f"📁 Training dataset saved to: {output_filename}")

if __name__ == "__main__":
    main() 