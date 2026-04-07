# -*- coding: utf-8 -*-
"""
Created for DDI Tox-Predict Project
Phase 1: Data Acquisition & Preprocessing

This script cleans the raw Training Matrix. It strips salts, drops missing values,
and validates SMILES using RDKit for Phase 2 readiness.
"""

import pandas as pd
from rdkit import Chem
import os

def strip_salt(smiles):
    """
    只保留 SMILES 中最長的部分（通常是活性藥物成分），
    剔除點號 (.) 後面的鹽類或溶劑分子。
    """
    if isinstance(smiles, str) and '.' in smiles:
        # 以點號分割，並回傳長度最長的那一段
        return max(smiles.split('.'), key=len)
    return smiles

def validate_smiles(smiles):
    """
    Validates a SMILES string by attempting to parse it into an RDKit 
    Molecule object. Returns False if parsing fails.
    """
    if pd.isna(smiles) or not isinstance(smiles, str):
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    print("Loading raw training matrix for quality control...")
    try:
        # 確保讀取正確嘅檔名（請根據你 014 輸出嘅名調整）
        df = pd.read_csv('training_matrix_raw.csv')
    except FileNotFoundError:
        print("Error: training_matrix_raw.csv not found. Please run script 014 first.")
        return

    initial_len = len(df)
    print(f"Initial row count: {initial_len}")
    
    # Step 1: Remove rows with missing SMILES
    df_cleaned = df.dropna(subset=['Canonical_SMILES']).copy() # 使用 .copy() 避免 SettingWithCopyWarning
    dropped_missing = initial_len - len(df_cleaned)
    print(f"Dropped {dropped_missing} rows due to missing SMILES data.")
    
    # NEW STEP: Salt Stripping
    # 喺驗證同計 Feature 之前，先將 Mesylate, DMSO 等雜質剔除
    print("Applying Salt Stripping: Extracting parent molecules...")
    df_cleaned['Canonical_SMILES'] = df_cleaned['Canonical_SMILES'].apply(strip_salt)
    
    # Step 2: RDKit SMILES Validity Check
    print("Validating chemical structure integrity via RDKit...")
    df_cleaned['is_valid_smiles'] = df_cleaned['Canonical_SMILES'].apply(validate_smiles)
    
    invalid_smiles_count = len(df_cleaned[~df_cleaned['is_valid_smiles']])
    df_cleaned = df_cleaned[df_cleaned['is_valid_smiles']].copy()
    
    # Clean up the temporary validation column
    df_cleaned = df_cleaned.drop(columns=['is_valid_smiles'])
    print(f"Dropped {invalid_smiles_count} rows due to invalid RDKit SMILES parsing.")
    
    # Step 3: Boundary verification for Target Variables (Percentages)
    target_cols = [col for col in df_cleaned.columns if str(col).startswith('Target_AE_')]
    if target_cols:
        print(f"Normalizing {len(target_cols)} target variables to [0, 100] range...")
        for col in target_cols:
            df_cleaned[col] = df_cleaned[col].clip(lower=0.0, upper=100.0)

    final_len = len(df_cleaned)
    retention_rate = (final_len / initial_len) * 100 if initial_len > 0 else 0
    print(f"Final QC Passed Rows: {final_len} (Retained {retention_rate:.2f}% of raw data)")

    # Export cleaned dataset
    output_filename = 'training_matrix_cleaned.csv'
    df_cleaned.to_csv(output_filename, index=False)
    print(f"Data Quality Check complete. Cleaned dataset saved to {output_filename}")

if __name__ == "__main__":
    main()