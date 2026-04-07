# -*- coding: utf-8 -*-
"""
Created for DDI Tox-Predict Project
Phase 1: Data Acquisition & Preprocessing

This script cleans the raw Training Matrix. It validates the Canonical SMILES 
for both drugs in a pair using RDKit to ensure Phase 2 compatibility.
"""

import pandas as pd
from rdkit import Chem
import os

def validate_smiles(smiles):
    """
    Validates a SMILES string by attempting to parse it into an RDKit 
    Molecule object. Returns False if parsing fails.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return False
    
    # Attempt to parse the molecule
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    print("--- Phase 1: Data Quality Control (DDI Matrix) ---")
    input_file = 'training_matrix_raw.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Please run script 014 first.")
        return

    df = pd.read_csv(input_file)
    initial_len = len(df)
    print(f"Initial row count: {initial_len}")

    # Step 1: Remove data points lacking primary chemical features
    # Drug_1 is mandatory for the model input.
    df_cleaned = df.dropna(subset=['Canonical_SMILES_1']).copy()
    dropped_missing = initial_len - len(df_cleaned)
    print(f"Dropped {dropped_missing} rows due to missing Primary SMILES.")

    # Step 2: RDKit SMILES Validity Check for both drugs
    print("Validating chemical structure integrity via RDKit...")
    
    # Check Drug 1
    df_cleaned['is_valid_1'] = df_cleaned['Canonical_SMILES_1'].apply(validate_smiles)
    
    # Check Drug 2 (only if Drug_2 column exists and is not null)
    if 'Canonical_SMILES_2' in df_cleaned.columns:
        df_cleaned['is_valid_2'] = df_cleaned['Canonical_SMILES_2'].apply(
            lambda x: validate_smiles(x) if pd.notna(x) else True
        )
    else:
        df_cleaned['is_valid_2'] = True

    # Drop invalid rows
    invalid_mask = (~df_cleaned['is_valid_1']) | (~df_cleaned['is_valid_2'])
    invalid_count = len(df_cleaned[invalid_mask])
    df_cleaned = df_cleaned[~invalid_mask].copy()

    # Drop temporary columns
    df_cleaned = df_cleaned.drop(columns=['is_valid_1', 'is_valid_2'])
    print(f"Dropped {invalid_count} rows due to invalid RDKit SMILES parsing.")

    # Step 3: Boundary verification for Target Variables (Percentages)
    target_cols = [col for col in df_cleaned.columns if str(col).startswith('Target_AE_')]
    for col in target_cols:
        df_cleaned[col] = df_cleaned[col].apply(lambda x: min(max(float(x), 0.0), 100.0))

    final_len = len(df_cleaned)
    retention_rate = (final_len / initial_len) * 100 if initial_len > 0 else 0
    print(f"\n✅ SUCCESS: QC Passed. Final Row Count: {final_len}")
    print(f"📊 Retention Rate: {retention_rate:.2f}%")

    output_filename = 'training_matrix_cleaned.csv'
    df_cleaned.to_csv(output_filename, index=False)
    print(f"📁 Cleaned dataset saved to: {output_filename}")

if __name__ == "__main__":
    main() 