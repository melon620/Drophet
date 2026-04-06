# -*- coding: utf-8 -*-
"""
Created for DDI Tox-Predict Project
Phase 1: Data Acquisition & Preprocessing

This script cleans the raw Training Matrix. It drops rows with missing values 
and rigorously validates the Canonical SMILES strings using RDKit to ensure 
that molecular graph extraction in Phase 2 will not fail.
"""

import pandas as pd
from rdkit import Chem

def validate_smiles(smiles):
    """
    Validates a SMILES string by attempting to parse it into an RDKit 
    Molecule object. Returns False if parsing fails.
    """
    # Check for NaN or non-string values
    if pd.isna(smiles) or not isinstance(smiles, str):
        return False
    
    # Attempt to parse the molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # If mol is None, RDKit could not parse the SMILES string
    return mol is not None

def main():
    print("Loading raw training matrix for quality control...")
    try:
        df = pd.read_csv('training_matrix_raw.csv')
    except FileNotFoundError:
        print("Error: training_matrix_raw.csv not found. Please run script 014 first.")
        return

    initial_len = len(df)
    print(f"Initial row count: {initial_len}")
    
    # Step 1: Remove data points lacking chemical features (SMILES)
    # Algorithms require input features; empty inputs are unusable.
    df_cleaned = df.dropna(subset=['Canonical_SMILES'])
    dropped_missing = initial_len - len(df_cleaned)
    print(f"Dropped {dropped_missing} rows due to missing SMILES data.")
    
    # Step 2: RDKit SMILES Validity Check
    # Ensures computational chemistry methods (e.g., Morgan Fingerprints) will succeed.
    print("Validating chemical structure integrity via RDKit...")
    df_cleaned['is_valid_smiles'] = df_cleaned['Canonical_SMILES'].apply(validate_smiles)
    
    invalid_smiles_count = len(df_cleaned[~df_cleaned['is_valid_smiles']])
    df_cleaned = df_cleaned[df_cleaned['is_valid_smiles']]
    
    # Clean up the temporary validation column
    df_cleaned = df_cleaned.drop(columns=['is_valid_smiles'])
    print(f"Dropped {invalid_smiles_count} rows due to invalid RDKit SMILES parsing.")
    
    # Step 3: Boundary verification for Target Variables (Percentages)
    # Ensures occurrence rates are strictly bound between 0.0 and 100.0 percent.
    target_cols = [col for col in df_cleaned.columns if str(col).startswith('Target_AE_')]
    for col in target_cols:
        df_cleaned[col] = df_cleaned[col].clip(lower=0.0, upper=100.0)

    final_len = len(df_cleaned)
    retention_rate = (final_len / initial_len) * 100 if initial_len > 0 else 0
    print(f"Final QC Passed Rows: {final_len} (Retained {retention_rate:.2f}% of raw data)")

    # Export cleaned dataset ready for Phase 2 (Feature Engineering)
    output_filename = 'training_matrix_cleaned.csv'
    df_cleaned.to_csv(output_filename, index=False)
    print(f"Data Quality Check complete. Cleaned dataset saved to {output_filename}")

if __name__ == "__main__":
    main()