# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 4.5: Negative Sampling (Injecting Safety Controls)

This script generates a list of clinically safe drug pairs (Negative Controls),
fetches their SMILES from PubChem, labels their incidence rate as 0.0%,
and appends them to the refined training matrix.

UPDATE: Automatically backs up old data, overwrites the target CSV,
and deletes old model weights to FORCE retraining in Phase 5.
"""

import pandas as pd
import requests
import urllib.parse
from rdkit import Chem
import os
import time
import shutil

# --- 1. Curated List of Safe Drug Pairs (Negative Controls) ---
# These are pairs with no known severe systemic pharmacokinetic interactions.
SAFE_PAIRS = [
    ("Paracetamol", "Vitamin C"),
    ("Ibuprofen", "Vitamin B12"),
    ("Amoxicillin", "Vitamin D3"),
    ("Loratadine", "Calcium Carbonate"),
    ("Cetirizine", "Magnesium Oxide"),
    ("Omeprazole", "Vitamin E"),
    ("Metformin", "Folic Acid"),
    ("Aspirin", "Biotin"),
    ("Pantoprazole", "Zinc Sulfate"),
    ("Simvastatin", "Vitamin K1"),
    ("Atorvastatin", "Riboflavin"),
    ("Levothyroxine", "Thiamine"),
    ("Amlodipine", "Niacinamide"),
    ("Losartan", "Pyridoxine"),
    ("Metoprolol", "Vitamin A"),
    ("Albuterol", "Vitamin C"),
    ("Fluticasone", "Vitamin D3"),
    ("Gabapentin", "Vitamin B12"),
    ("Sertraline", "Folic Acid"),
    ("Citalopram", "Biotin"),
    ("Fluoxetine", "Vitamin E"),
    ("Tamsulosin", "Zinc Sulfate"),
    ("Montelukast", "Vitamin K1"),
    ("Clopidogrel", "Riboflavin"),
    ("Meloxicam", "Thiamine"),
    ("Diclofenac", "Niacinamide"),
    ("Celecoxib", "Pyridoxine"),
    ("Tramadol", "Vitamin A"),
    ("Hydrochlorothiazide", "Vitamin C"),
    ("Furosemide", "Vitamin D3")
]

# --- 2. Helper Functions ---

def fetch_smiles(drug_name):
    """Fetches canonical SMILES from PubChem API with a small delay to avoid rate limits."""
    try:
        name = drug_name.strip()
        encoded = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            smiles = res.text.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"   ⚠️ Could not fetch SMILES for {drug_name}: {e}")
    return None

# --- 3. Main Execution ---

def main():
    print("--- Phase 4.5: Generating Negative Controls & Resetting Environment ---")

    input_file = 'training_matrix_refined_for_gnn.csv'
    backup_file = 'training_matrix_refined_for_gnn_backup.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find {input_file}. Please run Phase 3 scripts first.")
        return

    # 1. Backup the original file safely
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"💾 Backed up original matrix to '{backup_file}'")

    df_original = pd.read_csv(input_file)

    print(f"📊 Current dataset size: {len(df_original)} pairs.")
    print(f"🔬 Injecting {len(SAFE_PAIRS)} safe control pairs at 0.0% incidence...")

    new_rows = []

    for i, (drug1, drug2) in enumerate(SAFE_PAIRS):
        print(f"[{i+1}/{len(SAFE_PAIRS)}] Processing: {drug1} + {drug2}")
        s1 = fetch_smiles(drug1)
        time.sleep(0.2) # Polite delay for API
        s2 = fetch_smiles(drug2)
        time.sleep(0.2)

        if s1 and s2:
            # Create a row initializing all targets to 0.0
            row = {col: 0.0 for col in df_original.columns}
            row['Drug_1'] = drug1
            row['Drug_2'] = drug2
            row['SMILES_1'] = s1
            row['SMILES_2'] = s2
            new_rows.append(row)
        else:
            print(f"   ⏭️ Skipped {drug1} + {drug2} due to missing SMILES.")

    df_negatives = pd.DataFrame(new_rows)

    # 2. Merge and Shuffle
    df_augmented = pd.concat([df_original, df_negatives], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

    # OVERWRITE the original file so Script 019 picks it up automatically
    df_augmented.to_csv(input_file, index=False)

    print("\n" + "="*40)
    print(f"✅ Success! Generated {len(new_rows)} valid negative samples.")
    print(f"📦 Overwritten active dataset: '{input_file}'")
    print(f"📈 New total dataset size: {len(df_augmented)} pairs.")
    print("="*40)

    # 3. FORCE RETRAINING by deleting old artifacts
    print("\n🧹 Cleaning up old artifacts to force retraining...")
    artifacts_to_delete = ['ddi_gnn_best_model.pth', 'target_scaler.pkl']
    deleted_count = 0
    for artifact in artifacts_to_delete:
        if os.path.exists(artifact):
            os.remove(artifact)
            print(f"   🗑️ Deleted: {artifact}")
            deleted_count += 1

    if deleted_count > 0:
        print("💡 The GNN will now automatically retrain on the augmented data next time you run 019.")
    else:
        print("💡 No old artifacts found. Ready for fresh training.")

    print("\n👉 Next Step: Run 'python 019_train_gnn_model.py' to fine-tune the model with the new baseline!")

if __name__ == "__main__":
    main()
