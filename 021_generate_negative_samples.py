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

def _canonical_pair_key(d1, d2):
    def _norm(x):
        if x is None:
            return ""
        s = str(x).strip()
        return "" if s.lower() in ("nan", "none") else s.lower()
    a, b = sorted([_norm(d1), _norm(d2)])
    return f"{a}||{b}"


def main():
    print("--- Phase 4.5: Generating Negative Controls (Safety Augmentation) ---")

    input_file = 'training_matrix_refined_for_gnn.csv'
    output_file = 'training_matrix_augmented.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find {input_file}. Please run Phase 3 scripts first.")
        return

    df_original = pd.read_csv(input_file)

    # Cross-check: any hand-picked SAFE_PAIRS that already appear in the
    # positive set (in either order) get filtered out. Otherwise we'd
    # double-label the same pair with both its observed AE rate AND a 0.0%
    # negative control, which is pure label noise.
    positive_keys = set()
    for _, row in df_original.iterrows():
        positive_keys.add(_canonical_pair_key(row.get('Drug_1'), row.get('Drug_2')))

    filtered_pairs = []
    skipped_overlap = []
    for d1, d2 in SAFE_PAIRS:
        if _canonical_pair_key(d1, d2) in positive_keys:
            skipped_overlap.append((d1, d2))
        else:
            filtered_pairs.append((d1, d2))
    if skipped_overlap:
        print(f"⚠️ Skipping {len(skipped_overlap)} hand-picked safe pairs that already "
              f"appear in the positive set: {skipped_overlap}")

    print(f"📊 Current dataset size: {len(df_original)} pairs.")
    print(f"🔬 Injecting {len(filtered_pairs)} safe control pairs at 0.0% incidence...")

    new_rows = []

    for i, (drug1, drug2) in enumerate(filtered_pairs):
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

    # Merge and Shuffle, write to a NEW file so the input is preserved.
    # 019 already reads training_matrix_augmented.csv if it exists, falling
    # back to training_matrix_refined_for_gnn.csv. This way you can rerun
    # 021 with different SAFE_PAIRS without irrecoverably mutating the
    # upstream cleaned matrix.
    df_augmented = pd.concat([df_original, df_negatives], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    df_augmented.to_csv(output_file, index=False)

    print("\n" + "="*40)
    print(f"✅ Success! Generated {len(new_rows)} valid negative samples.")
    print(f"📦 Augmented dataset written to: '{output_file}' (original '{input_file}' untouched).")
    print(f"📈 New total dataset size: {len(df_augmented)} pairs.")
    print("="*40)

    print("\n💡 Run 019 next: it will auto-pick up '{}' over '{}'.".format(output_file, input_file))

if __name__ == "__main__":
    main()
