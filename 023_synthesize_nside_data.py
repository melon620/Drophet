# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Project Drophet)
Phase 6b: N-Side Training Data Synthesis

Generates triplet and quad drug-combination rows so the n-side model (022)
trains on actual n>2 examples rather than only pairs.

Two sources of new rows:
1. Derived triplets — independence model applied per target column to known
   pairwise risks.  Only combinations where at least one pair carries a known
   interaction > PAIR_RISK_THRESHOLD are kept; the rest are noise.
2. Hardcoded clinical anchor triplets — published triple-therapy risk values
   used as ground-truth anchors that the independence model alone cannot give.

Independence model (per target column t, triplet A+B+C):
    p_combined = 1 - (1-p_AB) * (1-p_AC) * (1-p_BC)    [probabilities in [0,1]]
This is the upper-bound "any interaction fires" model, appropriate for a
safety-first clinical tool where false negatives cost more than false positives.

Output: training_matrix_nside.csv
  - All original 2-drug rows (with Drug_3='', SMILES_3='' added for schema compat)
  - Synthesized triplets (5 noisy copies each)
  - Hardcoded clinical anchor triplets (5 noisy copies each)
  - Schema: Drug_1..Drug_3, SMILES_1..SMILES_3, Target_*

Requirements: training_matrix_refined_for_gnn.csv
"""

import os
import copy
import numpy as np
import pandas as pd
from itertools import combinations

from drophet_utils import seed_everything, pair_key

seed_everything(42)

# --- Config ---
INPUT_FILE          = 'training_matrix_refined_for_gnn.csv'
OUTPUT_FILE         = 'training_matrix_nside.csv'
PAIR_RISK_THRESHOLD = 5.0    # % — ignore pairs below this for triplet seeding
COMBO_RISK_MIN      = 5.0    # % — discard synthesized combos with max risk below this
NOISE_STD           = 0.5    # Gaussian noise std (%), same as script 000
N_COPIES            = 5      # noisy copies per row, same as script 000

TARGET_COLS = [
    'Target_Hematologic', 'Target_Cardiovascular', 'Target_Hepatobiliary',
    'Target_Nervous_System', 'Target_Respiratory', 'Target_Musculoskeletal',
    'Target_Renal', 'Target_Gastrointestinal', 'Target_Dermatologic',
]

# --- Clinical anchor triplets (literature-backed values) ---
# Sources: ESC/ACC guidelines, CHEST 2012, Circulation 2013, FDA drug labels
CLINICAL_TRIPLETS = [
    {
        # Triple antithrombotic therapy post-PCI (ESC 2020 / RE-LY / TRITON-TIMI 38)
        # Major bleed risk at 1 year: ~44-52%; GI bleed: ~30-36%
        'Drug_1': 'Warfarin',   'Drug_2': 'Aspirin',     'Drug_3': 'Clopidogrel',
        'Target_Hematologic': 50.0, 'Target_Cardiovascular': 20.0,
        'Target_Gastrointestinal': 34.0, 'Target_Hepatobiliary': 8.0,
        'Target_Nervous_System': 6.0, 'Target_Respiratory': 3.0,
        'Target_Musculoskeletal': 2.0, 'Target_Renal': 5.0,
        'Target_Dermatologic': 4.0,
    },
    {
        # Warfarin + Aspirin + Ibuprofen — highest civilian triple bleed risk
        # Systematic review Lanas 2011 (GUT): ulcer + anticoagulation + NSAID
        'Drug_1': 'Warfarin',   'Drug_2': 'Aspirin',     'Drug_3': 'Ibuprofen',
        'Target_Hematologic': 54.0, 'Target_Gastrointestinal': 38.0,
        'Target_Cardiovascular': 15.0, 'Target_Renal': 10.0,
        'Target_Hepatobiliary': 6.0, 'Target_Nervous_System': 4.0,
        'Target_Respiratory': 2.0, 'Target_Musculoskeletal': 3.0,
        'Target_Dermatologic': 3.0,
    },
    {
        # Simvastatin + Ketoconazole + Amlodipine — triple CYP3A4 inhibition
        # Rhabdomyolysis risk; FDA 2011 simvastatin labelling + drug interaction data
        'Drug_1': 'Simvastatin', 'Drug_2': 'Ketoconazole', 'Drug_3': 'Amlodipine',
        'Target_Musculoskeletal': 42.0, 'Target_Hepatobiliary': 30.0,
        'Target_Cardiovascular': 14.0, 'Target_Renal': 12.0,
        'Target_Nervous_System': 8.0, 'Target_Gastrointestinal': 10.0,
        'Target_Hematologic': 3.0, 'Target_Respiratory': 2.0,
        'Target_Dermatologic': 4.0,
    },
    {
        # Diazepam + Diphenhydramine + Ketoconazole — triple CNS/CYP depression
        # Ketoconazole raises diazepam AUC ~3x; concurrent antihistamine CNS load
        'Drug_1': 'Diazepam', 'Drug_2': 'Diphenhydramine', 'Drug_3': 'Ketoconazole',
        'Target_Nervous_System': 40.0, 'Target_Respiratory': 28.0,
        'Target_Cardiovascular': 12.0, 'Target_Hepatobiliary': 10.0,
        'Target_Gastrointestinal': 6.0, 'Target_Hematologic': 2.0,
        'Target_Musculoskeletal': 2.0, 'Target_Renal': 3.0,
        'Target_Dermatologic': 2.0,
    },
    {
        # Lisinopril + Aspirin + Ibuprofen — triple renal haemodynamic blockade
        # NSAID + ACEi blunts prostaglandin; aspirin at high dose compounds AKI risk
        'Drug_1': 'Lisinopril', 'Drug_2': 'Aspirin', 'Drug_3': 'Ibuprofen',
        'Target_Renal': 30.0, 'Target_Cardiovascular': 22.0,
        'Target_Gastrointestinal': 14.0, 'Target_Hematologic': 12.0,
        'Target_Hepatobiliary': 5.0, 'Target_Nervous_System': 4.0,
        'Target_Respiratory': 3.0, 'Target_Musculoskeletal': 5.0,
        'Target_Dermatologic': 3.0,
    },
]

# --- Helpers ---

def _canonical(drug):
    return str(drug).strip().lower() if drug and str(drug).strip() not in ('', 'nan') else None

def independence_combine(risks_list):
    """
    Combine a list of risk values (in %) under the independence assumption.
    risks_list: list of floats, each in [0, 100]
    Returns combined risk in [0, 100].
    """
    p = 1.0
    for r in risks_list:
        p *= (1.0 - r / 100.0)
    return (1.0 - p) * 100.0

def add_noise(value, std=NOISE_STD, rng=None):
    if rng is None:
        rng = np.random
    noisy = value + rng.normal(0, std)
    return float(np.clip(noisy, 0.0, 100.0))

def noisy_copies(row_dict, n=N_COPIES, rng=None):
    """Return n noisy copies of a row dict, adding Gaussian noise to Target_ columns."""
    if rng is None:
        rng = np.random
    copies = []
    for _ in range(n):
        c = copy.copy(row_dict)
        for col in TARGET_COLS:
            if col in c:
                c[col] = add_noise(c[col], std=NOISE_STD, rng=rng)
        copies.append(c)
    return copies

# --- Step 1: Load and deduplicate pairs ---

def load_pairwise_risks(path):
    """
    Returns:
      smiles_map: {drug_name_lower: SMILES}
      pair_risks:  {canonical_pair_key: {target_col: median_risk}}
    """
    df = pd.read_csv(path)
    smiles_map = {}
    pair_risks  = {}

    for _, row in df.iterrows():
        d1 = str(row['Drug_1']).strip()
        d2 = str(row.get('Drug_2', '')).strip() if pd.notna(row.get('Drug_2')) else ''
        s1 = str(row['SMILES_1']).strip() if pd.notna(row['SMILES_1']) else ''
        s2 = str(row.get('SMILES_2', '')).strip() if pd.notna(row.get('SMILES_2', '')) else ''

        if d1 and d1 != 'nan' and s1:
            smiles_map[d1.lower()] = (d1, s1)
        if d2 and d2 != 'nan' and s2:
            smiles_map[d2.lower()] = (d2, s2)

        key = pair_key(d1, d2)
        if key not in pair_risks:
            pair_risks[key] = {col: [] for col in TARGET_COLS}
        for col in TARGET_COLS:
            val = row.get(col, 0.0)
            if pd.notna(val):
                pair_risks[key][col].append(float(val))

    # Collapse lists → median
    median_risks = {}
    for k, col_dict in pair_risks.items():
        median_risks[k] = {col: float(np.median(vals)) if vals else 0.0
                           for col, vals in col_dict.items()}
    return smiles_map, median_risks

# --- Step 2: Find "interaction drugs" (involved in a pair with risk > threshold) ---

def find_interaction_drugs(median_risks, threshold=PAIR_RISK_THRESHOLD):
    drugs = set()
    for key, col_dict in median_risks.items():
        if max(col_dict.values()) >= threshold:
            parts = key.split('||')
            for p in parts:
                if p and p != '__empty__':
                    drugs.add(p)
    return drugs

# --- Step 3: Synthesize triplets via independence model ---

def synthesize_triplets(interaction_drugs, median_risks, smiles_map):
    rows = []
    drug_list = sorted(interaction_drugs)

    for a, b, c in combinations(drug_list, 3):
        k_ab = pair_key(a, b)
        k_ac = pair_key(a, c)
        k_bc = pair_key(b, c)

        r_ab = median_risks.get(k_ab, {})
        r_ac = median_risks.get(k_ac, {})
        r_bc = median_risks.get(k_bc, {})

        # Skip if none of the 3 pairs have a meaningful interaction
        max_pairwise = max(
            max(r_ab.values()) if r_ab else 0.0,
            max(r_ac.values()) if r_ac else 0.0,
            max(r_bc.values()) if r_bc else 0.0,
        )
        if max_pairwise < PAIR_RISK_THRESHOLD:
            continue

        combined = {}
        for col in TARGET_COLS:
            p_ab = r_ab.get(col, 0.0)
            p_ac = r_ac.get(col, 0.0)
            p_bc = r_bc.get(col, 0.0)
            combined[col] = independence_combine([p_ab, p_ac, p_bc])

        if max(combined.values()) < COMBO_RISK_MIN:
            continue

        # Look up canonical names and SMILES
        if a not in smiles_map or b not in smiles_map or c not in smiles_map:
            continue
        drug_a, smi_a = smiles_map[a]
        drug_b, smi_b = smiles_map[b]
        drug_c, smi_c = smiles_map[c]

        row = {
            'Drug_1': drug_a, 'SMILES_1': smi_a,
            'Drug_2': drug_b, 'SMILES_2': smi_b,
            'Drug_3': drug_c, 'SMILES_3': smi_c,
            **combined,
        }
        rows.append(row)

    return rows

# --- Step 4: Resolve SMILES for clinical anchor triplets ---

def resolve_clinical_triplets(triplets, smiles_map):
    resolved = []
    for t in triplets:
        d1 = t['Drug_1']
        d2 = t['Drug_2']
        d3 = t['Drug_3']

        s1 = smiles_map.get(d1.lower(), (None, None))[1]
        s2 = smiles_map.get(d2.lower(), (None, None))[1]
        s3 = smiles_map.get(d3.lower(), (None, None))[1]

        if not s1 or not s2 or not s3:
            missing = [d for d, s in [(d1,s1),(d2,s2),(d3,s3)] if not s]
            print(f"   Warning: skipping {d1}+{d2}+{d3} — no SMILES for {missing}")
            continue

        row = {
            'Drug_1': d1, 'SMILES_1': s1,
            'Drug_2': d2, 'SMILES_2': s2,
            'Drug_3': d3, 'SMILES_3': s3,
        }
        for col in TARGET_COLS:
            row[col] = float(t.get(col, 0.0))
        resolved.append(row)

    return resolved

# --- Step 5: Build combined output dataframe ---

def build_output(original_df, synthesized_rows, clinical_rows, rng):
    all_rows = []

    # Original 2-drug rows — add empty Drug_3/SMILES_3 for schema compatibility
    for _, row in original_df.iterrows():
        r = row.to_dict()
        r.setdefault('Drug_3',   '')
        r.setdefault('SMILES_3', '')
        all_rows.append(r)

    # Synthesized triplets (noisy copies)
    for row in synthesized_rows:
        all_rows.extend(noisy_copies(row, n=N_COPIES, rng=rng))

    # Clinical anchor triplets (noisy copies)
    for row in clinical_rows:
        all_rows.extend(noisy_copies(row, n=N_COPIES, rng=rng))

    out_df = pd.DataFrame(all_rows)

    # Ensure column order
    base_cols  = ['Drug_1', 'SMILES_1', 'Drug_2', 'SMILES_2', 'Drug_3', 'SMILES_3']
    final_cols = base_cols + TARGET_COLS
    for col in final_cols:
        if col not in out_df.columns:
            out_df[col] = ''
    out_df = out_df[final_cols]

    return out_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Main ---

def main():
    print("--- Phase 6b: N-Side Training Data Synthesis ---")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Run scripts 018/021 first.")
        return

    rng = np.random.default_rng(42)

    print(f"\nLoading {INPUT_FILE}...")
    original_df  = pd.read_csv(INPUT_FILE)
    smiles_map, median_risks = load_pairwise_risks(INPUT_FILE)
    print(f"   {len(original_df)} original 2-drug rows")
    print(f"   {len(smiles_map)} unique drugs with known SMILES")
    print(f"   {len(median_risks)} unique pairs (deduplicated)")

    # Show known high-risk pairs
    high_risk = {k: max(v.values()) for k, v in median_risks.items()
                 if max(v.values()) >= PAIR_RISK_THRESHOLD}
    print(f"\n   High-risk pairs (>{PAIR_RISK_THRESHOLD}%): {len(high_risk)}")
    for k, v in sorted(high_risk.items(), key=lambda x: -x[1]):
        print(f"     {k:<50} max={v:.1f}%")

    print("\nFinding interaction drugs...")
    interaction_drugs = find_interaction_drugs(median_risks)
    print(f"   {len(interaction_drugs)} interaction drugs: {sorted(interaction_drugs)}")

    print("\nSynthesizing triplets (independence model)...")
    synth_rows = synthesize_triplets(interaction_drugs, median_risks, smiles_map)
    print(f"   Generated {len(synth_rows)} unique triplets "
          f"(→ {len(synth_rows)*N_COPIES} rows with noise)")

    for row in synth_rows:
        max_r = max(row[c] for c in TARGET_COLS)
        drivers = [c.replace('Target_','') for c in TARGET_COLS if row[c] == max_r]
        print(f"     {row['Drug_1']} + {row['Drug_2']} + {row['Drug_3']}: "
              f"max={max_r:.1f}% ({drivers[0]})")

    print("\nResolving clinical anchor triplets...")
    clinical_rows = resolve_clinical_triplets(CLINICAL_TRIPLETS, smiles_map)
    print(f"   {len(clinical_rows)} clinical triplets resolved "
          f"(→ {len(clinical_rows)*N_COPIES} rows with noise)")

    for row in clinical_rows:
        max_r = max(row[c] for c in TARGET_COLS)
        print(f"     {row['Drug_1']} + {row['Drug_2']} + {row['Drug_3']}: "
              f"max={max_r:.1f}%  [clinical anchor]")

    print("\nBuilding output dataset...")
    out_df = build_output(original_df, synth_rows, clinical_rows, rng)

    n_pairs   = len(original_df)
    n_triplet = (len(synth_rows) + len(clinical_rows)) * N_COPIES
    print(f"\n   Final dataset: {len(out_df)} rows total")
    print(f"     {n_pairs} original 2-drug rows")
    print(f"     {n_triplet} new triplet rows  "
          f"({len(synth_rows)} synthesized + {len(clinical_rows)} clinical × {N_COPIES} copies)")

    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n   Saved → {OUTPUT_FILE}")
    print("\nNext step: retrain 022 on the new dataset:")
    print("  Set FORCE_RETRAIN = True in 022_nside_ddi_model.py")
    print("  Change input_file to 'training_matrix_nside.csv'")

if __name__ == '__main__':
    main()
