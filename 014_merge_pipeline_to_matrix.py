# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 014: Training Matrix Assembly

This script merges three critical data sources into a single flattened CSV:
1. Adverse Event Distributions (from Script 006)
2. Drug Group Assignments (from Script 011/013)
3. Canonical SMILES Mapping (from Script 007)
"""

import json
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
AE_DIST_FILE = 'trial_adverse_event_distributions_with_data_separated.json'
DRUG_GROUP_FILE = 'gpt_filteres-special-trials-w-or-final.json'
SMILES_FILE = 'drug_smiles_mapping.json'
OUTPUT_FILE = 'training_matrix_raw.csv'

def load_json(path):
    if not os.path.exists(path):
        logging.error(f"Missing file: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    logging.info("--- Phase 1: Training Matrix Assembly ---")

    # 1. Load data sources
    ae_data = load_json(AE_DIST_FILE)
    group_data = load_json(DRUG_GROUP_FILE)
    smiles_data = load_json(SMILES_FILE)

    if not all([ae_data, group_data, smiles_data]):
        logging.error("Pipeline interrupted: Missing input files.")
        return

    # 2. Build SMILES Lookup Dictionary
    # Mapping: Drug Name -> Canonical SMILES
    smiles_lookup = {item['drug_name']: item['canonical_smiles'] for item in smiles_data if item.get('canonical_smiles')}
    logging.info(f"Loaded {len(smiles_lookup)} drug-SMILES mappings.")

    # 3. Build Drug Group Mapping
    # Mapping: "NCTId_GroupId" -> [List of Drugs]
    drug_map = {}
    for trial in group_data:
        nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId') or trial.get('nctId')
        ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', {})
        for eg in ae_module.get('eventGroups', []):
            gid = eg.get('id')
            drugs = eg.get('drugs', [])
            if nct_id and gid:
                drug_map[f"{nct_id}_{gid}"] = drugs

    # 4. Flatten and Merge
    matrix_records = []
    for entry in ae_data:
        nct_id = entry.get('nctId')
        # We focus on seriousEvents for the primary toxicity matrix
        serious_events = entry.get('seriousEvents', {})

        for gid, categories in serious_events.items():
            lookup_key = f"{nct_id}_{gid}"
            drugs = drug_map.get(lookup_key, [])

            # Identify Drug 1 and Drug 2 (if present)
            d1 = drugs[0] if len(drugs) > 0 else None
            d2 = drugs[1] if len(drugs) > 1 else None

            # Only include rows where we at least have the primary drug SMILES
            s1 = smiles_lookup.get(d1)
            if not s1:
                continue

            # Create base record
            record = {
                'NCTId': nct_id,
                'GroupId': gid,
                'Drug_1': d1,
                'SMILES_1': s1,
                'Drug_2': d2,
                'SMILES_2': smiles_lookup.get(d2) if d2 else None
            }

            # Add AE categories as target columns
            for cat, incidence in categories.items():
                col_name = f"Target_{cat.replace(' ', '_')}"
                record[col_name] = incidence

            matrix_records.append(record)

    # 5. Convert to DataFrame and Finalize
    if not matrix_records:
        logging.warning("No records were merged. Check if drug names in 007 match 011.")
        return

    df = pd.DataFrame(matrix_records)

    # Fill NaN targets with 0.0 (meaning 0% incidence reported for that category)
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    df[target_cols] = df[target_cols].fillna(0.0)

    logging.info(f"Successfully assembled matrix with {len(df)} rows and {len(df.columns)} columns.")
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Raw training matrix saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
