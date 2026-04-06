# -*- coding: utf-8 -*-
"""Phase 1: Data Acquisition & Preprocessing"""

import json
import pandas as pd
import os

def load_json(filepath):
    """Utility function to load JSON files safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found.")
        return []

def main():
    print("Loading intermediate data files from previous pipeline steps...")
    
    # Load the distributions from Script 006
    trials_data = load_json('trial_adverse_event_distributions_with_data_separated.json')
    
    # Load the extracted SMILES mapping from Script 007
    # Assuming format: [{"drug_name": "Tenofovir", "canonical_smiles": "CC(C)OC(=O)..."}, ...]
    smiles_mapping = load_json('drug_smiles_mapping.json') 
    
    # Create a dictionary for O(1) lookup time
    smiles_dict = {}
    for item in smiles_mapping:
        if isinstance(item, dict) and 'drug_name' in item:
            smiles_dict[item['drug_name']] = item.get('canonical_smiles')

    flattened_records = []

    print("Flattening hierarchical JSON structures into a tabular format...")
    
    # Iterate through trials and extract target variables
    for trial in trials_data:
        # Safely extract NCT ID
        protocol_section = trial.get('protocolSection', {})
        identification_module = protocol_section.get('identificationModule', {})
        nct_id = identification_module.get('nctId', 'Unknown')
        
        # Extract distributions
        distributions = trial.get('adverseEventDistributions', {})
        serious_events = distributions.get('seriousEvents', {})
        
        # Iterate over each event group (e.g., EG000)
        for group_id, ae_categories in serious_events.items():
            
            # Base record instantiation
            # In a full pipeline, map the specific drug name to the group_id using outputs from script 011
            record = {
                'NCTId': nct_id,
                'GroupId': group_id,
                'DrugName': 'Placeholder_Drug', # Replace with matching logic from script 011
            }
            
            # Inject Chemical Feature (Canonical SMILES)
            record['Canonical_SMILES'] = smiles_dict.get(record['DrugName'], None)
            
            # Unroll Target Variables (Adverse Event Categories percentage affected)
            # Convert categories to separate columns
            for category, percentage in ae_categories.items():
                # Clean column name by replacing spaces with underscores
                clean_category = str(category).replace(' ', '_').replace('/', '_')
                col_name = f"Target_AE_{clean_category}"
                record[col_name] = percentage
                
            flattened_records.append(record)

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(flattened_records)
    
    # Fill missing target variables with 0.0% (Assuming unreported equals 0% incidence)
    target_cols = [col for col in df.columns if str(col).startswith('Target_AE_')]
    df[target_cols] = df[target_cols].fillna(0.0)

    print(f"Flattened matrix shape before export: {df.shape}")
    
    # Export to CSV (Raw Training Matrix)
    output_filename = 'training_matrix_raw.csv'
    df.to_csv(output_filename, index=False)
    print(f"Successfully exported Raw Training Matrix to {output_filename}")

if __name__ == "__main__":
    main()