# -*- coding: utf-8 -*-
import json
import pandas as pd
import os

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found!")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"📖 Loaded {len(data)} items from {filepath}")
        return data

def get_nctid(item):
    """
    通用 NCTId 提取器：嘗試不同層級的路徑
    """
    # 1. 嘗試官方層級
    nctid = item.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
    # 2. 嘗試頂層 (如果 011/006 做咗簡化)
    if not nctid:
        nctid = item.get('nctId') or item.get('NCTId')
    return nctid

def main():
    print("--- Phase 1: Data Merging Process ---")
    
    # 1. Load Data
    distributions_data = load_json('trial_adverse_event_distributions_with_data_separated.json')
    drug_groups_data = load_json('gpt_filteres-special-trials-w-or-final.json')
    smiles_mapping = load_json('drug_smiles_mapping.json') 

    if not distributions_data:
        print("🚨 CRITICAL: distributions_data is empty. Check Script 006!")
        return

    # 2. Build SMILES Lookup
    smiles_dict = {item['drug_name']: item.get('canonical_smiles') for item in smiles_mapping if 'drug_name' in item}
    print(f"✅ SMILES dictionary built with {len(smiles_dict)} drugs.")

    # 3. Build Group-Drug Mapping
    group_drug_map = {}
    for trial in drug_groups_data:
        nct_id = get_nctid(trial)
        # 處理 011 可能存在的嵌套結構
        ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', {})
        event_groups = ae_module.get('eventGroups', [])
        
        for eg in event_groups:
            g_id = eg.get('id')
            drugs = eg.get('drugs', [])
            if nct_id and g_id:
                group_drug_map[f"{nct_id}_{g_id}"] = drugs

    print(f"✅ Group-Drug map built with {len(group_drug_map)} mapping entries.")

    # 4. Flattening Process
    flattened_records = []
    for trial in distributions_data:
        nct_id = get_nctid(trial)
        
        # 嘗試搵 AE 分佈 (容許 Script 006 出產唔同層級嘅結構)
        distributions = trial.get('adverseEventDistributions', trial) 
        serious_events = distributions.get('seriousEvents', {})
        
        if not serious_events:
            # 萬一 006 output 係直接以 NCTId 為 key 嘅 format (常見於簡化版)
            serious_events = trial.get('seriousEvents', {})

        for group_id, ae_categories in serious_events.items():
            lookup_key = f"{nct_id}_{group_id}"
            drugs_in_group = group_drug_map.get(lookup_key, [])
            
            drug_1 = drugs_in_group[0] if len(drugs_in_group) > 0 else 'Unknown'
            drug_2 = drugs_in_group[1] if len(drugs_in_group) > 1 else None
            
            record = {
                'NCTId': nct_id,
                'GroupId': group_id,
                'Drug_1': drug_1,
                'Canonical_SMILES_1': smiles_dict.get(drug_1),
                'Drug_2': drug_2,
                'Canonical_SMILES_2': smiles_dict.get(drug_2) if drug_2 else None
            }
            
            # 加入 AE Categories
            for category, percentage in ae_categories.items():
                clean_cat = str(category).replace(' ', '_').replace('/', '_')
                record[f"Target_AE_{clean_cat}"] = percentage
                
            flattened_records.append(record)

    # 5. Output
    if not flattened_records:
        print("🚨 Result is still empty. Debugging info:")
        if distributions_data:
            print(f"Sample distribution key: {list(distributions_data[0].keys())}")
        return

    df = pd.DataFrame(flattened_records)
    target_cols = [c for c in df.columns if c.startswith('Target_AE_')]
    df[target_cols] = df[target_cols].fillna(0.0)

    print(f"🚀 SUCCESS: Final matrix shape: {df.shape}")
    df.to_csv('training_matrix_raw.csv', index=False)

if __name__ == "__main__":
    main()