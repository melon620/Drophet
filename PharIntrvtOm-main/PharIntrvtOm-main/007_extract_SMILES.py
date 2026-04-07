# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 007: Professional SMILES Extractor (with Success Rate Stats)
"""

import json
import requests
import re
import time
import os
from rdkit import Chem

def clean_drug_name(name):
    """清洗藥物名稱，移除劑量、單位、劑型等雜訊"""
    if not name or not isinstance(name, str):
        return ""
    cleaned = name.lower()
    cleaned = re.sub(r'\d+\s?(mg|ml|kg|g|mcg|unit|u|mcl)(/\w+)?', '', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    noise_words = ['tablets', 'tablet', 'capsules', 'capsule', 'injection', 'oral', 'solution', 'plus', 'iv', 'dose']
    for word in noise_words:
        cleaned = re.sub(rf'\b{word}\b', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_smiles_from_pubchem(name):
    """透過 PubChem API 獲取 SMILES"""
    search_terms = [clean_drug_name(name), name]
    search_terms = list(dict.fromkeys(search_terms))
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'
    
    for term in search_terms:
        if len(term) < 2: continue
        encoded_term = requests.utils.quote(term)
        url = f"{base_url}{encoded_term}/property/CanonicalSMILES/TXT"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()
        except:
            continue
    return None

def main():
    input_file = 'gpt_filteres-special-trials-w-or-final.json'
    output_file = 'drug_smiles_mapping.json'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        trials = json.load(f)

    unique_drugs = set()
    for trial in trials:
        ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', {})
        for event_group in ae_module.get('eventGroups', []):
            for drug in event_group.get('drugs', []):
                if drug and drug.lower() not in ['placebo', 'control', 'unknown']:
                    unique_drugs.add(drug)

    drug_list = sorted(list(unique_drugs))
    total_count = len(drug_list)
    found_count = 0
    failed_count = 0
    
    print(f"🚀 Starting SMILES extraction for {total_count} unique drugs...")

    smiles_mapping = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            smiles_mapping = json.load(f)
    
    processed_names = {item['drug_name'] for item in smiles_mapping}

    for i, name in enumerate(drug_list):
        if name in processed_names:
            # 統計返已經跑完嘅數據
            existing_smiles = next(item['canonical_smiles'] for item in smiles_mapping if item['drug_name'] == name)
            if existing_smiles: found_count += 1
            else: failed_count += 1
            continue
            
        print(f"[{i+1}/{total_count}] Processing: {name}")
        raw_smiles = get_smiles_from_pubchem(name)
        
        if raw_smiles:
            mol = Chem.MolFromSmiles(raw_smiles)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True) if mol else None
        else:
            canonical_smiles = None
        
        if canonical_smiles:
            found_count += 1
            smiles_mapping.append({"drug_name": name, "canonical_smiles": canonical_smiles})
            print(f"   ✅ Success")
        else:
            failed_count += 1
            smiles_mapping.append({"drug_name": name, "canonical_smiles": None})
            print(f"   ⚠️ Failed")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(smiles_mapping, f, indent=4)
        time.sleep(0.4)

    # --- 統計輸出 (Success Rate Stats) ---
    success_rate = (found_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n" + "="*40)
    print("📊 EXTRACTION SUMMARY")
    print("="*40)
    print(f"Total Unique Drugs:  {total_count}")
    print(f"SMILES Found:       {found_count}")
    print(f"SMILES Missing:     {failed_count}")
    print(f"Success Rate:       {success_rate:.2f}%")
    print("="*40)
    print(f"✨ Data saved to {output_file}\n")

if __name__ == "__main__":
    main()