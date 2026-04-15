# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Drophet)
Script 007: Automated Canonical SMILES Extraction (Multi-Database Support)

ENHANCED LOGIC:
This version implements a "Consensus Strategy" across two major databases:
1. PubChem PUG-REST (Primary)
2. ChEMBL REST API (Secondary/Fallback) - Excellent for clinical synonyms.
"""

import json
import os
import time
import requests
import re
import urllib.parse
from rdkit import Chem
import logging

# Configuration
INPUT_FILE = 'gpt_filteres-special-trials-w-or-final.json'
OUTPUT_FILE = 'drug_smiles_mapping.json'
IGNORED_TERMS = ['placebo', 'control', 'unknown', 'vehicle', 'study drug', 'n/a', 'none']

logging.basicConfig(level=logging.INFO, format='%(message)s')

def clean_name_aggressive(name):
    """Standard clinical cleanup for chemical databases."""
    if not name: return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\b\d+(\.\d+)?\s?(MG|ML|G|L|MCG|UNITS|IU|%|MG/ML)\b', '', name, flags=re.IGNORECASE)
    if '/' in name: name = name.split('/')[0]
    salts = ['Hydrochloride', 'HCl', 'Sodium', 'Sulfate', 'Phosphate', 'Mesylate', 'Acetate']
    for salt in salts:
        name = re.sub(rf'\b{salt}\b', '', name, flags=re.IGNORECASE)
    return name.strip().strip(',').strip('.')

def fetch_from_pubchem(term):
    """Query PubChem TXT endpoint."""
    try:
        encoded = urllib.parse.quote(term)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if res.status_code == 200:
            smiles = res.text.strip()
            if smiles and "Fault" not in smiles:
                return smiles
    except: pass
    return None

def fetch_from_chembl(term):
    """Query ChEMBL API as a fallback."""
    try:
        encoded = urllib.parse.quote(term)
        # ChEMBL Molecule API filtered by name synonym
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?molecule_synonyms__synonyms__iexact={encoded}&format=json"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            molecules = data.get('molecules', [])
            if molecules:
                # Extract the canonical SMILES from the first hit
                return molecules[0].get('molecule_structures', {}).get('canonical_smiles')
    except: pass
    return None

def get_smiles(drug_name):
    search_term = clean_name_aggressive(drug_name)
    if not search_term or search_term.lower() in IGNORED_TERMS:
        return None, "Ignored"

    # 1. Try PubChem
    smiles = fetch_from_pubchem(search_term)
    source = "PubChem"

    # 2. Fallback to ChEMBL
    if not smiles:
        smiles = fetch_from_chembl(search_term)
        source = "ChEMBL"

    if smiles:
        # Standardize via RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True), f"Success ({source})"

    return None, "Not Found"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"🚀 Initializing Multi-DB (PubChem + ChEMBL) SMILES Extraction...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_drugs = set()
    for trial in data:
        ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', trial.get('adverseEventsModule', {}))
        for g in ae_module.get('eventGroups', []):
            drugs = g.get('drugs', [])
            if isinstance(drugs, list):
                for d in drugs: unique_drugs.add(d.strip())
            elif isinstance(drugs, str):
                unique_drugs.add(drugs.strip())

    drug_list = sorted(list(unique_drugs))
    total = len(drug_list)
    print(f"📊 Querying {total} drugs...")

    mapping = []
    success_count = 0

    for i, drug in enumerate(drug_list, 1):
        print(f"[{i}/{total}] {drug}", end="", flush=True)
        smiles, status = get_smiles(drug)
        if smiles:
            print(f" -> ✅ {status}")
            success_count += 1
        else:
            print(f" -> ⚠️ {status}")
        mapping.append({"drug_name": drug, "canonical_smiles": smiles})
        time.sleep(0.3) # Rate limit protection

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)

    print(f"\n✅ Final Success Rate: {(success_count/total*100):.2f}%")

if __name__ == "__main__":
    main()
