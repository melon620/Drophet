# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 003: Local Categorizer (SciSpacy Light Edition)
Optimized for low-RAM environments (GitHub Codespaces)
"""

import json
import os
import spacy
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict

def recursive_find_terms(obj):
    extracted = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ['term', 'title'] and isinstance(v, str):
                if len(v) > 2 and not v.startswith("NCT"): 
                    extracted.add(v.strip())
            else:
                extracted.update(recursive_find_terms(v))
    elif isinstance(obj, list):
        for item in obj:
            extracted.update(recursive_find_terms(item))
    return extracted

def main():
    input_file = 'special-trials.json'
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    print(f"📖 Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        trials = json.load(f)

    event_list = sorted(list(recursive_find_terms(trials)))
    print(f"✅ Extracted {len(event_list)} raw clinical terms.")

    # 轉用 'sm' 模型以節省 RAM
    print("🧠 Loading Light Medical NLP Model (en_core_sci_sm)...")
    try:
        nlp = spacy.load("en_core_sci_sm")
    except OSError:
        print("❌ Error: Model 'en_core_sci_sm' not found. Please install it first.")
        return

    # 減少 Linker 載入嘅候選數量，進一步節省記憶體
    print("🔗 Adding UMLS Entity Linker (Optimized)...")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls",
        "max_entities_per_mention": 1 # 只攞最準嗰個，慳 RAM
    })
    
    final_mapping = defaultdict(list)

    print(f"🚀 Processing {len(event_list)} terms locally...")
    
    for i, term in enumerate(event_list):
        doc = nlp(term)
        
        if doc.ents:
            main_ent = doc.ents[0]
            category = "General Disorders"
            
            t = term.lower()
            # 關鍵字映射優先 (Keyword-first approach)
            if any(x in t for x in ['liver', 'alt', 'ast', 'bilirubin', 'hepatic']):
                category = "Hepatobiliary Disorders"
            elif any(x in t for x in ['nausea', 'vomiting', 'diarrhea', 'gastric', 'gi', 'abdominal']):
                category = "Gastrointestinal Disorders"
            elif any(x in t for x in ['headache', 'dizziness', 'neuropathy', 'seizure', 'somnolence']):
                category = "Neurologic Disorders"
            elif any(x in t for x in ['rash', 'skin', 'dermatitis', 'pruritus', 'urticaria']):
                category = "Dermatologic Disorders"
            elif any(x in t for x in ['heart', 'cardiac', 'vascular', 'hypertension', 'blood pressure']):
                category = "Cardiovascular Disorders"
            elif any(x in t for x in ['renal', 'kidney', 'creatinine', 'urinary']):
                category = "Renal/Urinary Disorders"
            elif any(x in t for x in ['fatigue', 'pyrexia', 'fever', 'chills']):
                category = "General Disorders"
            else:
                category = f"Systemic_{main_ent.label_}"
            
            final_mapping[category].append(term)
        
        if (i+1) % 50 == 0:
            print(f"   Done {i+1}/{len(event_list)}...")

    with open('categorized_adverse_events.json', 'w', encoding='utf-8') as f:
        json.dump(dict(final_mapping), f, indent=4)
        
    with open('input-gpt-prompt-organsys.json', 'w', encoding='utf-8') as f:
        json.dump(dict(final_mapping), f, indent=4)

    print(f"✨ SUCCESS: Saved {len(final_mapping)} categories locally.")

if __name__ == "__main__":
    main()