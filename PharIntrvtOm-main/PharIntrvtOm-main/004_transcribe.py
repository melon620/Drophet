# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 004: Map Adverse Events to Categories (Gemini Compatible)

This script acts as a 'bridge' between raw clinical trial data and 
organized medical categories. It requires the dictionary from Script 003.
"""

import json
import os

def load_json(filepath):
    """Safely load JSON and provide diagnostic feedback."""
    if not os.path.exists(filepath):
        print(f"⚠️  Missing: '{filepath}'")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"❌ Error: Could not parse '{filepath}'. File might be corrupted.")
            return None

def main():
    print("--- Phase 1: AE Categorization Process ---")
    
    # 1. 確定 Input File (優先用 012 嘅 ICD 檔案，冇就用 011)
    input_options = ['gpt_added_ICD.json', 'gpt_filteres-special-trials-w-or-final.json']
    trials_data = None
    used_input = ""

    for opt in input_options:
        trials_data = load_json(opt)
        if trials_data is not None:
            used_input = opt
            break

    # 2. 讀取副作用字典 (由 Script 003 產生)
    dictionary_file = 'categorized_adverse_events.json'
    category_dict = load_json(dictionary_file)

    # --- 關鍵診斷 (Critical Diagnostics) ---
    if trials_data is None:
        print("🚨 CRITICAL: No trial data found! Ensure Script 011 or 012 has finished.")
        return
    
    if category_dict is None:
        print(f"🚨 CRITICAL: Dictionary '{dictionary_file}' is missing!")
        print("💡 Solution: Run 'python 003_gemini_categorize_events.py' first.")
        return

    # 3. 建立反向查詢表 (Term -> Category)
    term_to_category = {}
    for category, terms in category_dict.items():
        for term in terms:
            term_to_category[term.lower().strip()] = category

    print(f"📖 Loaded {len(trials_data)} trials from '{used_input}'.")
    print(f"📖 Loaded {len(term_to_category)} unique terms from dictionary.")

    # 4. 開始對接
    updated_trials = []
    match_count = 0
    missing_terms = set()

    for trial in trials_data:
        ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', {})
        trial_ae_categories = set()
        
        for event_type in ['seriousEvents', 'otherEvents']:
            events = ae_module.get(event_type, [])
            for event in events:
                term = event.get('term', '').lower().strip()
                if term in term_to_category:
                    category = term_to_category[term]
                    trial_ae_categories.add(category)
                    event['mapped_category'] = category
                    match_count += 1
                else:
                    if term: missing_terms.add(term)
        
        # 將搵到嘅分類寫入頂層
        trial['AE_Categories'] = list(trial_ae_categories)
        updated_trials.append(trial)

    # 5. 輸出結果
    output_file = 'updated_clinical_trials.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_trials, f, indent=4)

    print("\n" + "="*40)
    print("📊 CATEGORIZATION SUMMARY")
    print("="*40)
    print(f"Trials Processed:   {len(updated_trials)}")
    print(f"Successful Matches: {match_count}")
    print(f"Unmapped Terms:     {len(missing_terms)}")
    print(f"Output File:        {output_file}")

    print("="*40)
    
    if match_count == 0:
        print("⚠️ Warning: Zero matches found. Check if your dictionary terms match your trial data.")

if __name__ == "__main__":
    main()