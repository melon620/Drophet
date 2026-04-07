# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 004: Map Adverse Events to Categories (Robust Local Version)
Modified to match terms anywhere in the JSON to fix the 'Zero Matches' issue.
"""

import json
import os

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: File '{filepath}' is missing.")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error: Failed to parse JSON in '{filepath}'.")
            return None

def find_categories_in_obj(obj, term_to_category):
    """
    遞歸搜索物件中的所有字串，如果命中字典中的 Term，返回對應的 Category。
    """
    found_categories = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                v_lower = v.lower().strip()
                if v_lower in term_to_category:
                    found_categories.add(term_to_category[v_lower])
            else:
                found_categories.update(find_categories_in_obj(v, term_to_category))
    elif isinstance(obj, list):
        for item in obj:
            found_categories.update(find_categories_in_obj(item, term_to_category))
    return found_categories

def main():
    print("--- Phase 1: Robust AE Categorization Process ---")
    
    # 1. 載入輸入數據 (優先使用 011/012 產出的檔案)
    input_file = 'gpt_filteres-special-trials-w-or-final.json'
    trials_data = load_json(input_file)
    
    # 2. 載入 003 產出的字典檔
    dictionary_file = 'categorized_adverse_events.json'
    category_dict = load_json(dictionary_file)

    if trials_data is None or category_dict is None:
        print("🚨 CRITICAL ERROR: Missing input data or dictionary.")
        return

    # 建立反向查詢表 (Term -> Category)
    term_to_category = {}
    for category, terms in category_dict.items():
        for term in terms:
            term_to_category[term.lower().strip()] = category

    print(f"📖 Loaded {len(trials_data)} trials and {len(term_to_category)} dictionary terms.")

    # 3. 進行強力對接 (Deep Scan Mapping)
    updated_trials = []
    total_matches = 0

    for trial in trials_data:
        # 喺成個 trial 內容入面搵對應嘅 Category
        # 咁樣就算副作用係寫喺 eventGroups 嘅 title 度都對得到
        found_cats = find_categories_in_obj(trial, term_to_category)
        
        trial['AE_Categories'] = list(found_cats)
        if found_cats:
            total_matches += 1
            
        updated_trials.append(trial)

    # 4. 輸出結果
    output_file = 'updated_clinical_trials.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_trials, f, indent=4)

    print("\n========================================")
    print("📊 CATEGORIZATION SUMMARY (Robust Mode)")
    print("========================================")
    print(f"Trials Processed:    {len(updated_trials)}")
    print(f"Trials with Matches: {total_matches}")
    print(f"Output File:         {output_file}")
    print("========================================")

if __name__ == "__main__":
    main()