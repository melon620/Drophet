# -*- coding: utf-8 -*-
import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

def recursive_find_terms(obj):
    """
    唔理個 JSON 有幾深，只要見到 'term' 就擸。
    如果搵唔到 'term'，就擸埋 'title' 同 'description' 做 Candidate。
    """
    extracted = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ['term', 'title'] and isinstance(v, str):
                if len(v) > 2: # 避開太短嘅 ID
                    extracted.add(v.strip())
            else:
                extracted.update(recursive_find_terms(v))
    elif isinstance(obj, list):
        for item in obj:
            extracted.update(recursive_find_terms(item))
    return extracted

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is missing!")
        return

    client = genai.Client(api_key=api_key)
    model_id = 'gemini-2.0-flash' # 快、平、準

    input_file = 'special-trials.json'
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    print(f"📖 Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        trials = json.load(f)

    # 1. 深度提取 (Deep Extraction)
    print("🔎 Deep scanning for terms...")
    all_potential_terms = recursive_find_terms(trials)
    
    # 移除一啲明顯唔係副作用嘅字 (例如 NCT ID)
    event_list = sorted([t for t in all_potential_terms if not t.startswith("NCT")])
    print(f"✅ Extracted {len(event_list)} potential terms.")

    if not event_list:
        print("🚨 CRITICAL: Still 0 terms found. Check your JSON again.")
        return

    # 2. Gemini Categorization Logic
    # 呢度個 Prompt 好重要：叫佢幫你分邊啲係 AE，邊啲係藥名(藥名要掉咗佢)
    prompt = """
    Analyze the following clinical terms. 
    1. Identify which ones are Adverse Events (AEs).
    2. Categorize them into MedDRA System Organ Classes (SOCs).
    3. Discard any terms that are drug names, study titles, or descriptions.
    Return ONLY a JSON dictionary: {"SOC Name": ["Term1", "Term2"]}
    """

    batch_size = 50
    final_mapping = {}

    for i in range(0, len(event_list), batch_size):
        batch = event_list[i:i+batch_size]
        print(f"📦 Processing Batch {i//batch_size + 1}/{ (len(event_list)//batch_size)+1 }...")
        
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=f"{prompt}\nTerms: {', '.join(batch)}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                batch_data = json.loads(response.text)
                for soc, terms in batch_data.items():
                    if soc not in final_mapping:
                        final_mapping[soc] = []
                    final_mapping[soc].extend(terms)
            time.sleep(1)
        except Exception as e:
            print(f"   ⚠️ Batch Error: {e}")

    # 3. 儲存結果
    if final_mapping:
        with open('categorized_adverse_events.json', 'w', encoding='utf-8') as f:
            json.dump(final_mapping, f, indent=4)
        print(f"✨ SUCCESS: Dictionary saved with {len(final_mapping)} SOCs.")
    else:
        print("🚨 Error: No data returned from Gemini.")

if __name__ == "__main__":
    main()