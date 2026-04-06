# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 012: Map Conditions to ICD Codes (Powered by the NEW google-genai SDK)
"""

import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

def main():
    # 1. Load API Key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")

    # 2. Initialize the NEW Client
    # 新版 SDK 會自動食 os.environ 入面嘅 GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    
    # 定義模型版本 (建議用 2.5-flash，速度更快更準)
    model_id = 'gemini-2.5-flash'

    prompt_template = '''Please provide the most accurate ICD-10 classification code for the following condition. Only provide the ICD code string as a response (e.g., "C50.9"), and do not include any extra words, symbols, or explanations.'''

    # 3. Load input data
    input_file = 'special-trials.json'
    print(f"Loading trial data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run script 013 first.")
        return

    categorized_results = []
    condition_counter = 0
    icd_cache = {}

    print(f"Starting Gemini processing for ICD codes using {model_id}...")

    # 4. Process conditions
    for trial in data:
        protocol_section = trial.get("protocolSection", {})
        conditions_module = protocol_section.get("conditionsModule", {})
        conditions = conditions_module.get("conditions", [])
        
        for condition in conditions:
            if condition in icd_cache:
                icd_code = icd_cache[condition]
            else:
                full_prompt = f"{prompt_template}\n\nCondition: {condition}"
                
                try:
                    # NEW SYNTAX for generating content with System Instructions
                    response = client.models.generate_content(
                        model=model_id,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are an expert in epidemiology and disease classification.",
                            temperature=0.0 # Deterministic output
                        )
                    )
                    icd_code = response.text.strip()
                    
                    icd_cache[condition] = icd_code
                    time.sleep(1) # Rate limit protection
                    
                except Exception as e:
                    print(f"API Error processing '{condition}': {e}")
                    icd_code = "Unknown"

            categorized_event = {
                'condition': condition,
                'ICD': icd_code
            }
            categorized_results.append(categorized_event)
            condition_counter += 1

            if condition_counter % 20 == 0:
                print(f"{condition_counter} conditions processed...")
                with open('gpt_added_ICD.json', 'w', encoding='utf-8') as file:
                    json.dump(categorized_results, file, indent=4)

    # 5. Final save
    with open('gpt_added_ICD.json', 'w', encoding='utf-8') as file:
        json.dump(categorized_results, file, indent=4)

    print(f"✅ Finished! Mapped {condition_counter} conditions to ICD codes using new SDK.")

if __name__ == "__main__":
    main()