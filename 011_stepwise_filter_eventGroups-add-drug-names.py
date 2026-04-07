# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 011: Stepwise Filter EventGroups & Add Drug Names (Powered by NEW google-genai)

This script uses Gemini to extract drug names from clinical trial event groups,
expand abbreviations, and handle combination therapies.
"""

import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

def main():
    # 1. Load API Key from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")

    # 2. Initialize Gemini Client
    client = genai.Client(api_key=api_key)
    model_id = 'gemini-2.5-flash'

    # Load input data from Script 013
    input_file = 'special-trials.json'
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run Script 013 first.")
        return

    # Define the Prompt for drug extraction
    prompt_template = """
    For each eventGroup in resultsSection.adverseEventsModule of the input JSON, 
    extract only drug names from the title or description. 
    1. Add a 'drugs' key with these names as an array.
    2. Expand abbreviations (e.g., 'TDF' to 'Tenofovir Disoproxil Fumarate').
    3. List each component of a combination drug separately.
    4. For Placebo/Control groups, write 'placebo' or 'control' in the array.
    Return ONLY the updated JSON array.
    """

    filtered_trials = []
    batch_size = 5 # Number of trials per API call
    
    print(f"Starting Gemini extraction for {len(data)} trials...")

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_json_str = json.dumps(batch, indent=2)
        
        full_content = f"{prompt_template}\n\nInput JSON:\n{batch_json_str}"

        try:
            # Call Gemini API
            response = client.models.generate_content(
                model=model_id,
                contents=full_content,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert in pharmacology and clinical trial documentation.",
                    temperature=0.1,
                    response_mime_type="application/json" # Ensure JSON output
                )
            )
            
            # Parse the response text as JSON
            batch_result = json.loads(response.text)
            filtered_trials.extend(batch_result)
            
            # Save progress after each batch
            with open('gpt_filteres-special-trials-w-or-final.json', 'w', encoding='utf-8') as f:
                json.dump(filtered_trials, f, indent=4)
            
            print(f"Processed {min(i + batch_size, len(data))}/{len(data)} trials...")
            
            # Rate limit protection for Free Tier
            time.sleep(2)

        except Exception as e:
            print(f"Error at batch starting index {i}: {e}")

    print("✅ Successfully completed drug name extraction!")

if __name__ == "__main__":
    main()