# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 003: Categorize Unique Adverse Events (Powered by google-genai)

This script extracts all unique Adverse Event terms from the trials and 
uses Gemini to map them to MedDRA System Organ Classes (SOCs).
"""

import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model_id = 'gemini-2.5-flash'

    # 1. Load the trials to find unique AE terms
    print("Extracting unique adverse events from special-trials.json...")
    with open('special-trials.json', 'r', encoding='utf-8') as f:
        trials = json.load(f)

    unique_events = set()
    for trial in trials:
        ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', {})
        # Extract from Serious and Other events
        for event_type in ['seriousEvents', 'otherEvents']:
            for event in ae_module.get(event_type, []):
                term = event.get('term')
                if term:
                    unique_events.add(term)

    event_list = sorted(list(unique_events))
    print(f"Found {len(event_list)} unique adverse event terms.")

    # 2. Ask Gemini to categorize them
    prompt = """
    Categorize the following list of adverse event terms into MedDRA System Organ Classes (SOCs). 
    Return the result as a flat JSON dictionary where the KEY is the System Organ Class 
    and the VALUE is a list of terms belonging to it.
    
    Example format:
    {"Gastrointestinal disorders": ["Nausea", "Vomiting"], "Nervous system disorders": ["Headache"]}
    
    Terms to categorize:
    """

    # Process in batches to avoid token limits
    batch_size = 50
    final_mapping = {}

    for i in range(0, len(event_list), batch_size):
        batch = event_list[i:i+batch_size]
        print(f"Categorizing batch {i//batch_size + 1}...")
        
        response = client.models.generate_content(
            model=model_id,
            contents=f"{prompt}\n{', '.join(batch)}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        try:
            batch_mapping = json.loads(response.text)
            for soc, terms in batch_mapping.items():
                if soc not in final_mapping:
                    final_mapping[soc] = []
                final_mapping[soc].extend(terms)
        except Exception as e:
            print(f"Error parsing batch: {e}")

    # 3. Save the "Dictionary" files required by Script 004
    # Script 004 sometimes looks for both filenames, so we save both to be safe
    with open('categorized_adverse_events.json', 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=4)
    
    with open('input-gpt-prompt-organsys.json', 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=4)

    print("✅ Successfully created 'categorized_adverse_events.json' and 'input-gpt-prompt-organsys.json'!")

if __name__ == "__main__":
    main()