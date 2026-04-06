# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:23:01 2024

@author: smunk
"""

import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Securely load your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
# Load the JSON data
with open('adverse_event_terms.json', 'r') as file:
    data = json.load(file)

# Define the prompt template
prompt_template = """
You are an expert in clinical trial data analysis and pharmacology. Given the following adverse event, please categorize it into the most appropriate category from the list below. Only provide the category number and name as the response. Do not include any additional explanations.

Categories:
1. General/Constitutional Symptoms (includes General Disorders and Administration Site Conditions)
2. Gastrointestinal
3. Hematologic
4. Dermatologic/Skin
5. Neurologic
6. Cardiovascular (includes Cardiac Disorders and Vascular Disorders)
7. Respiratory (includes Respiratory, Thoracic, and Mediastinal Disorders)
8. Endocrine
9. Hepatobiliary Disorders
10. Renal/Urinary
11. Musculoskeletal
12. Ophthalmic
13. Psychiatric
14. Metabolic/Nutritional
15. Blood and Lymphatic System Disorders
16. Immune System Disorders
17. Infections and Infestations
18. Skin and Subcutaneous Tissue Disorders
19. Other Disorders
"""

# Initialize an empty list to store the categorized results
categorized_results = []
total_tokens_used = 0

# Define the maximum number of events to process
max_events = len(data) # Change this value to limit the number of events processed

# Process each adverse event individually
for i, event in enumerate(data):
    if i >= max_events:
        break  # Exit the loop once the limit is reached

    full_prompt = f"{prompt_template}\n\nEvent: {event}\n\nPlease provide the most appropriate category for this event."

    try:
        # Make a request to the OpenAI GPT API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": "Category:"}
            ],
            timeout=1200  # Timeout in seconds, adjust as necessary
        )

        tokens_used = response.get('usage', {}).get('total_tokens', 0)
        total_tokens_used += tokens_used

        # Store the response text
        category = response['choices'][0]['message']['content'].strip()

        # Store the event-category pair
        categorized_event = {
            "event": event,
            "category": category
        }
        categorized_results.append(categorized_event)
        print(f"Processed event {i+1}/{max_events}: {event}")
        print(f"Tokens used: {tokens_used}\n")
        # Save the categorized results to a JSON file
        with open('categorized_adverse_events2.json', 'w') as file:
            json.dump(categorized_results, file, indent=4)
                
    except Exception as e:
        print(f"An error occurred while processing {event}: {str(e)}")

print("Categorized adverse events have been saved to JSON")
print(f"Total tokens used: {total_tokens_used}")
