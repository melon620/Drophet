''' Gets the ICD name for each condition mentioned and lisst in the json

To do:
    extract traial swere ther is insufficient information in eventgroups
'''
import openai
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load existing data or initialize a new list
try:
    with open('ctg-studies-only-conditions.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    data = []

prompt_template = '''Please provide the ICD classification code for the following condition, ensuring the code is as accurate and relevant as possible. Only provide the ICD as a reposnce and do not include any extra explanations.'''

total_tokens_used = 0
max_conditions = len(data)  # You can adjust this if needed

# Initialize a counter for conditions processed
condition_counter = 0

categorized_results = []

# Go through each trial
for trial in data[:max_conditions]:  # Limit to max_conditions if necessary
    conditions = trial["protocolSection"]["conditionsModule"]["conditions"]
    for condition in conditions:
        full_prompt = f"{prompt_template}\n\nCondition: {condition}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Adjust the model as needed
            messages=[
                {"role": "system", "content": "You are an expert in epidemiology and disease clasification."},
                {"role": "user", "content": full_prompt}
            ],
            timeout=1200  # Timeout in seconds, adjust as necessary
        )
            
            tokens_used = response.get('usage', {}).get('total_tokens', 0)
            total_tokens_used += tokens_used
            icd_code = response['choices'][0]['message']['content'].strip()

            categorized_event = {
                'condition': condition,
                'ICD': icd_code
            }

            categorized_results.append(categorized_event)
            # Save all data back to the file
            with open('gpt_added_ICD.json', 'w') as file:
                json.dump(categorized_results, file, indent=4)

        except Exception as e:
            print(f"An error occurred while processing condition '{condition}': {str(e)}")

            
        # Increment the condition counter and check if it's time to print a status message
        condition_counter += 1
        if condition_counter % 20 == 0:
            print(f"{condition_counter} conditions processed so far.")


print("ICD codes have been added and saved to JSON")
print(f"Total tokens used: {total_tokens_used}")