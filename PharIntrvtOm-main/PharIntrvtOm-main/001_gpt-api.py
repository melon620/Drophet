import openai
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_requestor.APIRequestor.request_timeout = 1200  # Set timeout to 1200 seconds (20 minutes)

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


# Load the JSON data
with open('/Users/marie/dev/clinical_trials/ctg-filtered-drug-trials.json', 'r') as file:
    data = json.load(file)

# Define the prompt for GPT
prompt_template = '''I have a JSON file of clinical trials with fields such as "NCT Number," "Study Title," "Summary," and more. I need to filter this dataset to identify trials that:

Investigate small molecules as the primary investigational drugs. Include trials where at least one of the primary investigational drugs is a small molecule. Exclude trials where all primary investigational drugs are proteins, antibodies, or larger biologic molecules.
Focus on systemic applications, such as oral or intravenous administration. Exclude trials that focus on dermal or topical applications.
Please generate a new JSON file with the filtered trials including only the following keys for each trial:

Please provide the filtered results in the following JSON format, enclosed in triple backticks:

```json
[
    {
        "nctId": "<trial ID>",
        "primaryInvestigationalDrugs": ["<drug name 1>", "<drug name 2>", ...]
    },
    ...
]
'''
# Initialize an empty list to store the filtered results
filtered_trials = []
total_tokens_used = 0

# Process in batches
for i in range(0, 44445, 10):
    # Extract a batch of 10 trials
    batch = data[i:i+10]

    # Check if the batch is empty (end of data)
    if not batch:
        break

    # Convert the batch to a JSON string and append to the prompt
    trials_json_str = json.dumps(batch, indent=4)
    full_prompt = prompt_template + trials_json_str

    # Make a request to the OpenAI GPT API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in clinical trial data analysis and pharmacology."},
            {"role": "user", "content": full_prompt}
        ]
    )

    tokens_used = response.get('usage', {}).get('total_tokens', 0)
    total_tokens_used += tokens_used

    # Attempt to extract and parse the response text
    try:
        response_text = response['choices'][0]['message']['content']
        json_string = re.search(r'```json\n([\s\S]*?)\n```', response_text, re.DOTALL)
        if json_string:
            batch_filtered_trials = json.loads(json_string.group(1))
            filtered_trials.extend(batch_filtered_trials)
        else:
            print(f"No JSON data found in response for batch starting at index {i}.")
    except KeyError:
        print("Error in accessing the response data. Check the response keys.")
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        print("Response received")

    # Log progress
    if i % 100 == 0:
        print(f"Processed {i} trials")
        


# Save the collected filtered results to a new JSON file
with open('gpt_filtered_ctg_trials_full.json', 'w') as file:
    json.dump(filtered_trials, file, indent=4)

print("Filtered trials have been saved to JSON")
print(f"Total tokens used: {total_tokens_used}")