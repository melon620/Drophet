''' only drug treatment trila should be included (not medicial drug trials) and
and make it easier searchable in pubchem , (beautify the names)

-next gpt prompt
- identify whihc is a small molecule ()


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

# Load the JSON data
with open('special-trials.json', 'r') as file:
    data = json.load(file)

# Define the prompt for GPT

#add the descitption part t teh input and include placebo and the control groups
prompt_template = '''For each eventGroup in resultsSection.adverseEventsModule of the input JSON, extract only drug names from title or description, add a drugs key with these names as an array, and expand any abbreviations. Please also list every drug in a combination drug separately. For Placebo and Control groups without further specification write placebo or control accordingly in the drugs key. Return the updated JSON. And here is the input json:
'''


# Initialize an empty list to store the filtered results
filtered_trials = []
total_tokens_used = 0


# Process in batches
n = 5 # Batches


# Process in batches
for i in range(0, len(data), n):  
    batch = data[i:i+n]
    trials_json_str = json.dumps(batch, indent=4)
    full_prompt = prompt_template + "\n" + trials_json_str

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in clinical trial data analysis and pharmacology."},
                {"role": "user", "content": full_prompt}
            ],
            timeout=1200  # Set timeout for the API request
        )

        response_text = response['choices'][0]['message']['content']

        # Correct the extraction by focusing on JSON portion accurately
        try:
            json_data_start = response_text.index('[')
            json_data_end = response_text.rindex(']') + 1
            json_string = response_text[json_data_start:json_data_end]
            batch_filtered_trials = json.loads(json_string)
            filtered_trials.extend(batch_filtered_trials)
        except ValueError as e:
            print(f"JSON extraction error: {e}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            print("Response text that caused the error:\n", response_text)

        tokens_used = response.get('usage', {}).get('total_tokens', 0)
        total_tokens_used += tokens_used

    except Exception as e:
        print(f"An error occurred: {e}")

    # Save the filtered results after each batch by overwriting the same file
    with open('gpt_filteres-special-trials-w-or-final.json', 'w') as file:
        json.dump(filtered_trials, file, indent=4)

    # Log progress
    if i % 10 == 0:
        print(f"Processed {i} trials")


print("Filtered trials have been saved to JSON")
print(f"Total tokens used: {total_tokens_used}")