''' only drug treatment trila should be included (not medicial drug trials) and
and make it easier searchable in pubchem , (beautify the names)

-next gpt prompt
- identify whihc is a small molecule ()
'''
import openai
import json
import re


# Set your OpenAI API key
openai.api_key = ''  # Replace with your actual API key or load from environment variable

# Load the JSON data
with open('/Users/marie/dev/PharIntrvtOm/ctg-studies_with_eventGroups.json', 'r') as file:
    data = json.load(file)

# Define the prompt for GPT
prompt_template = '''Given a list of clinical trial data in JSON format, where each entry contains information about that clinical trial perform the following task:
Remove any trials that are solely focusing on medical devices, behavioral interventions or dietary supplements.
Please provide the processed results in JSON format.
'''


# Initialize an empty list to store the filtered results
filtered_trials = []
total_tokens_used = 0


# Process in batches
n = 5 # Batches

for i in range(0, 25,n):  
    batch = data[i:i+n]
    trials_json_str = json.dumps(batch, indent=4)
    full_prompt = prompt_template + "\n" + trials_json_str

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in clinical trial data analysis and pharmacology."},
                {"role": "user", "content": full_prompt}
            ],
            timeout=1200  # Set timeout for the API request
        )

        response_text = response['choices'][0]['message']['content']
        #print("Full GPT Response:\n", response_text)

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
        # Save the collected filtered results to a new JSON file
        with open('gpt_4o_mini_filtered_drug_treatment_full-45k.json', 'a') as file:
            json.dump(filtered_trials, file, indent=4)

        tokens_used = response.get('usage', {}).get('total_tokens', 0)
        total_tokens_used += tokens_used

    except Exception as e:
        print(f"An error occurred: {e}")

    # Log progress
    if i % 10 == 0:
        print(f"Processed {i} trials")


print("Filtered trials have been saved to JSON")
print(f"Total tokens used: {total_tokens_used}")