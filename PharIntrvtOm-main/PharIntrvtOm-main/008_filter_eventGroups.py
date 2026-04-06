''' only drug treatment trila should be included (not medicial drug trials) and
and make it easier searchable in pubchem , (beautify the names)

-next gpt prompt
- identify whihc is a small molecule ()
'''
import openai
import json
import re


# Set your OpenAI API key
openai.api_key = ''

# Load the JSON data
with open('/Users/marie/dev/PharIntrvtOm/ctg-studies_with_eventGroups.json', 'r') as file:
    data = json.load(file)

# Define the prompt for GPT
prompt_template = '''Given a list of clinical trial data in JSON format, where each entry contains information about protocol and results including adverse events and their respective drug treatments, perform the following tasks:
1. Filter and include only trials that involve drug treatments.
2. Extract the full and concrete names of the drugs listed in the event groups' titles for each trial. Standardize drug names:
   - Replace abbreviations and short forms with full drug names (e.g., "LEN" to "Lenacapavir").
   - List each drug separately if multiple drugs are combined (e.g., "LEN + F/TAF + BIC" should be separated into "Lenacapavir, Emtricitabine, Tenofovir Alafenamide, Bictegravir").
3. Extract any additional information such as drug doses and frequencies of administration.
   - Standardize frequencies into a consistent format (e.g., "twice daily" to "BID", "every other week" to "EOW").
   - Clearly separate dose from frequency (e.g., "40 mg EOW/EW" should be broken into "dose": "40 mg" and "frequency": "EOW, EW").
4. Output the processed data in JSON format with the following structure:
   - Each entry should have nctId, eventGroups (with id, drugName, dose if applicable, and frequency if applicable).
   - Ensure that placebos and controls are included and clearly labeled in the output.
Please provide the processed results in JSON format.
'''


# Initialize an empty list to store the filtered results
filtered_trials = []
total_tokens_used = 0


# Process in batches
for i in range(0, 20,5):  # Assuming you want to process the data in batches of 10
    batch = data[i:i+10]
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

        tokens_used = response.get('usage', {}).get('total_tokens', 0)
        total_tokens_used += tokens_used

    except Exception as e:
        print(f"An error occurred: {e}")

    # Log progress
    if i % 10 == 0:
        print(f"Processed {i} trials")

# Save the collected filtered results to a new JSON file
with open('gpt_4o_mini_filtered_eventGroups.json', 'w') as file:
    json.dump(filtered_trials, file, indent=4)

print("Filtered trials have been saved to JSON")
print(f"Total tokens used: {total_tokens_used}")