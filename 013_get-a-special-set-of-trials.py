import json

# Load the original JSON file
with open('ctg-studies-groupTitle-and-description.json', 'r') as file:
    data = json.load(file)

# Extract the first 100 trials
first_100_trials = data[:100]

# List of specific nctIds to extract
specific_nctIds = {"NCT04143594", "NCT02612194", "NCT06076694", "NCT00313560"}

# Extract trials with specific nctIds
specific_trials = [trial for trial in data if trial['protocolSection']['identificationModule']['nctId'] in specific_nctIds]

# Combine the two lists, ensuring no duplicates
unique_trials = {trial['protocolSection']['identificationModule']['nctId']: trial for trial in (first_100_trials + specific_trials)}
final_trials = list(unique_trials.values())

# Save the extracted trials to a new JSON file
with open('special-trials.json', 'w') as file:
    json.dump(final_trials, file, indent=4)

print(f"Extracted {len(final_trials)} trials.")
