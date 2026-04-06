import json

# Load JSON data from files
with open('gpt_4o_mini_filtered_drug_treatment_trials_with_include-50trials.json') as f1, open('gpt_4o_mini_filtered_drug_treatment_trials_with_remove-behavioral-50trials.json') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Function to safely extract nctId
def extract_nct_ids(data):
    nct_ids = set()
    for item in data:
        try:
            nct_id = item['protocolSection']['identificationModule']['nctId']
            nct_ids.add(nct_id)
        except KeyError:
            # If 'protocolSection' or any key is missing, just skip this item
            continue
    return nct_ids

# Extract nctId values from both JSONs
nct_ids_1 = extract_nct_ids(data1)
nct_ids_2 = extract_nct_ids(data2)

# Find common and unique nctIds
common_nct_ids = nct_ids_1.intersection(nct_ids_2)
unique_to_file1 = nct_ids_1 - nct_ids_2
unique_to_file2 = nct_ids_2 - nct_ids_1

# Display the results
print("Common nctIds:", common_nct_ids)
print("Unique to file1:", unique_to_file1)
print("Unique to file2:", unique_to_file2)

# Display the results
print(f"Total nctIds in file1: {len(nct_ids_1)}")
print(f"Total nctIds in file2: {len(nct_ids_2)}")
print(f"Common nctIds: {len(common_nct_ids)}")
print(f"Unique to file1: {len(unique_to_file1)}")
print(f"Unique to file2: {len(unique_to_file2)}")

print(f'wanted: {nct_ids_2}')