# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 17:37:05 2024

@author: smunk
"""
import json

# Load the dictionary JSON file
with open('categorized_adverse_events.json', 'r') as f:
    dictionary_data = json.load(f)

# Load the clinical trials JSON file
with open('input-gpt-prompt-organsys.json', 'r') as f:
    clinical_trials_data = json.load(f)

# Create a mapping of events to categories
event_to_category = {entry["event"].strip('"'): entry["category"] for entry in dictionary_data}

# Replace the adverse event terms in the clinical trials data
for trial in clinical_trials_data:
    if 'resultsSection' in trial and 'adverseEventsModule' in trial['resultsSection']:
        adverse_events_module = trial['resultsSection']['adverseEventsModule']
        
        # Replace terms in seriousEvents
        if 'seriousEvents' in adverse_events_module:
            for event in adverse_events_module['seriousEvents']:
                if 'term' in event:
                    term = event['term']
                    if term in event_to_category:
                        event['category'] = event_to_category[term]
                    else:
                        event['category'] = "Uncategorized"
                        
        # Replace terms in otherEvents
        if 'otherEvents' in adverse_events_module:
            for event in adverse_events_module['otherEvents']:
                if 'term' in event:
                    term = event['term']
                    if term in event_to_category:
                        event['category'] = event_to_category[term]
                    else:
                        event['category'] = "Uncategorized"

# Save the updated clinical trials data to a new JSON file
with open('updated_clinical_trials.json', 'w') as f:
    json.dump(clinical_trials_data, f, indent=4)

print("Clinical trials data has been updated and saved to 'updated_clinical_trials.json'")