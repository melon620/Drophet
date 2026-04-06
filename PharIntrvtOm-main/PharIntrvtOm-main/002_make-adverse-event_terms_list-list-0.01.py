# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:41:46 2024

@author: smunk
"""
import json

def extract_adverse_event_terms(json_file_path, output_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize a set to store unique adverse event terms
    adverse_event_terms = set()

    # Iterate over each item in the JSON list
    for item in data:
        results_section = item.get("resultsSection", {})
        adverse_events_module = results_section.get("adverseEventsModule", {})

        # Extract terms from seriousEvents
        serious_events = adverse_events_module.get("seriousEvents", [])
        for event in serious_events:
            term = event.get("term")
            if term:
                adverse_event_terms.add(term)

        # Extract terms from otherEvents
        other_events = adverse_events_module.get("otherEvents", [])
        for event in other_events:
            term = event.get("term")
            if term:
                adverse_event_terms.add(term)

    # Convert the set to a sorted list
    sorted_terms = sorted(adverse_event_terms)

    # Save the sorted terms to a new JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(sorted_terms, output_file, indent=4)

    print(f"Extracted {len(sorted_terms)} adverse event terms and saved to {output_file_path}.")

# Example usage
extract_adverse_event_terms('input-gpt-prompt-organsys.json', 'adverse_event_terms.json')

