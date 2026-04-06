# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:05:09 2024

@author: smunk
"""

import json

def print_json_structure(data, indent=0):
    """
    Recursively prints the structure of a JSON object.
    
    Args:
        data: The JSON data (could be a dict, list, etc.)
        indent: The current level of indentation (used for nested structures)
    """
    indent_str = "  " * indent  # Creates an indentation string
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent_str}{key}: {type(value).__name__}")
            print_json_structure(value, indent + 1)
    elif isinstance(data, list):
        print(f"{indent_str}List of {len(data)} items: {type(data[0]).__name__}" if data else "Empty list")
        if len(data) > 0:
            print_json_structure(data[0], indent + 1)
    else:
        print(f"{indent_str}{type(data).__name__}")

# Load the JSON file
with open('updated_clinical_trials.json', 'r') as f:
    clinical_trials_data = json.load(f)

# Print the structure of the JSON data
print_json_structure(clinical_trials_data)
