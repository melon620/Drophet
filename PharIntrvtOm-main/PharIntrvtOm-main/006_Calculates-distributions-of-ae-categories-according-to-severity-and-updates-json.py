import json
from collections import defaultdict

# Load the updated clinical trials JSON file
with open('updated_clinical_trials.json', 'r') as f:
    clinical_trials_data = json.load(f)

# Initialize a structure to hold the results
trial_results = []

# Process each trial
for trial in clinical_trials_data:
    trial_id = trial['protocolSection']['identificationModule']['nctId']
    distribution_serious = defaultdict(lambda: defaultdict(float))
    distribution_other = defaultdict(lambda: defaultdict(float))
    
    if 'resultsSection' in trial and 'adverseEventsModule' in trial['resultsSection']:
        adverse_events_module = trial['resultsSection']['adverseEventsModule']
        
        # Process serious events
        if 'seriousEvents' in adverse_events_module:
            for event in adverse_events_module['seriousEvents']:
                category = event.get('category', "Uncategorized")
                for stat in event['stats']:
                    group_id = stat['groupId']
                    num_affected = stat.get('numAffected', 0)  # Use 0 if numAffected is missing
                    num_at_risk = stat.get('numAtRisk', 1)  # Use 1 if numAtRisk is missing to avoid division by zero
                    if num_at_risk > 0:
                        distribution_serious[group_id][category] += (num_affected / num_at_risk) * 100

        # Process other events
        if 'otherEvents' in adverse_events_module:
            for event in adverse_events_module['otherEvents']:
                category = event.get('category', "Uncategorized")
                for stat in event['stats']:
                    group_id = stat['groupId']
                    num_affected = stat.get('numAffected', 0)  # Use 0 if numAffected is missing
                    num_at_risk = stat.get('numAtRisk', 1)  # Use 1 if numAtRisk is missing to avoid division by zero
                    if num_at_risk > 0:
                        distribution_other[group_id][category] += (num_affected / num_at_risk) * 100

    # Add the distribution data to the trial data
    trial['adverseEventDistributions'] = {
        "seriousEvents": dict(distribution_serious),
        "otherEvents": dict(distribution_other)
    }
    trial_results.append(trial)

# Save the updated clinical trials data to a new JSON file
with open('trial_adverse_event_distributions_with_data_separated.json', 'w') as f:
    json.dump(trial_results, f, indent=4)

print("Adverse event distributions (separated for serious and other events) have been calculated and saved to 'trial_adverse_event_distributions_with_data_separated.json'")
