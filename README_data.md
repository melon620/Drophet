# PharmacoInterventome

The Pharmacointerventome dataset is a curated collection of pharmacological interventions derived from clinical trials. This dataset focuses on drug testing and aims to provide researchers with essential information regarding small molecule interventions, their conditions, and associated side effects.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Creation](#dataset-creation)
- [Data Description](#data-description)

## Introduction
This dataset specifically targets trials involving small molecules, providing a resource for researchers interested in pharmacological interventions and their effects.

## Dataset Creation
The Pharmacointerventome dataset was created through a series of systematic steps:

1. **Data Collection**:
   - The initial raw data was obtained from [ClinicalTrials.gov](https://clinicaltrials.gov/), focusing on all trials in phases 1-4 that reported results. This collection yielded approximately **45,000 trials**.

2. **Initial Filtering**:
   - Trials were filtered to exclude those focused solely on:
     - Medical devices
     - Behavioral interventions
     - Dietary supplements
   - After this filtering, approximately **37,000 trials** remained.

3. **Small molecules as primary investigational drug**:
   - Trials were identified where small molecules were the primary investigational drugs, resulting in around **18,000 trials**. And around **19,000** trials that countained drugs in any of the treatment arms.
     
   - **Distribution of Number of Drugs in `primaryInvestigationalDrugs`:**
     

    | **Number of Drugs** | **Number of Trials** | **Number of Drugs** | **Number of Trials** |
    |---------------------|----------------------|---------------------|----------------------|
    | 1 drug(s)           | 10,726 trial(s)      | 9 drug(s)           | 16 trial(s)          |
    | 2 drug(s)           | 5,457 trial(s)       | 10 drug(s)          | 7 trial(s)           |
    | 3 drug(s)           | 1,701 trial(s)       | 11 drug(s)          | 6 trial(s)           |
    | 4 drug(s)           | 552 trial(s)         | 12 drug(s)          | 1 trial(s)           |
    | 5 drug(s)           | 218 trial(s)         | 13 drug(s)          | 5 trial(s)           |
    | 6 drug(s)           | 83 trial(s)          | 14 drug(s)          | 1 trial(s)           |
    | 7 drug(s)           | 33 trial(s)          | 18 drug(s)          | 1 trial(s)           |
    | 8 drug(s)           | 33 trial(s)          | 19 drug(s)          | 1 trial(s)           |

4. **Statistical Information Check**:
   - Trials lacking statistical information about their treatment arms or for which no drug could be identified across all treatment arms were removed. This left the dataset with approximately **33,000 studies**.

5. **Final Dataset Enrichment**:
   - For the final dataset, ICD codes were generated for the main condition of each trial.
   - Canonical SMILES representations and PubChem links were included for all small molecules mentioned in the trials.

## Data Description
- **Total Entries**: Approximately 33,000 trials
- **Key Features**:
  - NctId
  - Primary Investigational Drug
  - Main Condition (with ICD codes)
  - Treatment Arms (with statistical information)
  - Canonical SMILES
  - PubChem Links
