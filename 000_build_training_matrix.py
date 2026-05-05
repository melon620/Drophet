# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Project Drophet)
Phase 0: Training Matrix Bootstrapper

Hand-curated bootstrap with broad DDI mechanism coverage (CYP3A4 inhibition,
serotonergic, hematologic, electrolyte, PDE5/nitrate, antifolate). Targets
are literature-anchored approximations, not prospective trial data.
"""

import pandas as pd
import itertools
import numpy as np

np.random.seed(42)

known_ddis = [
    # Bleeding
    {"d1": "Warfarin", "s1": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
     "d2": "Aspirin",  "s2": "CC(=O)OC1=CC=CC=C1C(=O)O",
     "Target_Hematologic": 38.45, "Target_Cardiovascular": 5.2, "Target_Hepatobiliary": 1.1},
    {"d1": "Warfarin", "s1": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
     "d2": "Ibuprofen","s2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
     "Target_Hematologic": 32.1, "Target_Renal": 6.4, "Target_Gastrointestinal": 8.2},
    {"d1": "Warfarin", "s1": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
     "d2": "Phenytoin","s2": "O=C1NC(=O)C(N1)(C2=CC=CC=C2)C3=CC=CC=C3",
     "Target_Hematologic": 25.0, "Target_Nervous_System": 4.0},
    {"d1": "Aspirin",  "s1": "CC(=O)OC1=CC=CC=C1C(=O)O",
     "d2": "Heparin",  "s2": "OS(=O)(=O)OC1C(O)C(O)C(NS(=O)(=O)O)C(C(=O)O)O1",
     "Target_Hematologic": 28.5, "Target_Gastrointestinal": 3.0},

    # CYP3A4 / statin myopathy
    {"d1": "Simvastatin",  "s1": "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
     "d2": "Ketoconazole", "s2": "CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OCC3COC(O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl",
     "Target_Hepatobiliary": 32.4, "Target_Musculoskeletal": 28.5},
    {"d1": "Simvastatin",   "s1": "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
     "d2": "Clarithromycin","s2": "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
     "Target_Hepatobiliary": 24.8, "Target_Musculoskeletal": 30.2},
    {"d1": "Atorvastatin", "s1": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
     "d2": "Diltiazem",    "s2": "CC(=O)OC1C(SC2=CC=CC=C2N(C1=O)CCN(C)C)C3=CC=C(C=C3)OC",
     "Target_Hepatobiliary": 11.5, "Target_Musculoskeletal": 10.0},

    # PDE5 + nitrate
    {"d1": "Sildenafil",    "s1": "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
     "d2": "Nitroglycerin", "s2": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
     "Target_Cardiovascular": 47.0, "Target_Nervous_System": 9.0},

    # Serotonin syndrome
    {"d1": "Fluoxetine", "s1": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
     "d2": "Tramadol",   "s2": "CN(C)CC1(CCCCC1=O)C2=CC(=CC=C2)OC",
     "Target_Nervous_System": 22.5, "Target_Cardiovascular": 4.0},
    {"d1": "Fluoxetine", "s1": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
     "d2": "Selegiline", "s2": "CC(CC1=CC=CC=C1)N(C)CC#C",
     "Target_Nervous_System": 35.0, "Target_Cardiovascular": 8.0},

    # Lithium toxicity
    {"d1": "Lithium",   "s1": "[Li+]",
     "d2": "Ibuprofen", "s2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
     "Target_Nervous_System": 21.0, "Target_Renal": 14.0, "Target_Gastrointestinal": 3.5},

    # Antifolate
    {"d1": "Methotrexate","s1": "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
     "d2": "Trimethoprim","s2": "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
     "Target_Hematologic": 27.0, "Target_Hepatobiliary": 6.0, "Target_Gastrointestinal": 8.0},

    # Hyperkalemia
    {"d1": "Lisinopril",     "s1": "C1CC(N(C1)C(=O)C(CCCCN)NC(CCC2=CC=CC=C2)C(=O)O)C(=O)O",
     "d2": "Spironolactone", "s2": "CC12CCC(=O)C=C1CCC3C2CCC4(C3CCC4(C(=O)C)SC(=O)C)C",
     "Target_Renal": 23.0, "Target_Cardiovascular": 9.0},

    # ACE-I + NSAID
    {"d1": "Lisinopril","s1": "C1CC(N(C1)C(=O)C(CCCCN)NC(CCC2=CC=CC=C2)C(=O)O)C(=O)O",
     "d2": "Ibuprofen", "s2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
     "Target_Renal": 14.2, "Target_Cardiovascular": 12.0, "Target_Gastrointestinal": 5.5},

    # CYP1A2
    {"d1": "Ciprofloxacin","s1": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
     "d2": "Theophylline", "s2": "CN1C2=C(C(=O)N(C1=O)C)NC=N2",
     "Target_Nervous_System": 19.5, "Target_Cardiovascular": 11.0},

    # Digoxin
    {"d1": "Digoxin",    "s1": "CC1OC(CC(C1O)O)OC2C(OC(CC2O)OC3C(OC(CC3O)OC4CCC5(C(C4)CCC6C5CCC7(C6(CCC7C8=CC(=O)OC8)O)C)C)C)C",
     "d2": "Amiodarone", "s2": "CCCCC1=C(C(=O)C2=CC(=C(C(=C2O1)I)OCCN(CC)CC)I)CCCC",
     "Target_Cardiovascular": 25.0, "Target_Nervous_System": 6.0},

    # Moderate
    {"d1": "Aspirin",       "s1": "CC(=O)OC1=CC=CC=C1C(=O)O",
     "d2": "Acetaminophen", "s2": "CC(=O)NC1=CC=C(C=C1)O",
     "Target_Renal": 7.5, "Target_Gastrointestinal": 6.0, "Target_Hepatobiliary": 4.5},
    {"d1": "Metformin",  "s1": "CN(C)C(=N)N=C(N)N",
     "d2": "Furosemide", "s2": "C1=CC(=C(C=C1Cl)S(=O)(=O)N)NCC2=CC=CO2",
     "Target_Renal": 9.0, "Target_Hepatobiliary": 5.5},

    # CNS depression (preserved)
    {"d1": "Diazepam",        "s1": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
     "d2": "Diphenhydramine", "s2": "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2",
     "Target_Nervous_System": 15.8, "Target_Respiratory": 8.5, "Target_Cardiovascular": 2.1},
]

# Safe pool: extended for broader negative-control coverage. C(12,2)=66 pairs.
safe_pool = [
    {"d": "Paracetamol",   "s": "CC(=O)NC1=CC=C(C=C1)O"},
    {"d": "Vitamin C",     "s": "C(C(C1C(=C(C(=O)O1)O)O)O)O"},
    {"d": "Amoxicillin",   "s": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"},
    {"d": "Cholecalciferol","s": "CC(C)CCCC(C)C1CCC2C1(CCCC2=CC=C3CC(CCC3=C)O)C"},
    {"d": "Omeprazole",    "s": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC"},
    {"d": "Metformin",     "s": "CN(C)C(=N)N=C(N)N"},
    {"d": "Atorvastatin",  "s": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"},
    {"d": "Lisinopril",    "s": "C1CC(N(C1)C(=O)C(CCCCN)NC(CCC2=CC=CC=C2)C(=O)O)C(=O)O"},
    {"d": "Loratadine",    "s": "CCOC(=O)N1CCC(=C2C3=C(CCC4=CC(=CN=C24)Cl)C=CC=C3)CC1"},
    {"d": "Levothyroxine", "s": "C1=CC(=C(C=C1CC(C(=O)O)N)I)OC2=CC(=CC=C2)I"},
    {"d": "Aspirin",       "s": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"d": "Ibuprofen",     "s": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
]

TARGET_KEYS = [
    "Target_Hematologic", "Target_Cardiovascular", "Target_Hepatobiliary",
    "Target_Nervous_System", "Target_Respiratory", "Target_Musculoskeletal",
    "Target_Renal", "Target_Gastrointestinal", "Target_Dermatologic",
]


def fill(item):
    return {k: item.get(k, 0.0) for k in TARGET_KEYS}


records = []
for item in known_ddis:
    r = {"Drug_1": item["d1"], "SMILES_1": item["s1"],
         "Drug_2": item["d2"], "SMILES_2": item["s2"]}
    r.update(fill(item))
    records.append(r)

for d1, d2 in itertools.combinations(safe_pool, 2):
    noise = lambda: round(np.random.uniform(0.1, 2.5), 2)
    r = {"Drug_1": d1["d"], "SMILES_1": d1["s"],
         "Drug_2": d2["d"], "SMILES_2": d2["s"]}
    r.update({k: noise() for k in TARGET_KEYS})
    records.append(r)

for drug in safe_pool:
    noise = lambda: round(np.random.uniform(0.5, 4.0), 2)
    r = {"Drug_1": drug["d"], "SMILES_1": drug["s"],
         "Drug_2": "", "SMILES_2": ""}
    r.update({k: noise() for k in TARGET_KEYS})
    records.append(r)

df = pd.DataFrame(records)
df = pd.concat([df] * 5, ignore_index=True)
for col in df.columns:
    if col.startswith("Target_"):
        df[col] = np.clip(df[col] + np.random.normal(0, 0.5, len(df)), 0.0, 100.0)

df.to_csv("training_matrix_cleaned.csv", index=False)
print(f"✅ Successfully generated 'training_matrix_cleaned.csv' with {len(df)} records.")
print(f"   {len(known_ddis)} DDIs, C({len(safe_pool)},2)={len(safe_pool)*(len(safe_pool)-1)//2} safe combos, {len(safe_pool)} monos × 5 dupes.")
