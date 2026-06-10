# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Project Drophet)
Phase 7: Real Training Data — FDA FAERS + OnSIDES Reference

Replaces the hardcoded training matrix with data derived from:
  - FDA FAERS (OpenFDA API): real co-reporting rates for drug pairs
    and single-drug adverse event profiles from ~18M+ adverse event reports
  - OnSIDES schema reference: MedDRA-structured drug-AE vocabulary conventions

Pipeline:
  1. For each drug in SEED_DRUGS, query FAERS for total reports + AE counts by category
  2. For every drug pair where both have ≥ MIN_SINGLE_REPORTS:
       a. Query FAERS for co-reports + DDI AE counts
       b. If co-reports ≥ MIN_PAIR_REPORTS → use real DDI rates
       c. Else → fall back to independence model (like script 023)
  3. Fetch SMILES from PubChem; discard macromolecules (MW > 1200 Da)
  4. Output training_matrix_real.csv in the existing schema

FAERS rate semantics:
  category_rate = (co-reports mentioning AEs in that organ system)
                / (total co-reports for the pair) × 100
  This is a co-reporting fraction, not a clinical trial incidence rate.
  Relative ordering (high-risk pair > low-risk pair) is preserved across all pairs.

Caching: all API responses are saved to faers_cache/ so re-runs are fast.

Requirements: internet connection, PubChem + OpenFDA APIs (no API key required but
  set OPENFDA_API_KEY env var to raise the rate limit from 40/min to 240/min).
"""

import os
import json
import time
import urllib.parse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

from drophet_utils import seed_everything, pair_key

warnings.filterwarnings('ignore')
seed_everything(42)

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR          = 'faers_cache'
OUTPUT_FILE        = 'training_matrix_real.csv'
MIN_SINGLE_REPORTS = 500     # skip a drug if FAERS has fewer total reports
MIN_PAIR_REPORTS   = 50      # use real DDI rate only if this many co-reports exist
N_AE_TOP           = 200     # top N MedDRA PT terms to pull per query
API_KEY            = os.environ.get('OPENFDA_API_KEY', '')
REQUEST_DELAY      = 0.35    # seconds between API calls (stay under rate limit)

TARGET_COLS = [
    'Target_Hematologic', 'Target_Cardiovascular', 'Target_Hepatobiliary',
    'Target_Nervous_System', 'Target_Respiratory', 'Target_Musculoskeletal',
    'Target_Renal', 'Target_Gastrointestinal', 'Target_Dermatologic',
]

# ── Seed drug list ─────────────────────────────────────────────────────────────
# Common small-molecule drugs with broad FAERS coverage across organ systems.
# Names must match FAERS medicinalproduct field (UPPERCASE works best).
SEED_DRUGS = [
    # Anticoagulants / antiplatelets
    'WARFARIN', 'ASPIRIN', 'CLOPIDOGREL', 'HEPARIN', 'ENOXAPARIN', 'APIXABAN', 'RIVAROXABAN',
    # Statins / cardiovascular
    'ATORVASTATIN', 'SIMVASTATIN', 'ROSUVASTATIN', 'LOVASTATIN',
    'METOPROLOL', 'ATENOLOL', 'BISOPROLOL', 'CARVEDILOL',
    'LISINOPRIL', 'ENALAPRIL', 'RAMIPRIL',
    'AMLODIPINE', 'NIFEDIPINE', 'VERAPAMIL', 'DILTIAZEM',
    'DIGOXIN', 'AMIODARONE',
    'FUROSEMIDE', 'SPIRONOLACTONE', 'HYDROCHLOROTHIAZIDE',
    'LOSARTAN', 'IRBESARTAN',
    # NSAIDs / pain
    'IBUPROFEN', 'NAPROXEN', 'DICLOFENAC', 'CELECOXIB', 'INDOMETHACIN',
    'ACETAMINOPHEN', 'TRAMADOL', 'CODEINE', 'MORPHINE', 'OXYCODONE',
    'GABAPENTIN', 'PREGABALIN',
    # CNS / psychiatry
    'DIAZEPAM', 'ALPRAZOLAM', 'LORAZEPAM', 'ZOLPIDEM',
    'QUETIAPINE', 'OLANZAPINE', 'RISPERIDONE', 'HALOPERIDOL',
    'FLUOXETINE', 'SERTRALINE', 'ESCITALOPRAM', 'PAROXETINE',
    'AMITRIPTYLINE', 'LITHIUM',
    'VALPROATE', 'CARBAMAZEPINE', 'PHENYTOIN', 'LAMOTRIGINE',
    # Antimicrobials / antifungals
    'AMOXICILLIN', 'CIPROFLOXACIN', 'AZITHROMYCIN', 'DOXYCYCLINE',
    'METRONIDAZOLE', 'TRIMETHOPRIM', 'FLUCONAZOLE', 'KETOCONAZOLE',
    'CLINDAMYCIN', 'VANCOMYCIN',
    # Immunosuppressants / oncology-adjacent
    'METHOTREXATE', 'CYCLOSPORINE', 'TACROLIMUS',
    'PREDNISONE', 'DEXAMETHASONE', 'METHYLPREDNISOLONE',
    'ALLOPURINOL', 'COLCHICINE', 'HYDROXYCHLOROQUINE',
    # Diabetes / metabolic
    'METFORMIN', 'GLIPIZIDE', 'SITAGLIPTIN', 'INSULIN',
    # GI / other
    'OMEPRAZOLE', 'PANTOPRAZOLE', 'ONDANSETRON', 'LOPERAMIDE',
    'LEVOTHYROXINE', 'DIPHENHYDRAMINE',
]

# ── MedDRA PT → organ system keyword mapping ──────────────────────────────────
# Terms are UPPERCASE (matching FAERS output).
# A PT is assigned to the FIRST category whose keywords appear in the term.
# Priority order matters for ambiguous terms (e.g. THROMBOSIS → Cardiovascular > Hematologic).
CATEGORY_KEYWORDS = {
    'Target_Hematologic': [
        'HAEMORRHAGE', 'HEMORRHAGE', 'HAEMATOMA', 'HEMATOMA', 'HAEMOPTYSIS',
        'HAEMATURIA', 'HAEMATEMESIS', 'RECTAL HAEMORRHAGE', 'GASTROINTESTINAL HAEMORRHAGE',
        'SUBDURAL HAEMATOMA', 'ANAEMIA', 'ANEMIA', 'THROMBOCYTOPENIA', 'NEUTROPENIA',
        'LEUKOPENIA', 'LEUCOPENIA', 'LYMPHOPENIA', 'PANCYTOPENIA', 'AGRANULOCYTOSIS',
        'COAGULOPATHY', 'PROTHROMBIN TIME', 'INTERNATIONAL NORMALISED RATIO',
        'PLATELET COUNT', 'WHITE BLOOD CELL', 'RED BLOOD CELL', 'HAEMOGLOBIN',
        'HAEMATOCRIT', 'PURPURA', 'PETECHIAE', 'ECCHYMOSIS', 'EPISTAXIS',
        'BLOOD COAGULATION', 'DISSEMINATED INTRAVASCULAR COAGULATION',
        'HAEMOLYSIS', 'HAEMOLYTIC', 'BLEEDING', 'POLYCYTHAEMIA',
    ],
    'Target_Cardiovascular': [
        'CARDIAC ARREST', 'MYOCARDIAL INFARCTION', 'HEART FAILURE', 'CARDIAC FAILURE',
        'ATRIAL FIBRILLATION', 'VENTRICULAR FIBRILLATION', 'VENTRICULAR TACHYCARDIA',
        'TACHYCARDIA', 'BRADYCARDIA', 'ARRHYTHMIA', 'PALPITATIONS',
        'HYPERTENSION', 'HYPOTENSION', 'ANGINA', 'QT PROLONGATION', 'QTC',
        'SYNCOPE', 'MYOCARDITIS', 'PERICARDITIS', 'CARDIOMYOPATHY',
        'DEEP VEIN THROMBOSIS', 'PULMONARY EMBOLISM', 'THROMBOEMBOLISM',
        'THROMBOSIS', 'EMBOLISM', 'STROKE', 'TRANSIENT ISCHAEMIC',
        'CEREBROVASCULAR', 'PERIPHERAL ARTERIAL', 'AORTIC', 'VASCULAR',
        'ISCHAEMIA', 'ISCHEMIA', 'CORONARY', 'ELECTROCARDIOGRAM',
    ],
    'Target_Hepatobiliary': [
        'HEPATOTOXICITY', 'HEPATITIS', 'HEPATIC FAILURE', 'LIVER FAILURE',
        'HEPATIC NECROSIS', 'LIVER INJURY', 'DRUG-INDUCED LIVER',
        'ALANINE AMINOTRANSFERASE', 'ASPARTATE AMINOTRANSFERASE',
        'GAMMA-GLUTAMYLTRANSFERASE', 'ALKALINE PHOSPHATASE', 'TRANSAMINASE',
        'BILIRUBIN', 'JAUNDICE', 'CHOLESTASIS', 'CHOLESTATIC', 'BILIARY',
        'HEPATIC', 'HEPATOCELLULAR', 'LIVER DISORDER', 'HEPATOMEGALY',
        'CIRRHOSIS', 'PORTAL HYPERTENSION', 'CHOLANGITIS',
    ],
    'Target_Nervous_System': [
        'HEADACHE', 'MIGRAINE', 'DIZZINESS', 'VERTIGO', 'SOMNOLENCE',
        'INSOMNIA', 'PERIPHERAL NEUROPATHY', 'NEUROPATHY', 'SEIZURE',
        'CONVULSION', 'EPILEPSY', 'TREMOR', 'CONFUSION', 'ENCEPHALOPATHY',
        'ENCEPHALITIS', 'COGNITIVE DISORDER', 'MEMORY IMPAIRMENT', 'AMNESIA',
        'PARAESTHESIA', 'PARESTHESIA', 'ATAXIA', 'DYSARTHRIA', 'DYSGEUSIA',
        'HALLUCINATION', 'SEDATION', 'CENTRAL NERVOUS', 'CEREBRAL',
        'MENINGITIS', 'NEUROTOXICITY', 'DEPRESSED LEVEL OF CONSCIOUSNESS',
        'LOSS OF CONSCIOUSNESS', 'SEROTONIN SYNDROME',
    ],
    'Target_Respiratory': [
        'DYSPNOEA', 'DYSPNEA', 'RESPIRATORY FAILURE', 'RESPIRATORY DISTRESS',
        'RESPIRATORY DEPRESSION', 'COUGH', 'PNEUMONIA', 'PNEUMONITIS',
        'INTERSTITIAL LUNG DISEASE', 'PULMONARY FIBROSIS', 'PULMONARY TOXICITY',
        'PULMONARY OEDEMA', 'PULMONARY EDEMA', 'PLEURAL EFFUSION',
        'BRONCHOSPASM', 'ASTHMA', 'BRONCHOCONSTRICTION', 'WHEEZING',
        'PHARYNGITIS', 'NASOPHARYNGITIS', 'RHINITIS', 'HYPOXIA',
        'OXYGEN SATURATION', 'RESPIRATORY TRACT', 'UPPER RESPIRATORY',
    ],
    'Target_Musculoskeletal': [
        'RHABDOMYOLYSIS', 'MYOPATHY', 'MYOSITIS', 'MYALGIA', 'MUSCLE',
        'CREATINE KINASE', 'CREATINE PHOSPHOKINASE', 'ARTHRALGIA', 'ARTHRITIS',
        'JOINT PAIN', 'BACK PAIN', 'BONE PAIN', 'OSTEOPOROSIS', 'OSTEOPENIA',
        'MUSCULOSKELETAL', 'TENDON', 'TENDINITIS', 'SYNOVITIS', 'GOUT',
        'MUSCULAR WEAKNESS', 'MUSCLE WEAKNESS', 'MUSCLE SPASM',
    ],
    'Target_Renal': [
        'ACUTE KIDNEY INJURY', 'RENAL FAILURE', 'RENAL IMPAIRMENT',
        'RENAL INSUFFICIENCY', 'NEPHROTOXICITY', 'NEPHROPATHY',
        'GLOMERULONEPHRITIS', 'TUBULOINTERSTITIAL', 'PROTEINURIA',
        'CREATININE', 'BLOOD UREA NITROGEN', 'OLIGURIA', 'ANURIA',
        'POLYURIA', 'HAEMATURIA', 'URINARY RETENTION', 'URINARY TRACT',
        'RENAL DISORDER', 'KIDNEY DISORDER', 'ELECTROLYTE IMBALANCE',
        'HYPONATRAEMIA', 'HYPERNATRAEMIA', 'HYPERKALAEMIA', 'HYPOKALAEMIA',
        'RENAL TUBULAR', 'NEPHROLITHIASIS',
    ],
    'Target_Gastrointestinal': [
        'NAUSEA', 'VOMITING', 'DIARRHOEA', 'DIARRHEA', 'CONSTIPATION',
        'ABDOMINAL PAIN', 'ABDOMINAL DISCOMFORT', 'GASTROINTESTINAL',
        'GASTRITIS', 'GASTROENTERITIS', 'COLITIS', 'PANCREATITIS',
        'PEPTIC ULCER', 'ULCER', 'DYSPEPSIA', 'MUCOSITIS', 'STOMATITIS',
        'FLATULENCE', 'BLOATING', 'BOWEL', 'COLON', 'RECTAL',
        'OESOPHAGEAL', 'ESOPHAGEAL', 'GASTROOESOPHAGEAL', 'REFLUX',
        'ILEITIS', 'ENTERITIS', 'INTESTINAL', 'MESENTERIC',
    ],
    'Target_Dermatologic': [
        'STEVENS-JOHNSON', 'TOXIC EPIDERMAL NECROLYSIS', 'DRUG REACTION WITH EOSINOPHILIA',
        'DRESS SYNDROME', 'RASH', 'PRURITUS', 'URTICARIA', 'ANGIOEDEMA',
        'ALOPECIA', 'DERMATITIS', 'ECZEMA', 'ERYTHEMA MULTIFORME', 'ERYTHEMA',
        'PHOTOSENSITIVITY', 'SKIN REACTION', 'SKIN DISORDER', 'SKIN RASH',
        'ACNEIFORM', 'BULLOUS', 'VESICULAR', 'SWEATING', 'HYPERHIDROSIS',
        'DRY SKIN', 'EXFOLIATIVE', 'MACULOPAPULAR', 'EXANTHEM',
        'NAIL DISORDER', 'HYPERPIGMENTATION',
    ],
}

# ── Cache helpers ──────────────────────────────────────────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key):
    safe = key.replace('/', '_').replace(' ', '_').replace('+', '_PLUS_')
    return os.path.join(CACHE_DIR, f"{safe}.json")

def _load_cache(key):
    p = _cache_path(key)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def _save_cache(key, data):
    with open(_cache_path(key), 'w') as f:
        json.dump(data, f)

# ── FAERS API ──────────────────────────────────────────────────────────────────
FAERS_BASE = 'https://api.fda.gov/drug/event.json'

def _faers_get(params, cache_key):
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached
    if API_KEY:
        params['api_key'] = API_KEY
    try:
        resp = requests.get(FAERS_BASE, params=params, timeout=20)
        time.sleep(REQUEST_DELAY)
        if resp.status_code == 200:
            data = resp.json()
            _save_cache(cache_key, data)
            return data
        if resp.status_code == 404:
            _save_cache(cache_key, {})
            return {}
    except Exception as e:
        print(f"   FAERS API error ({cache_key}): {e}")
        time.sleep(2)
    return {}

def drug_query(drug_name):
    """Quoted FAERS medicinalproduct search term."""
    return f'patient.drug.medicinalproduct:"{drug_name.upper()}"'

def get_total_reports(drug_name):
    key    = f"total_{drug_name}"
    params = {'search': drug_query(drug_name), 'limit': 1}
    data   = _faers_get(params, key)
    try:
        return data['meta']['results']['total']
    except (KeyError, TypeError):
        return 0

def get_ae_counts(drug_name):
    """Returns {meddra_pt_upper: count} for top-N AEs for a single drug."""
    key    = f"ae_{drug_name}"
    params = {'search': drug_query(drug_name),
              'count': 'patient.reaction.reactionmeddrapt.exact',
              'limit': N_AE_TOP}
    data   = _faers_get(params, key)
    try:
        return {r['term'].upper(): r['count'] for r in data.get('results', [])}
    except (KeyError, TypeError):
        return {}

def get_pair_total(drug_a, drug_b):
    """Total co-reports for the drug pair (order-invariant)."""
    a, b   = sorted([drug_a.upper(), drug_b.upper()])
    key    = f"ptotal_{a}_{b}"
    search = f'{drug_query(a)} AND {drug_query(b)}'
    params = {'search': search, 'limit': 1}
    data   = _faers_get(params, key)
    try:
        return data['meta']['results']['total']
    except (KeyError, TypeError):
        return 0

def get_pair_ae_counts(drug_a, drug_b):
    """MedDRA AE counts for co-reports of drug pair."""
    a, b   = sorted([drug_a.upper(), drug_b.upper()])
    key    = f"paes_{a}_{b}"
    search = f'{drug_query(a)} AND {drug_query(b)}'
    params = {'search': search,
              'count': 'patient.reaction.reactionmeddrapt.exact',
              'limit': N_AE_TOP}
    data   = _faers_get(params, key)
    try:
        return {r['term'].upper(): r['count'] for r in data.get('results', [])}
    except (KeyError, TypeError):
        return {}

# ── MedDRA PT → 9 target categories ───────────────────────────────────────────
def _classify_pt(pt_upper):
    """Return the target category name for a MedDRA PT, or None if unclassified."""
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in pt_upper:
                return cat
    return None

def ae_counts_to_category_rates(ae_counts, total_reports):
    """
    Compute per-category adverse event rates using the union probability formula:
      P(≥1 AE in category) = 1 - ∏(1 - P(AEᵢ))   for all AEᵢ in the category

    Summing individual AE counts inflates rates because one report can mention
    several AEs from the same organ system. The union formula counts each report
    at most once per category regardless of how many category AEs it lists.

    Returns {col: rate_%} where rate is in [0, 100].
    """
    if total_reports == 0:
        return {col: 0.0 for col in TARGET_COLS}

    cat_rates = {col: [] for col in TARGET_COLS}
    for pt, count in ae_counts.items():
        cat = _classify_pt(pt)
        if cat:
            cat_rates[cat].append(min(count / total_reports, 1.0))

    # Clinical ceiling: no common drug combination causes >65% AE incidence in
    # a single organ system.  FAERS CNS-drug pairs saturate near 100% because
    # epilepsy / psychiatry patients always have a CNS event in the report;
    # capping at 65% preserves relative ordering while staying clinically plausible.
    CAP = 65.0

    result = {}
    for col, rates in cat_rates.items():
        if not rates:
            result[col] = 0.0
        else:
            p_none = 1.0
            for r in rates:
                p_none *= (1.0 - r)
            result[col] = round(min((1.0 - p_none) * 100.0, CAP), 4)
    return result

# ── PubChem SMILES ─────────────────────────────────────────────────────────────
def fetch_smiles(drug_name):
    key    = f"smiles_{drug_name}"
    cached = _load_cache(key)
    if cached is not None:
        return cached.get('smiles', '')
    try:
        name = drug_name.strip().lower()
        url  = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{urllib.parse.quote(name)}/property/CanonicalSMILES/TXT")
        resp = requests.get(url, timeout=10)
        time.sleep(0.2)
        if resp.status_code == 200:
            smi = resp.text.strip()
            _save_cache(key, {'smiles': smi})
            return smi
    except Exception:
        pass
    _save_cache(key, {'smiles': ''})
    return ''

def is_small_molecule(smiles, max_mw=1200.0):
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    return Descriptors.MolWt(mol) <= max_mw

# ── Independence model fallback ────────────────────────────────────────────────
def independence_combine(rates_list):
    """Combine risk rates (%) assuming independence — upper-bound estimate."""
    p = 1.0
    for r in rates_list:
        p *= (1.0 - r / 100.0)
    return (1.0 - p) * 100.0

# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_drug_profiles(drug_list):
    """
    Returns dict: {drug_name: {'total': int, 'rates': {col: float}, 'smiles': str}}
    Filters out drugs with too few reports or no valid SMILES.
    """
    profiles = {}
    print(f"\nBuilding single-drug profiles for {len(drug_list)} candidates...")
    for drug in drug_list:
        total = get_total_reports(drug)
        if total < MIN_SINGLE_REPORTS:
            print(f"   Skip {drug}: only {total} reports (< {MIN_SINGLE_REPORTS})")
            continue

        smiles = fetch_smiles(drug)
        if not is_small_molecule(smiles):
            print(f"   Skip {drug}: no valid SMILES or macromolecule")
            continue

        ae_raw = get_ae_counts(drug)
        rates  = ae_counts_to_category_rates(ae_raw, total)

        profiles[drug] = {'total': total, 'rates': rates, 'smiles': smiles}
        covered = sum(1 for r in rates.values() if r > 0)
        print(f"   {drug:<20} {total:>8,} reports | "
              f"{covered}/9 categories covered | SMILES: {smiles[:30]}...")

    print(f"\n   {len(profiles)} drugs with sufficient data.")
    return profiles

def build_pair_rows(profiles):
    """
    Generate one training row per drug pair.
    Uses real FAERS DDI rates where co-reports ≥ MIN_PAIR_REPORTS,
    otherwise falls back to independence model on single-drug rates.
    """
    drug_list = sorted(profiles.keys())
    rows      = []
    real_ddi  = 0
    fallback  = 0

    total_pairs = len(drug_list) * (len(drug_list) - 1) // 2
    print(f"\nGenerating rows for {total_pairs} drug pairs...")

    for i, (drug_a, drug_b) in enumerate(combinations(drug_list, 2)):
        if (i + 1) % 100 == 0:
            print(f"   Pair {i+1}/{total_pairs}...")

        pair_total = get_pair_total(drug_a, drug_b)

        if pair_total >= MIN_PAIR_REPORTS:
            # Real DDI rates from FAERS co-reports
            pair_ae = get_pair_ae_counts(drug_a, drug_b)
            rates   = ae_counts_to_category_rates(pair_ae, pair_total)
            real_ddi += 1
        else:
            # Independence model fallback
            r_a  = profiles[drug_a]['rates']
            r_b  = profiles[drug_b]['rates']
            rates = {col: independence_combine([r_a[col], r_b[col]])
                     for col in TARGET_COLS}
            fallback += 1

        row = {
            'Drug_1':   drug_a.capitalize(),
            'SMILES_1': profiles[drug_a]['smiles'],
            'Drug_2':   drug_b.capitalize(),
            'SMILES_2': profiles[drug_b]['smiles'],
            **{col: round(rates[col], 4) for col in TARGET_COLS},
            '_source': 'FAERS_DDI' if pair_total >= MIN_PAIR_REPORTS else 'independence',
            '_pair_reports': pair_total,
        }
        rows.append(row)

    print(f"\n   {len(rows)} pairs total:")
    print(f"     {real_ddi}  with real FAERS DDI rates (≥{MIN_PAIR_REPORTS} co-reports)")
    print(f"     {fallback}  using independence model fallback")
    return rows

def add_monotherapy_rows(profiles):
    """Monotherapy rows (Drug_2 empty) for baseline single-drug risk."""
    rows = []
    for drug, p in profiles.items():
        row = {
            'Drug_1':   drug.capitalize(),
            'SMILES_1': p['smiles'],
            'Drug_2':   '',
            'SMILES_2': '',
            **{col: round(p['rates'][col], 4) for col in TARGET_COLS},
            '_source':       'FAERS_single',
            '_pair_reports': p['total'],
        }
        rows.append(row)
    return rows

def main():
    print("=" * 60)
    print("Phase 7: Building Real Training Data from FDA FAERS")
    print("=" * 60)
    print(f"Cache dir:   {CACHE_DIR}/")
    print(f"API key:     {'set' if API_KEY else 'not set (40 req/min limit)'}")
    print(f"Seed drugs:  {len(SEED_DRUGS)}")

    profiles = build_drug_profiles(SEED_DRUGS)
    if len(profiles) < 2:
        print("Error: fewer than 2 drugs with valid profiles. Check API connectivity.")
        return

    pair_rows  = build_pair_rows(profiles)
    mono_rows  = add_monotherapy_rows(profiles)
    all_rows   = pair_rows + mono_rows

    df = pd.DataFrame(all_rows)
    print(f"\nDataset summary:")
    print(f"  Total rows:         {len(df)}")
    print(f"  Pair rows:          {len(pair_rows)}")
    print(f"  Monotherapy rows:   {len(mono_rows)}")

    max_risks = df[TARGET_COLS].max(axis=1)
    print(f"  Max risk (any col): {max_risks.max():.2f}%  "
          f"mean={max_risks.mean():.2f}%  "
          f"zero={( max_risks == 0).sum()}")

    # Show top 10 highest-risk pairs
    pair_df = df[df['Drug_2'] != ''].copy()
    pair_df['_max'] = pair_df[TARGET_COLS].max(axis=1)
    print(f"\n  Top 10 highest-risk pairs:")
    for _, row in pair_df.nlargest(10, '_max').iterrows():
        driver = max(TARGET_COLS, key=lambda c: row[c]).replace('Target_', '')
        print(f"    {row['Drug_1']} + {row['Drug_2']:<18} "
              f"max={row['_max']:.2f}%  ({driver})  [{row['_source']}]")

    # Save — drop internal columns
    out_cols = ['Drug_1', 'SMILES_1', 'Drug_2', 'SMILES_2'] + TARGET_COLS
    df[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}  ({len(df)} rows)")
    print("\nNext step: retrain 022 using training_matrix_real.csv")
    print("  Update train_pipeline() input_file to 'training_matrix_real.csv'")

if __name__ == '__main__':
    main()
