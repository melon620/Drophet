# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project (Project Drophet)
Phase 7b: OnSIDES Integration

Replaces FAERS single-drug AE baselines with label-stated profiles from
OnSIDES v3.1.1 (7.1 M drug-ADE pairs from official US/EU/UK/JP labels).

Problem with FAERS single-drug rates:
  FAERS spontaneous-report fractions are confounded by patient population.
  A CNS drug gets high Nervous_System FAERS rates not because it is uniquely
  neurotoxic, but because its patients already have CNS disease.
  Official drug labels (OnSIDES) attribute AEs to the drug specifically.

Integration strategy:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Row type              │ Treatment                              │
  │  ─────────────────── ─ │ ─────────────────────────────────────  │
  │  FAERS_DDI pair        │ Keep FAERS DDI rates unchanged         │
  │  (co-reports ≥ 50)     │ (pair-level signal most accurate)      │
  │                        │                                        │
  │  Independence fallback │ Recompute using blended single-drug    │
  │  pair (co-reports < 50)│ rates: 0.4 × FAERS + 0.6 × OnSIDES   │
  │                        │                                        │
  │  Monotherapy row       │ Replace with blended rates             │
  └─────────────────────────────────────────────────────────────────┘

OnSIDES score per category:
  For each drug ingredient, collect all unique MedDRA AEs with
  pred1 > PRED_THRESHOLD from AR / BW / WP label sections (US + EU).
  score[cat] = count(AEs in cat) / total_classified_AEs × 100
  Capped at 65 (same ceiling as FAERS pipeline in script 024).

Blend weight 0.6 for OnSIDES reflects:
  - Label-stated AEs are more specifically attributed to the drug
  - FAERS rates still provide calibration to real-world frequency
  - Equal blend empirically tested on warfarin / simvastatin reference drugs

Requirements: onsides_data/onsides.db (built by 025_onsides_integration.py
  using OnSIDES v3.1.1 CSV files), training_matrix_real.csv, faers_cache/
"""

import json
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd

from drophet_utils import seed_everything, pair_key

warnings.filterwarnings('ignore')
seed_everything(42)

# ── Config ─────────────────────────────────────────────────────────────────────
ONSIDES_DB       = 'onsides_data/onsides.db'
INPUT_FILE       = 'training_matrix_real.csv'
OUTPUT_FILE      = 'training_matrix_real.csv'       # overwrite in place
FAERS_CACHE_DIR  = 'faers_cache'
PRED_THRESHOLD   = 0.5                              # OnSIDES BERT confidence cutoff
MIN_PAIR_REPORTS = 50                               # same as script 024
ONSIDES_WEIGHT   = 0.6                              # blend weight for OnSIDES
FAERS_WEIGHT     = 1.0 - ONSIDES_WEIGHT
AE_SCALE_CAP     = 65.0                             # max % per category

TARGET_COLS = [
    'Target_Hematologic', 'Target_Cardiovascular', 'Target_Hepatobiliary',
    'Target_Nervous_System', 'Target_Respiratory', 'Target_Musculoskeletal',
    'Target_Renal', 'Target_Gastrointestinal', 'Target_Dermatologic',
]

# Mirrors CATEGORY_KEYWORDS from 024 — must stay in sync
CATEGORY_KEYWORDS = {
    'Target_Hematologic': [
        'HAEMORRHAGE', 'HEMORRHAGE', 'HAEMATOMA', 'HEMATOMA', 'HAEMOPTYSIS',
        'HAEMATURIA', 'HAEMATEMESIS', 'RECTAL HAEMORRHAGE',
        'GASTROINTESTINAL HAEMORRHAGE', 'SUBDURAL HAEMATOMA',
        'ANAEMIA', 'ANEMIA', 'THROMBOCYTOPENIA', 'NEUTROPENIA',
        'LEUKOPENIA', 'LEUCOPENIA', 'LYMPHOPENIA', 'PANCYTOPENIA',
        'AGRANULOCYTOSIS', 'COAGULOPATHY', 'PROTHROMBIN TIME',
        'INTERNATIONAL NORMALISED RATIO', 'PLATELET COUNT',
        'WHITE BLOOD CELL', 'RED BLOOD CELL', 'HAEMOGLOBIN',
        'HAEMATOCRIT', 'PURPURA', 'PETECHIAE', 'ECCHYMOSIS', 'EPISTAXIS',
        'BLOOD COAGULATION', 'DISSEMINATED INTRAVASCULAR COAGULATION',
        'HAEMOLYSIS', 'HAEMOLYTIC', 'BLEEDING', 'POLYCYTHAEMIA',
    ],
    'Target_Cardiovascular': [
        'CARDIAC ARREST', 'MYOCARDIAL INFARCTION', 'HEART FAILURE',
        'CARDIAC FAILURE', 'ATRIAL FIBRILLATION', 'VENTRICULAR FIBRILLATION',
        'VENTRICULAR TACHYCARDIA', 'TACHYCARDIA', 'BRADYCARDIA',
        'ARRHYTHMIA', 'PALPITATIONS', 'HYPERTENSION', 'HYPOTENSION',
        'ANGINA', 'QT PROLONGATION', 'QTC', 'SYNCOPE', 'MYOCARDITIS',
        'PERICARDITIS', 'CARDIOMYOPATHY', 'DEEP VEIN THROMBOSIS',
        'PULMONARY EMBOLISM', 'THROMBOEMBOLISM', 'THROMBOSIS', 'EMBOLISM',
        'STROKE', 'TRANSIENT ISCHAEMIC', 'CEREBROVASCULAR',
        'PERIPHERAL ARTERIAL', 'AORTIC', 'VASCULAR', 'ISCHAEMIA',
        'ISCHEMIA', 'CORONARY', 'ELECTROCARDIOGRAM',
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
        'STEVENS-JOHNSON', 'TOXIC EPIDERMAL NECROLYSIS',
        'DRUG REACTION WITH EOSINOPHILIA', 'DRESS SYNDROME', 'RASH',
        'PRURITUS', 'URTICARIA', 'ANGIOEDEMA', 'ALOPECIA', 'DERMATITIS',
        'ECZEMA', 'ERYTHEMA MULTIFORME', 'ERYTHEMA', 'PHOTOSENSITIVITY',
        'SKIN REACTION', 'SKIN DISORDER', 'SKIN RASH', 'ACNEIFORM',
        'BULLOUS', 'VESICULAR', 'SWEATING', 'HYPERHIDROSIS',
        'DRY SKIN', 'EXFOLIATIVE', 'MACULOPAPULAR', 'EXANTHEM',
        'NAIL DISORDER', 'HYPERPIGMENTATION',
    ],
}


def classify_pt(meddra_name: str) -> str | None:
    n = meddra_name.upper()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in n:
                return cat
    return None


# ── OnSIDES query ──────────────────────────────────────────────────────────────

def build_onsides_profiles(drug_names: list[str], db_path: str) -> dict:
    """
    Returns {drug_name_upper: {Target_*: score_%}} for each drug found in OnSIDES.
    score_% = (distinct AEs classified in category) / (total classified AEs) × 100,
    capped at AE_SCALE_CAP.

    Uses US + EU labels. Each MedDRA AE is counted once per drug regardless of
    how many product labels repeat it (deduplicated by meddra_id).
    Multi-source confirmed AEs (present in both US and EU) get weight 2 vs 1.
    """
    con = sqlite3.connect(db_path)

    # vocab_rxnorm_ingredient_to_product was loaded from CSV with columns
    # (product_id, ingredient_id) into SQLite schema (ingredient_id, product_id)
    # — so the DB columns are swapped relative to their names.
    # Correct join: ip.product_id = ingredient rxnorm_id
    #               ip.ingredient_id = product rxnorm_id (joins to product_to_rxnorm)

    query = """
        SELECT
            m.meddra_name,
            COUNT(DISTINCT pl.source) AS num_sources
        FROM vocab_rxnorm_ingredient           i
        JOIN vocab_rxnorm_ingredient_to_product ip  ON ip.product_id       = i.rxnorm_id
        JOIN vocab_rxnorm_product               p   ON p.rxnorm_id         = ip.ingredient_id
        JOIN product_to_rxnorm                  ptr ON ptr.rxnorm_product_id = p.rxnorm_id
        JOIN product_label                      pl  ON pl.label_id         = ptr.label_id
        JOIN product_adverse_effect             a   ON a.product_label_id  = pl.label_id
        JOIN vocab_meddra_adverse_effect        m   ON m.meddra_id         = a.effect_meddra_id
        WHERE lower(i.rxnorm_name) = ?
          AND a.pred1  > ?
          AND a.label_section IN ('AR', 'BW', 'WP')
          AND pl.source IN ('US', 'EU')
        GROUP BY m.meddra_id
        HAVING COUNT(DISTINCT pl.source) >= 1
    """

    profiles = {}
    found = 0
    for drug in drug_names:
        rows = con.execute(query, (drug.lower(), PRED_THRESHOLD)).fetchall()
        if not rows:
            continue

        cat_weight = {c: 0.0 for c in TARGET_COLS}
        total_w = 0.0
        for meddra_name, num_sources in rows:
            cat = classify_pt(meddra_name)
            if cat is None:
                continue
            w = float(num_sources)          # US-only=1, US+EU=2 (higher confidence)
            cat_weight[cat] += w
            total_w += w

        if total_w == 0:
            profiles[drug.upper()] = {c: 0.0 for c in TARGET_COLS}
            continue

        profiles[drug.upper()] = {
            c: round(min(cat_weight[c] / total_w * 100.0, AE_SCALE_CAP), 4)
            for c in TARGET_COLS
        }
        found += 1

    con.close()
    print(f"   OnSIDES profiles built: {found}/{len(drug_names)} drugs found")
    return profiles


# ── FAERS cache helpers ────────────────────────────────────────────────────────

def _cache_pair_total(drug_a: str, drug_b: str) -> int | None:
    """Read cached FAERS co-report count for a drug pair, or None if not cached."""
    a, b = sorted([drug_a.upper(), drug_b.upper()])
    safe = lambda s: s.replace('/', '_').replace(' ', '_').replace('+', '_PLUS_')
    path = os.path.join(FAERS_CACHE_DIR, f"ptotal_{safe(a)}_{safe(b)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get('meta', {}).get('results', {}).get('total', 0)
    except Exception:
        return None


def _cache_single_rates(drug: str) -> dict | None:
    """Read cached FAERS single-drug category rates, or None if not cached."""
    safe = lambda s: s.replace('/', '_').replace(' ', '_').replace('+', '_PLUS_')
    ae_path    = os.path.join(FAERS_CACHE_DIR, f"ae_{safe(drug.upper())}.json")
    total_path = os.path.join(FAERS_CACHE_DIR, f"total_{safe(drug.upper())}.json")
    if not (os.path.exists(ae_path) and os.path.exists(total_path)):
        return None
    try:
        with open(ae_path) as f:
            ae_data = json.load(f)
        with open(total_path) as f:
            total_data = json.load(f)
        total = total_data.get('meta', {}).get('results', {}).get('total', 0)
        ae_counts = {r['term'].upper(): r['count']
                     for r in ae_data.get('results', [])}
    except Exception:
        return None

    if total == 0:
        return {c: 0.0 for c in TARGET_COLS}

    # Mirror ae_counts_to_category_rates from script 024
    cat_rates: dict[str, list[float]] = {c: [] for c in TARGET_COLS}
    for pt, count in ae_counts.items():
        cat = classify_pt(pt)
        if cat:
            cat_rates[cat].append(min(count / total, 1.0))

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


# ── Independence model ─────────────────────────────────────────────────────────

def independence_combine(rates: list[float]) -> float:
    p = 1.0
    for r in rates:
        p *= (1.0 - r / 100.0)
    return (1.0 - p) * 100.0


def blend_rates(faers: dict, onsides: dict) -> dict:
    """Blend FAERS and OnSIDES single-drug category rates."""
    blended = {}
    for col in TARGET_COLS:
        f = faers.get(col, 0.0)
        o = onsides.get(col, 0.0)
        if o > 0 and f > 0:
            blended[col] = round(FAERS_WEIGHT * f + ONSIDES_WEIGHT * o, 4)
        elif o > 0:
            # FAERS missed it — use OnSIDES at reduced weight
            blended[col] = round(0.5 * o, 4)
        else:
            blended[col] = round(f, 4)
    return blended


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("Phase 7b: OnSIDES Integration")
    print("=" * 62)

    if not os.path.exists(ONSIDES_DB):
        print(f"Error: {ONSIDES_DB} not found.")
        print("  The SQLite database should have been built already.")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run script 024 first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    # All unique drugs that appear as Drug_1 or Drug_2
    all_drugs = set(df['Drug_1'].dropna().unique())
    all_drugs.update(df['Drug_2'].dropna().unique())
    all_drugs.discard('')
    drug_list = sorted(all_drugs)
    print(f"Unique drugs in matrix: {len(drug_list)}")

    # Build OnSIDES profiles
    print("\nQuerying OnSIDES for per-drug AE profiles...")
    onsides_profiles = build_onsides_profiles(drug_list, ONSIDES_DB)

    # Build FAERS single-drug rates from cache
    print("Loading FAERS single-drug rates from cache...")
    faers_single: dict[str, dict] = {}
    for drug in drug_list:
        rates = _cache_single_rates(drug.upper())
        if rates:
            faers_single[drug.upper()] = rates

    print(f"   FAERS cache hits: {len(faers_single)}/{len(drug_list)}")
    print(f"   OnSIDES coverage: {len(onsides_profiles)}/{len(drug_list)}")

    # Build blended single-drug profiles
    blended_profiles: dict[str, dict] = {}
    for drug in drug_list:
        key = drug.upper()
        f = faers_single.get(key, {c: 0.0 for c in TARGET_COLS})
        o = onsides_profiles.get(key, {c: 0.0 for c in TARGET_COLS})
        blended_profiles[drug] = blend_rates(f, o)

    # Show OnSIDES coverage summary
    onsides_covered = sum(1 for d in drug_list if d.upper() in onsides_profiles)
    print(f"\n   Blended profiles built for {len(blended_profiles)} drugs")
    print(f"   OnSIDES data available: {onsides_covered}/{len(drug_list)} drugs")

    # Print sample blended profiles
    print("\n   Sample blended profiles (top 2 categories):")
    for drug in ['Warfarin', 'Simvastatin', 'Ciprofloxacin', 'Metformin']:
        if drug in blended_profiles:
            p = blended_profiles[drug]
            top2 = sorted(p.items(), key=lambda x: -x[1])[:2]
            print(f"     {drug:<18} {[(k.replace('Target_',''),round(v,1)) for k,v in top2]}")

    # Rebuild rows
    print("\nRebuilding training matrix...")
    new_rows = []
    faers_ddi_count  = 0
    recomputed_count = 0
    mono_count       = 0
    no_onsides       = 0

    for _, row in df.iterrows():
        d1 = str(row['Drug_1']) if pd.notna(row['Drug_1']) else ''
        d2 = str(row['Drug_2']) if pd.notna(row['Drug_2']) else ''
        s1 = str(row['SMILES_1']) if pd.notna(row['SMILES_1']) else ''
        s2 = str(row['SMILES_2']) if pd.notna(row['SMILES_2']) else ''

        new_row = {
            'Drug_1': d1, 'SMILES_1': s1,
            'Drug_2': d2, 'SMILES_2': s2,
        }

        if not d2 or d2 == 'nan':
            # Monotherapy row — use blended single-drug profile
            profile = blended_profiles.get(d1, {c: float(row[c]) for c in TARGET_COLS})
            for col in TARGET_COLS:
                new_row[col] = round(profile.get(col, float(row[col])), 4)
            mono_count += 1

        else:
            # Pair row — check if FAERS has DDI data
            pair_reports = _cache_pair_total(d1, d2)

            if pair_reports is not None and pair_reports >= MIN_PAIR_REPORTS:
                # Real FAERS DDI data — keep as-is
                for col in TARGET_COLS:
                    new_row[col] = float(row[col])
                faers_ddi_count += 1

            else:
                # Independence fallback — rebuild with blended single-drug rates
                p1 = blended_profiles.get(d1)
                p2 = blended_profiles.get(d2)

                if p1 is None or p2 is None:
                    # No profile available — keep original
                    for col in TARGET_COLS:
                        new_row[col] = float(row[col])
                    no_onsides += 1
                else:
                    for col in TARGET_COLS:
                        new_row[col] = round(
                            independence_combine([p1[col], p2[col]]), 4)
                    recomputed_count += 1

        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    print(f"\n   Rows processed:")
    print(f"     FAERS DDI pairs (kept)    : {faers_ddi_count:5}")
    print(f"     Independence (recomputed) : {recomputed_count:5}")
    print(f"     Monotherapy  (blended)    : {mono_count:5}")
    print(f"     No profile (kept as-is)   : {no_onsides:5}")

    # Stats comparison
    print("\n   Before vs After target distribution (mean %):")
    print(f"   {'Category':<25} {'Before':>7} {'After':>7}")
    for col in TARGET_COLS:
        b = df[col].mean()
        a = new_df[col].mean()
        print(f"   {col.replace('Target_',''):<25} {b:>7.2f} {a:>7.2f}")

    out_cols = ['Drug_1', 'SMILES_1', 'Drug_2', 'SMILES_2'] + TARGET_COLS
    new_df[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}  ({len(new_df)} rows)")
    print("Next step: retrain 022 on the updated matrix (FORCE_RETRAIN=True)")


if __name__ == '__main__':
    main()
