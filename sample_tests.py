# -*- coding: utf-8 -*-
"""
Drophet — sample inference tests
Runs the trained N-Side DDI model against a set of clinically meaningful
drug combinations and prints per-organ-system risk breakdowns.
"""

import importlib.util, sys, os, warnings
warnings.filterwarnings('ignore')

# Load 022 as a module without executing __main__
spec = importlib.util.spec_from_file_location("nside", "022_nside_ddi_model.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NSideInferenceTool = mod.NSideInferenceTool

tool = NSideInferenceTool()
assert tool.is_ready, "Model not loaded — run training first."

COMBOS = [
    # (label, [drugs])
    ("Warfarin alone (baseline)",
     ["Warfarin"]),
    ("Aspirin alone (baseline)",
     ["Aspirin"]),
    ("Warfarin + Aspirin  (common anticoagulant + antiplatelet)",
     ["Warfarin", "Aspirin"]),
    ("Triple antithrombotic: Warfarin + Aspirin + Clopidogrel  (highest civilian bleed risk)",
     ["Warfarin", "Aspirin", "Clopidogrel"]),
    ("Simvastatin + Ketoconazole  (CYP3A4 inhibition → rhabdomyolysis risk)",
     ["Simvastatin", "Ketoconazole"]),
    ("Metformin + Lisinopril  (diabetes + ACE inhibitor, generally well tolerated)",
     ["Metformin", "Lisinopril"]),
    ("Ciprofloxacin + Amiodarone  (dual QT-prolongation risk)",
     ["Ciprofloxacin", "Amiodarone"]),
    ("Methotrexate + Ibuprofen  (NSAID reduces MTX renal clearance → toxicity)",
     ["Methotrexate", "Ibuprofen"]),
    ("Fluoxetine + Tramadol  (serotonin syndrome risk)",
     ["Fluoxetine", "Tramadol"]),
    ("Omeprazole + Levothyroxine  (low-risk: PPI reduces T4 absorption slightly)",
     ["Omeprazole", "Levothyroxine"]),
    ("Valproate + Carbamazepine + Phenytoin  (triple AED polypharmacy)",
     ["Valproate", "Carbamazepine", "Phenytoin"]),
    ("Tacrolimus + Fluconazole + Ibuprofen  (immunosuppressant + CYP inhibitor + NSAID)",
     ["Tacrolimus", "Fluconazole", "Ibuprofen"]),
]

BAR_SCALE = 40   # characters = 100%

def bar(pct):
    filled = int(round(pct / 100 * BAR_SCALE))
    if   pct < 5:  colour = "\033[92m"   # green
    elif pct < 20: colour = "\033[93m"   # yellow
    else:          colour = "\033[91m"   # red
    reset = "\033[0m"
    return f"{colour}{'█' * filled}{'░' * (BAR_SCALE - filled)}{reset}"

print("\n" + "═" * 70)
print("  DROPHET — N-SIDE DDI  ·  Sample Test Predictions")
print("═" * 70)

for label, drugs in COMBOS:
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"  Drugs: {', '.join(drugs)}")

    res = tool.predict(drugs)

    if "error" in res:
        print(f"  ✗  Error: {res['error']}")
        continue

    tier_sym  = res["tier"].split()[0]     # emoji
    tier_word = res["tier"].split()[-1]    # LOW / MODERATE / HIGH
    print(f"\n  Combined risk: {res['incidence']:>7}  {tier_sym} {tier_word}\n")
    print(f"  {'Organ system':<22}  {'%':>6}  {'':2}  Risk bar")
    print(f"  {'─'*22}  {'─'*6}  {'─'*2}  {'─'*BAR_SCALE}")
    for cat, risk_str in res["per_category"].items():
        pct = float(risk_str.replace("%", ""))
        print(f"  {cat:<22}  {risk_str:>6}  {'  '}{bar(pct)}")

print(f"\n{'═' * 70}\n")
