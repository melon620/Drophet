# -*- coding: utf-8 -*-
"""
Synthetic-data generator for smoke-testing the modeling pipeline without
the real Codespaces-trained dataset.

Produces:
    ddi_training_dataset_final.csv
        ↳ schema for 017_baseline_xgboost.py
        ↳ Drug_1, Drug_2, D{1,2}_Bit_*, D{1,2}_{MW,LogP,TPSA,HDonors,HAcceptors},
          Target_AE_<category>

    training_matrix_augmented.csv
        ↳ schema for 019_train_gnn_model.py
        ↳ Drug_1, Drug_2, SMILES_1, SMILES_2, Target_AE_<category>

This is a SMOKE-TEST fixture. The labels are random; metrics on this
data are meaningless. The point is to exercise the new code paths
end-to-end (nested CV, Optuna, calibration plots, weighted multi-task
loss) without touching the real training data.

Run from repo root:
    .venv-local/bin/python tests/make_synthetic_fixtures.py
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

REAL_DRUGS = [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("Acetaminophen", "CC(=O)NC1=CC=C(C=C1)O"),
    ("Warfarin", "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"),
    ("Metformin", "CN(C)C(=N)N=C(N)N"),
    ("Lisinopril", "C1CC(N(C1)C(=O)C(CCCCN)NC(CCC2=CC=CC=C2)C(=O)O)C(=O)O"),
    ("Simvastatin", "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C"),
    ("Atorvastatin", "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"),
    ("Sildenafil", "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"),
    ("Clarithromycin", "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O"),
    ("Methotrexate", "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("Trimethoprim", "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N"),
    ("Loratadine", "CCOC(=O)N1CCC(=C2C3=C(CCC4=CC(=CN=C24)Cl)C=CC=C3)CC1"),
    ("Amoxicillin", "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"),
    ("Furosemide", "C1=CC(=C(C=C1Cl)S(=O)(=O)N)NCC2=CC=CO2"),
    ("Diltiazem", "CC(=O)OC1C(SC2=CC=CC=C2N(C1=O)CCN(C)C)C3=CC=C(C=C3)OC"),
    ("Tramadol", "CN(C)CC1(CCCCC1=O)C2=CC(=CC=C2)OC"),
    ("Fluoxetine", "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F"),
    ("Levothyroxine", "C1=CC(=C(C=C1CC(C(=O)O)N)I)OC2=CC(=C(C(=C2)I)O)I"),
    ("Selegiline", "CC(CC1=CC=CC=C1)N(C)CC#C"),
]

N_BITS = 2048
TARGET_CATEGORIES = ["GASTROINTESTINAL", "HEPATIC", "RENAL", "CARDIAC", "DERMATOLOGICAL"]


def descriptor_block(smiles: str, prefix: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        bits = [0] * N_BITS
        return bits, [0.0, 0.0, 0.0, 0.0, 0.0]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)
    bits = list(fp)
    descs = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHAcceptors(mol)),
    ]
    return bits, descs


def make_pairs(rng: np.random.Generator, n_pairs: int = 60):
    rows = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(REAL_DRUGS), size=2)
        if i == j:
            j = (j + 1) % len(REAL_DRUGS)
        d1, s1 = REAL_DRUGS[i]
        d2, s2 = REAL_DRUGS[j]
        rows.append((d1, s1, d2, s2))
    # Add a few monotherapy rows (Drug 2 empty)
    for _ in range(8):
        i = rng.integers(0, len(REAL_DRUGS))
        d1, s1 = REAL_DRUGS[i]
        rows.append((d1, s1, "", ""))
    return rows


def synth_targets(rng: np.random.Generator, n_rows: int):
    targets = {}
    for cat in TARGET_CATEGORIES:
        # Class-imbalanced: ~85% near-zero, ~15% elevated
        is_pos = rng.random(n_rows) < 0.15
        vals = np.where(
            is_pos,
            rng.uniform(5.0, 60.0, n_rows),
            rng.uniform(0.0, 4.0, n_rows),
        )
        targets[f"Target_AE_{cat}"] = vals.round(3)
    return targets


def write_xgboost_fixture(rows, targets, path: str):
    bits1, descs1, bits2, descs2 = [], [], [], []
    for d1, s1, d2, s2 in rows:
        b1, x1 = descriptor_block(s1, "D1")
        b2, x2 = descriptor_block(s2 if s2 else "", "D2") if s2 else (
            [0] * N_BITS, [0.0, 0.0, 0.0, 0.0, 0.0]
        )
        bits1.append(b1); descs1.append(x1)
        bits2.append(b2); descs2.append(x2)

    cols1_bits = [f"D1_Bit_{i}" for i in range(N_BITS)]
    cols2_bits = [f"D2_Bit_{i}" for i in range(N_BITS)]
    cols1_desc = ["D1_MW", "D1_LogP", "D1_TPSA", "D1_HDonors", "D1_HAcceptors"]
    cols2_desc = ["D2_MW", "D2_LogP", "D2_TPSA", "D2_HDonors", "D2_HAcceptors"]

    df = pd.DataFrame({
        "Drug_1": [r[0] for r in rows],
        "Drug_2": [r[2] for r in rows],
    })
    df = pd.concat([
        df,
        pd.DataFrame(bits1, columns=cols1_bits),
        pd.DataFrame(descs1, columns=cols1_desc),
        pd.DataFrame(bits2, columns=cols2_bits),
        pd.DataFrame(descs2, columns=cols2_desc),
    ], axis=1)
    for k, v in targets.items():
        df[k] = v
    df.to_csv(path, index=False)
    return df.shape


def write_gnn_fixture(rows, targets, path: str):
    df = pd.DataFrame({
        "Drug_1": [r[0] for r in rows],
        "SMILES_1": [r[1] for r in rows],
        "Drug_2": [r[2] for r in rows],
        "SMILES_2": [r[3] for r in rows],
    })
    for k, v in targets.items():
        df[k] = v
    df.to_csv(path, index=False)
    return df.shape


def main():
    # Refuse to overwrite real data. The synthetic fixtures share the same
    # filenames as real pipeline outputs, so on Codespaces (where the real
    # data lives) running this script blindly would silently destroy it.
    for path in ("ddi_training_dataset_final.csv", "training_matrix_augmented.csv"):
        if os.path.exists(path):
            raise SystemExit(
                f"❌ Refusing to overwrite existing {path}. "
                f"If this is real data, leave it alone. If this is a stale "
                f"synthetic fixture, `rm` it manually first."
            )

    rng = np.random.default_rng(42)
    rows = make_pairs(rng, n_pairs=60)
    targets = synth_targets(rng, len(rows))

    xshape = write_xgboost_fixture(rows, targets, "ddi_training_dataset_final.csv")
    print(f"📁 Wrote ddi_training_dataset_final.csv: shape={xshape}")

    gshape = write_gnn_fixture(rows, targets, "training_matrix_augmented.csv")
    print(f"📁 Wrote training_matrix_augmented.csv: shape={gshape}")


if __name__ == "__main__":
    main()
