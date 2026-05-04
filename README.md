# Drophet

Quantitative Risk Analysis (QRA) predictive model for forecasting toxicological
effects and incidence rate distributions of drug–drug interactions (DDIs).

> Status: research prototype. **Not** approved for clinical use.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# scispacy biomedical NER model (used by 003b)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

Create a `.env` at the repo root with whatever LLM keys you need:

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

`.env` is gitignored; never commit it.

---

## Pipeline overview

The codebase is a sequence of numbered scripts. Each one reads files produced
by an earlier step and writes new ones. Inputs/outputs:

### Phase 1 — Trial ingestion & adverse-event categorization (`PharIntrvtOm-main/PharIntrvtOm-main/`)

| Step | Script | In | Out |
|------|--------|----|-----|
| 001 | `001_gpt-api.py` | `ctg-filtered-drug-trials.json` (path via `DROPHET_TRIALS_JSON`) | filtered trials JSON |
| 002 | `002_make-adverse-event_terms_list-list-0.01.py` | trials JSON | `adverse_event_terms_list.json` |
| 003 / 003b | `003_*` / `003b_gemini_categorize_events.py` | adverse-event terms | categorized AE JSON |
| 004 | `004_transcribe.py` | categorized AE JSON | transcribed AE JSON |
| 005 | `005_show_json_structure.py` | any JSON | (utility — prints structure) |
| 006 | `006_*` | categorized AE JSON | severity-weighted AE JSON |
| 007 | `007_extract_SMILES.py` | filtered trials JSON | trials w/ SMILES |
| 008 | `008_filter_eventGroups.py` | trials w/ AE groups | filtered eventGroups |
| 009 | `009_stepwise_filter_eventGroups.py` | eventGroups | refined eventGroups |
| 010 | `010_compare-nctid-s.py` | two JSONs | overlap report |
| 011 | `011_stepwise_filter_eventGroups-add-drug-names.py` | eventGroups | + drug names |
| 012 | `012_stepwise_filter_eventGroups-conditions-ICD.py` | + ICD codes | `012_gpt_added_ICD.json` |
| 013 | `013_get-a-special-set-of-trials.py` | filtered trials | special-set trials |
| 014 | `014_merge_pipeline_to_matrix.py` | trial+AE+drug JSONs | `training_matrix.csv` |
| 015 | `015_data_quality_check.py` | `training_matrix.csv` | `training_matrix_cleaned.csv` |
| 016 | `016_feature_engineering.py` | `training_matrix_cleaned.csv` | `ddi_training_dataset_final.csv` (Morgan bits + descriptors) |

### Phase 2 — Modeling (repo root)

| Step | Script | In | Out |
|------|--------|----|-----|
| 017 | `017_baseline_xgboost.py` | `ddi_training_dataset_final.csv` | `clinical_risk_predictions.csv` (XGBoost baseline + SHAP) |
| 018 | `018_peptide_filter_and_data_refinement.py` | `training_matrix_cleaned.csv` | `training_matrix_refined_for_gnn.csv` (peptides removed) |
| 020 | `020_gnn_pretraining.py` | `training_matrix_refined_for_gnn.csv` (SMILES set) | `gnn_pretrained_backbone.pth` |
| 021 | `021_generate_negative_samples.py` | `training_matrix_refined_for_gnn.csv` | augmented matrix (overwrites in place; backup kept as `*_backup.csv`) |
| 019 | `019_train_gnn_model.py` | `training_matrix_augmented.csv` ↷ falls back to `training_matrix_refined_for_gnn.csv`; `gnn_pretrained_backbone.pth` | `ddi_gnn_best_model.pth`, interactive CLI |

> The script numbering 017→018→020→021→019 reflects how the project was
> developed, not the run order. Run order is: **017 → 018 → 020 → 021 → 019**.

### Frontend

`index.html` is a single-page UI demo. The Gemini API key must be supplied at
runtime via `window.GEMINI_API_KEY` (see comment in the file). For production,
this should call a backend proxy that holds the key as a server-side secret
instead of hitting the Gemini API directly from the browser.

---

## Reproducibility

All training scripts call `drophet_utils.seed_everything(42)` at import time
and use a `GroupShuffleSplit` keyed on a canonical (sorted) drug-pair key,
so the same drug pair never appears in both train and test.

---

## Known gaps

- No unit tests, no CI.
- Hyperparameters are hand-tuned (XGBoost) or fixed (GNN); no Optuna/grid search.
- Negative samples in `021_*` are 30 hand-picked "obviously safe" pairs, not
  literature-vetted controls.
- Pretraining → finetuning weight loading uses `strict=False`; architecture
  drift can pass silently.
- Output is a point estimate with no uncertainty quantification — unsuitable
  for clinical decision support as-is.

---

## License

See `LICENSE`.
