# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 4: Interpretability & Risk Calibration

This script extends the tuned XGBoost model with:
1. SHAP Analysis: Identifying which molecular bits drive the toxicity.
2. Clinical Risk Stratification: Mapping % incidence to Low/Med/High risk.
3. Residual Profiling: Identifying drug pairs with the highest prediction errors.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import os

from drophet_utils import seed_everything

seed_everything(42)

# Robust Matplotlib/SHAP Import
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ Warning: 'shap' module not found. Use 'pip install shap' for explainability analysis.")

def categorize_risk(prob):
    """Maps continuous incidence rate to clinical risk tiers."""
    if prob < 5: return "Low Risk"
    if prob < 20: return "Moderate Risk"
    return "High Risk"

def main():
    print("--- Phase 4: Model Interpretability & Clinical Calibration ---")
    input_file = 'ddi_training_dataset_final.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run 016 first.")
        return

    # 1. Load Data
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Setup Features and Target
    feature_cols = [col for col in df.columns if "_Bit_" in col or any(x in col for x in ["MW", "LogP", "TPSA"])]
    X_raw = df[feature_cols]

    target_cols = [col for col in df.columns if col.startswith('Target_')]
    target_stats = df[target_cols].mean().sort_values(ascending=False)
    target_name = target_stats.index[0] # Focus on the primary signal
    y = df[target_name]

    print(f"🎯 Target: {target_name} (Mean: {target_stats.iloc[0]:.2f}%)")

    # 3. Preprocessing (Mirroring the tuned settings)
    selector = VarianceThreshold(threshold=0.01)
    X_reduced = pd.DataFrame(selector.fit_transform(X_raw), columns=X_raw.columns[selector.get_support()])

    pre_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    pre_model.fit(X_reduced, y)
    sfm = SelectFromModel(pre_model, threshold="0.8*mean", prefit=True)
    X_selected = X_reduced.loc[:, sfm.get_support()]
    final_features = X_selected.columns.tolist()

    # 4. Final Model with Optimized Hyperparameters (from your last run)
    best_params = {
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'subsample': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    model = xgb.XGBRegressor(**best_params)

    # 5. Cross-Validation & Error Analysis
    # SHAP values are accumulated PER FOLD on each fold's held-out test rows so
    # that the explanation matrix matches the OOF predictions. Refitting on the
    # full dataset for global SHAP (the prior approach) leaks every row into
    # the explainer, making the SHAP attributions inconsistent with the
    # CV metrics reported below.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_actuals, all_metadata = [], [], []
    fold_shap_values = []

    for train_idx, test_idx in kf.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = np.maximum(model.predict(X_test), 0)

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_metadata.extend(df.iloc[test_idx][['Drug_1', 'Drug_2']].to_dict('records'))

        if HAS_SHAP:
            explainer = shap.TreeExplainer(model)
            fold_shap_values.append(explainer.shap_values(X_test))

    # Build Analysis DataFrame
    analysis_df = pd.DataFrame(all_metadata)
    analysis_df['Actual_%'] = all_actuals
    analysis_df['Predicted_%'] = all_preds
    analysis_df['Error'] = abs(analysis_df['Actual_%'] - analysis_df['Predicted_%'])
    analysis_df['Risk_Level'] = analysis_df['Predicted_%'].apply(categorize_risk)

    print("\n" + "="*40)
    print("📊 CLINICAL PREDICTION EXAMPLES (Top 5)")
    print("="*40)
    print(analysis_df.sort_values('Error').head(5).to_string(index=False))
    print("="*40)

    # 6. SHAP Explainability (XAI) — aggregated across out-of-fold predictions
    if HAS_SHAP and fold_shap_values:
        print("\n🔍 Aggregating out-of-fold SHAP attributions...")
        oof_shap = np.vstack(fold_shap_values)

        shap_importance = np.abs(oof_shap).mean(axis=0)
        feature_importance = pd.DataFrame({'feature': final_features, 'importance': shap_importance})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        print("\n🧪 Top 10 Structural Drivers (SHAP Impact):")
        for _, row in feature_importance.head(10).iterrows():
            col_idx = final_features.index(row['feature'])
            direction = "Toxicity ↑" if np.mean(oof_shap[:, col_idx]) > 0 else "Toxicity ↓"
            print(f"   - {row['feature']}: {row['importance']:.4f} ({direction})")

    # 7. Identify High-Error Outliers (Data Debugging)
    print("\n⚠️ High Error Outliers (Possible clinical noise):")
    outliers = analysis_df.sort_values('Error', ascending=False).head(3)
    for _, row in outliers.iterrows():
        print(f"   - {row['Drug_1']} + {row['Drug_2']}: Actual {row['Actual_%']:.1f}% vs Pred {row['Predicted_%']:.1f}%")

    # 8. Persistence
    analysis_df.to_csv('clinical_risk_predictions.csv', index=False)
    print(f"\n✅ Analysis complete. Risk predictions saved to 'clinical_risk_predictions.csv'.")

if __name__ == "__main__":
    main()
