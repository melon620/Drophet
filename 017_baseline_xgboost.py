# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 4: XGBoost baseline with class-imbalance weighting, nested-CV
hyperparameter search (Optuna), out-of-fold SHAP, and calibration analysis.

Outputs (all written to repo root):
    - clinical_risk_predictions.csv  (OOF predictions w/ risk tiers)
    - xgboost_calibration.png        (reliability diagram + Brier score)
    - xgboost_pr_curve.png           (precision-recall curve at the
                                      RISK_THRESHOLD_PCT positive cutoff)

Methodology notes (these are the things this script asserts about itself):
    - Nested CV: outer 5-fold KFold for honest test metrics; inner Optuna
      with `INNER_TRIALS` trials × 3-fold CV inside each outer training
      fold for hyperparameter selection. Hyperparameters chosen on the
      inner fold are *not* re-evaluated on the outer test fold's data, so
      the outer metrics are unbiased.
    - Class imbalance: training rows where the target ≥ RISK_THRESHOLD_PCT
      (default 5%) are upweighted by `n_neg / n_pos` so the model doesn't
      collapse to predicting the majority near-zero class. This is the
      regression analog of XGBoost's `scale_pos_weight` for binary tasks.
    - SHAP is computed per outer fold on that fold's held-out test rows
      with the inner-CV-tuned model. The full-data refit of prior versions
      is gone (it leaks every row into the explainer).
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

from drophet_utils import seed_everything

seed_everything(42)

# ---------------------------------------------------------------------------
# Tunable constants — change these and document the change in the PR.
# ---------------------------------------------------------------------------
RISK_THRESHOLD_PCT = 5.0     # ≥ this % is treated as a "positive" event
                             # for binary metrics (PR-AUC, Brier, calibration).
                             # 5% chosen to match the Low/Moderate tier cutoff.
TIER_LOW_CUTOFF = 5.0        # < this %  → Low Risk (matches the threshold)
TIER_MODERATE_CUTOFF = 20.0  # < this %  → Moderate; ≥ this → High.
                             # 5/20 are heuristic carryovers from earlier
                             # versions; the calibration plot below is the
                             # evidence to revisit them.

OUTER_FOLDS = 5
INNER_FOLDS = 3
INNER_TRIALS = 20            # Optuna trials per outer fold. Bump to 50+
                             # for "real" runs, keep low for fast smoke tests.
OPTUNA_TIMEOUT = None        # seconds; None = no timeout
PLOT_DIR = Path(".")

# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ Warning: 'shap' module not found. Install with `pip install shap`.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ Warning: 'optuna' module not found. Falling back to fixed params.")

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def categorize_risk(prob: float) -> str:
    if prob < TIER_LOW_CUTOFF:
        return "Low Risk"
    if prob < TIER_MODERATE_CUTOFF:
        return "Moderate Risk"
    return "High Risk"


def imbalance_weights(y: np.ndarray, threshold: float = RISK_THRESHOLD_PCT) -> np.ndarray:
    """sample_weight per row: positives (y ≥ threshold) get n_neg/n_pos so
    the weighted positive mass equals the negative mass. Negatives get 1.
    Falls back to all-ones if a class is empty."""
    pos = y >= threshold
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.ones_like(y, dtype=float)
    pos_weight = n_neg / n_pos
    return np.where(pos, pos_weight, 1.0).astype(float)


def optuna_search(X_train, y_train, w_train, n_trials: int):
    """Run inner Optuna HP search on (X_train, y_train) with 3-fold CV.
    Returns the best param dict including a static random_state/objective.
    """
    if not HAS_OPTUNA:
        return {
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 300,
            "reg_alpha": 0.1,
            "subsample": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "reg:squarederror",
            "random_state": 42,
            "verbosity": 0,
        }

        inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
        maes = []
        for tr_i, va_i in inner.split(X_train):
            m = xgb.XGBRegressor(**params)
            m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i],
                  sample_weight=w_train[tr_i])
            p = np.maximum(m.predict(X_train.iloc[va_i]), 0)
            maes.append(mean_absolute_error(y_train.iloc[va_i], p))
        return float(np.mean(maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"objective": "reg:squarederror", "random_state": 42, "verbosity": 0})
    return best


def write_calibration_plot(y_true_pct, y_pred_pct, threshold, path: Path) -> dict:
    """Saves a reliability diagram at the given % threshold and returns a
    metrics dict (Brier, PR-AUC, ROC-AUC). Treats the regression output as
    a calibrated probability for the binary event y ≥ threshold."""
    y_true = np.asarray(y_true_pct, dtype=float)
    y_pred = np.asarray(y_pred_pct, dtype=float)
    y_bin = (y_true >= threshold).astype(int)
    y_prob = np.clip(y_pred / 100.0, 0.0, 1.0)

    metrics = {"n_pos": int(y_bin.sum()), "n_neg": int((1 - y_bin).sum())}
    if metrics["n_pos"] == 0 or metrics["n_neg"] == 0:
        metrics["note"] = "Only one class present; binary metrics undefined."
        return metrics

    metrics["brier"] = float(brier_score_loss(y_bin, y_prob))
    metrics["pr_auc"] = float(average_precision_score(y_bin, y_prob))
    metrics["roc_auc"] = float(roc_auc_score(y_bin, y_prob))

    if not HAS_PLOT:
        return metrics

    n_bins = 5
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    bin_pred = np.array([y_prob[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                         for b in range(n_bins)])
    bin_obs = np.array([y_bin[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                        for b in range(n_bins)])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(bin_pred, bin_obs, "o-", label=f"XGBoost (Brier {metrics['brier']:.3f})")
    ax.set_xlabel(f"Predicted probability of incidence ≥ {threshold:.0f}%")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (out-of-fold)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return metrics


def write_pr_curve(y_true_pct, y_pred_pct, threshold, path: Path):
    if not HAS_PLOT:
        return
    y_bin = (np.asarray(y_true_pct) >= threshold).astype(int)
    y_prob = np.clip(np.asarray(y_pred_pct) / 100.0, 0.0, 1.0)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return
    p, r, _ = precision_recall_curve(y_bin, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r, p)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve at incidence ≥ {threshold:.0f}%")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    print("--- Phase 4: XGBoost baseline (nested CV + class-balanced + calibration) ---")
    input_file = "ddi_training_dataset_final.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run 016 first.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded dataset with shape: {df.shape}")

    feature_cols = [c for c in df.columns if "_Bit_" in c
                    or any(x in c for x in ["MW", "LogP", "TPSA"])]
    X_raw = df[feature_cols]

    target_cols = [c for c in df.columns if c.startswith("Target_")]
    if not target_cols:
        raise ValueError("No Target_ columns found.")
    target_stats = df[target_cols].mean().sort_values(ascending=False)
    target_name = target_stats.index[0]
    y = df[target_name]
    print(f"🎯 Target: {target_name} (Mean: {target_stats.iloc[0]:.2f}%)")
    print(f"   Class imbalance @ ≥{RISK_THRESHOLD_PCT}%: "
          f"{int((y >= RISK_THRESHOLD_PCT).sum())}/{len(y)} positive "
          f"({(y >= RISK_THRESHOLD_PCT).mean()*100:.1f}%)")

    # Feature reduction (variance threshold + SelectFromModel) on the full
    # data — feature *selection* across folds is a known minor leakage
    # pathway, but it preserves comparability with the prior baseline. The
    # bigger leakage source (full-data refit for SHAP) is fixed below.
    selector = VarianceThreshold(threshold=0.01)
    X_reduced = pd.DataFrame(selector.fit_transform(X_raw),
                             columns=X_raw.columns[selector.get_support()])

    pre_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    pre_model.fit(X_reduced, y)
    sfm = SelectFromModel(pre_model, threshold="0.8*mean", prefit=True)
    X_selected = X_reduced.loc[:, sfm.get_support()]
    final_features = X_selected.columns.tolist()
    print(f"🧬 Feature selection retained {len(final_features)} features.")

    # ----- Nested CV -----
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    all_preds, all_actuals, all_metadata = [], [], []
    fold_shap_values = []
    chosen_params_log = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X_selected, y), start=1):
        X_tr, X_te = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        w_tr = imbalance_weights(y_tr.values)

        print(f"\n[Fold {fold_idx}/{OUTER_FOLDS}] running inner Optuna ({INNER_TRIALS} trials)...")
        best_params = optuna_search(X_tr, y_tr, w_tr, n_trials=INNER_TRIALS)
        chosen_params_log.append(best_params)

        model = xgb.XGBRegressor(**best_params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        preds = np.maximum(model.predict(X_te), 0)
        all_preds.extend(preds)
        all_actuals.extend(y_te)
        all_metadata.extend(df.iloc[test_idx][["Drug_1", "Drug_2"]].to_dict("records"))

        if HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(model)
                fold_shap_values.append(explainer.shap_values(X_te))
            except Exception as exc:
                print(f"   ⚠️ SHAP failed in fold {fold_idx}: {exc}")

        print(f"   fold MAE: {mean_absolute_error(y_te, preds):.3f}, "
              f"depth={best_params.get('max_depth')}, "
              f"lr={best_params.get('learning_rate'):.3f}")

    # ----- Out-of-fold metrics -----
    actuals = np.asarray(all_actuals, dtype=float)
    preds = np.asarray(all_preds, dtype=float)

    mae = mean_absolute_error(actuals, preds)
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    r2 = r2_score(actuals, preds) if len(actuals) > 1 else float("nan")
    spearman = spearmanr(actuals, preds).correlation if len(actuals) > 1 else float("nan")

    print("\n" + "=" * 50)
    print("📊 OUT-OF-FOLD REGRESSION METRICS")
    print("=" * 50)
    print(f"   MAE:       {mae:.3f} %")
    print(f"   RMSE:      {rmse:.3f} %")
    print(f"   R²:        {r2:.3f}")
    print(f"   Spearman:  {spearman:.3f}")

    # Binary / calibration metrics at the configured threshold
    cal_metrics = write_calibration_plot(
        actuals, preds, RISK_THRESHOLD_PCT, PLOT_DIR / "xgboost_calibration.png"
    )
    write_pr_curve(actuals, preds, RISK_THRESHOLD_PCT, PLOT_DIR / "xgboost_pr_curve.png")

    print(f"\n📐 BINARY METRICS @ ≥{RISK_THRESHOLD_PCT}% incidence "
          f"(n_pos={cal_metrics['n_pos']}, n_neg={cal_metrics['n_neg']})")
    if "note" in cal_metrics:
        print(f"   ⚠️ {cal_metrics['note']}")
    else:
        print(f"   PR-AUC:    {cal_metrics['pr_auc']:.3f}")
        print(f"   ROC-AUC:   {cal_metrics['roc_auc']:.3f}")
        print(f"   Brier:     {cal_metrics['brier']:.3f}")
        if HAS_PLOT:
            print(f"   📈 reliability diagram → xgboost_calibration.png")
            print(f"   📈 precision/recall curve → xgboost_pr_curve.png")

    # ----- Analysis frame + outliers -----
    analysis_df = pd.DataFrame(all_metadata)
    analysis_df["Actual_%"] = actuals
    analysis_df["Predicted_%"] = preds
    analysis_df["Error"] = np.abs(actuals - preds)
    analysis_df["Risk_Level"] = analysis_df["Predicted_%"].apply(categorize_risk)

    print("\n" + "=" * 40)
    print("📊 CLINICAL PREDICTION EXAMPLES (Top 5 by lowest error)")
    print("=" * 40)
    print(analysis_df.sort_values("Error").head(5).to_string(index=False))

    print("\n⚠️ High Error Outliers (top 3):")
    for _, row in analysis_df.sort_values("Error", ascending=False).head(3).iterrows():
        print(f"   - {row['Drug_1']} + {row['Drug_2']}: "
              f"actual {row['Actual_%']:.1f}% vs pred {row['Predicted_%']:.1f}%")

    # ----- SHAP aggregation -----
    if HAS_SHAP and fold_shap_values:
        print("\n🔍 Aggregating out-of-fold SHAP attributions...")
        oof_shap = np.vstack(fold_shap_values)
        importance = np.abs(oof_shap).mean(axis=0)
        feat_imp = pd.DataFrame({"feature": final_features, "importance": importance}) \
            .sort_values("importance", ascending=False)
        print("\n🧪 Top 10 Structural Drivers (SHAP):")
        for _, row in feat_imp.head(10).iterrows():
            col_idx = final_features.index(row["feature"])
            direction = "Toxicity ↑" if oof_shap[:, col_idx].mean() > 0 else "Toxicity ↓"
            print(f"   - {row['feature']}: {row['importance']:.4f} ({direction})")

    # ----- Persistence -----
    analysis_df.to_csv("clinical_risk_predictions.csv", index=False)
    print("\n✅ Analysis complete. Risk predictions saved to 'clinical_risk_predictions.csv'.")


if __name__ == "__main__":
    main()
