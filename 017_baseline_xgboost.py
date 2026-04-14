# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Phase 3: Model Training (Baseline)

This script trains an XGBoost Regressor using the engineered molecular features
to predict the incidence rate (%) of a specific Adverse Event category.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def main():
    print("--- Phase 3: Baseline XGBoost Model Training ---")
    input_file = 'ddi_training_dataset_final.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run 016 first.")
        return

    # 1. Load Data
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Define Features (X) and Target (y)
    # Extract all feature columns starting with D1_ and D2_
    feature_cols = [col for col in df.columns if col.startswith('D1_') or col.startswith('D2_')]
    X = df[feature_cols]
    
    # Find all Target columns
    target_cols = [col for col in df.columns if col.startswith('Target_AE_')]
    if not target_cols:
        print("❌ Error: No Target variables found.")
        return
        
    print(f"\nAvailable Targets: {target_cols}")
    
    # Select the first Target for baseline testing (can be modified later)
    target_name = target_cols[0] 
    y = df[target_name]
    
    print(f"\n🎯 Training model for Target: {target_name}")

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. Initialize and Train XGBoost Regressor
    # Using conservative hyperparameters to prevent overfitting on the small dataset
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
    
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    # 5. Predictions & Evaluation
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n✅ Model Evaluation Results (Test Set):")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}%")
    print(f"R-squared (R2): {r2:.4f}")
    
    # 6. Feature Importance Preview
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    print("\n🔍 Top 5 Most Important Features:")
    for idx in top_indices:
        print(f"{feature_cols[idx]}: {importances[idx]:.4f}")

if __name__ == "__main__":
    main()