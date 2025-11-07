import joblib
import shap
import pandas as pd
import numpy as np
import os

MODEL_JOBLIB = 'models/lgb_model.txt.joblib'

def load_model(model_path=None):
    model_path = model_path or MODEL_JOBLIB
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    return joblib.load(model_path)

def compute_shap_values(df, feature_cols, model=None):
    model = model or load_model()
    # LightGBM model wrapper from joblib is the original booster object from lgb.train;
    # shap.TreeExplainer works with Booster objects
    # If joblib saved the Booster, it will be loaded here.
    X = df[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # For binary classification, shap_values is a list [neg, pos]; take pos class (index 1) if present
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = np.array(shap_values[1])
    else:
        shap_vals = np.array(shap_values)
    return shap_vals, feature_cols

def top_features_for_row(shap_vals, feature_cols, row_idx, top_k=5):
    row_shap = shap_vals[row_idx]
    # absolute importance
    idx_sorted = np.argsort(-np.abs(row_shap))[:top_k]
    return [(feature_cols[i], float(row_shap[i])) for i in idx_sorted]
