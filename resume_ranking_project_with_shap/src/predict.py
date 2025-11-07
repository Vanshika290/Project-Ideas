import pandas as pd
import joblib
import os

def load_model(model_path='models/lgb_model.txt.joblib'):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    # try other extension
    alt = model_path.replace('.joblib','')
    if os.path.exists(alt):
        return joblib.load(alt + '.joblib')
    raise FileNotFoundError(model_path)

def score_and_rank(df, feature_cols, model):
    X = df[feature_cols]
    df['score'] = model.predict(X)
    df_sorted = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df_sorted
