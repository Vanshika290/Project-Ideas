import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

def train_model(df, feature_cols, label_col='label', group_col=None, model_path='models/lgb_model.txt'):
    X = df[feature_cols]
    y = df[label_col].astype(int)

    # simple train/test split (no leakage): if group_col provided, use GroupKFold
    if group_col and group_col in df.columns:
        gkf = GroupKFold(n_splits=3)
        groups = df[group_col]
        # take first split as simple example
        train_idx, val_idx = next(gkf.split(X, y, groups))
    else:
        # fallback random split
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42, stratify=y)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42
    }

    model = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_val], early_stopping_rounds=20, verbose_eval=False)

    # save model (text)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    # save lightgbm as joblib for convenience
    joblib.dump(model, model_path + '.joblib')

    # validation metric
    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds) if len(set(y_val))>1 else 0.5
    print(f"Validation AUC: {auc:.4f}")
    return model
