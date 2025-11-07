import os
from utils import read_csv, ensure_dir
from features import compute_features
from train import train_model
from predict import score_and_rank, load_model
import joblib

DATA_PATH = 'data/sample_candidates.csv'
MODEL_PATH = 'models/lgb_model.txt'

def run_end_to_end(required_skills=None):
    ensure_dir('models')
    df = read_csv(DATA_PATH)
    # compute features
    df_feat, feature_cols = compute_features(df, required_skills=required_skills)
    # train
    model = train_model(df_feat, feature_cols, label_col='label', model_path=MODEL_PATH)
    # also save a joblib wrapper
    joblib.dump(model, MODEL_PATH + '.joblib')
    # predict & rank
    ranked = score_and_rank(df_feat, feature_cols, model)
    ranked.to_csv('data/predictions.csv', index=False)
    print('Saved predictions to data/predictions.csv')
    print(ranked[['candidate_id','score']].head(10))

if __name__ == '__main__':
    # Example: require python and aws for the role
    required_skills = ['python','aws']
    run_end_to_end(required_skills=required_skills)
