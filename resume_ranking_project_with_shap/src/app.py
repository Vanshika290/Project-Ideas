from flask import Flask, render_template, send_file, abort, request, redirect, url_for
import pandas as pd
import os
from explain import compute_shap_values, top_features_for_row, load_model
import joblib
import io
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for PNG generation
import matplotlib.pyplot as plt

app = Flask(__name__)

DATA_CSV = 'data/predictions.csv'
MODEL_JOBLIB = 'models/lgb_model.txt.joblib'

def load_predictions():
    if not os.path.exists(DATA_CSV):
        return pd.DataFrame()
    return pd.read_csv(DATA_CSV)

@app.route('/')
def index():
    df = load_predictions()
    # show top 50 by score in UI
    if df.empty:
        message = 'No predictions found. Run src/main.py to generate predictions first.'
        return render_template('index.html', df=None, message=message)
    df_sorted = df.sort_values('score', ascending=False).head(50)
    return render_template('index.html', df=df_sorted.to_dict(orient='records'), message=None)

@app.route('/candidate/<candidate_id>')
def candidate(candidate_id):
    df = load_predictions()
    if df.empty:
        abort(404, 'Predictions not found. Run training and prediction.')
    df = df.reset_index(drop=True)
    row = df[df['candidate_id'] == candidate_id]
    if row.empty:
        abort(404, 'Candidate not found')
    row_idx = int(row.index[0])
    # load model and compute shap
    model = load_model(MODEL_JOBLIB)
    # load features from main pipeline by reusing columns present in CSV
    # assume feature columns are those not in a small ignore list
    ignore = set(['candidate_id','skills','education_level','last_company','location','label','score'])
    feature_cols = [c for c in df.columns if c not in ignore]
    shap_vals, feat_cols = compute_shap_values(df, feature_cols, model=model)
    top = top_features_for_row(shap_vals, feat_cols, row_idx, top_k=7)
    # also generate a PNG bar chart for top features
    figbuf = io.BytesIO()
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    plt.figure(figsize=(6,3))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.xlabel('SHAP value (impact on model output)')
    plt.tight_layout()
    plt.savefig(figbuf, format='png')
    plt.close()
    figbuf.seek(0)
    img_b64 = 'data:image/png;base64,' + (figbuf.getvalue()).encode('base64') if False else None
    # we will instead serve the image via a dedicated route that recomputes it
    return render_template('candidate.html', candidate=row.to_dict(orient='records')[0], top=top)

@app.route('/candidate_plot/<candidate_id>')
def candidate_plot(candidate_id):
    df = load_predictions()
    if df.empty:
        abort(404)
    df = df.reset_index(drop=True)
    row = df[df['candidate_id'] == candidate_id]
    if row.empty:
        abort(404)
    row_idx = int(row.index[0])
    model = load_model(MODEL_JOBLIB)
    ignore = set(['candidate_id','skills','education_level','last_company','location','label','score'])
    feature_cols = [c for c in df.columns if c not in ignore]
    shap_vals, feat_cols = compute_shap_values(df, feature_cols, model=model)
    top = top_features_for_row(shap_vals, feat_cols, row_idx, top_k=7)
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    import io
    buf = io.BytesIO()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.xlabel('SHAP value (impact on model output)')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # debug mode only
    app.run(host='0.0.0.0', port=5000, debug=True)
