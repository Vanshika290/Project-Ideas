# Resume Ranking with AI — Starter Project
**What this is:** A minimal, runnable starter project that implements a structured resume ranking pipeline using feature engineering and LightGBM.

**What's included**
- `data/sample_candidates.csv` — small example dataset
- `requirements.txt`
- `src/`:
  - `main.py` — end-to-end run: feature generation, train, predict
  - `features.py` — feature engineering functions
  - `train.py` — model training & evaluation
  - `predict.py` — scoring & ranking function
  - `utils.py` — helper utilities (I/O, canonicalization)
- `models/` — saved model will be written here when training
- `resume_ranking_project.zip` — this archive (created for download)

**Quick start**
1. Create and activate a Python environment (Python 3.8+).
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run end-to-end:
   ```
   python src/main.py
   ```
4. The script trains a LightGBM model on `data/sample_candidates.csv` and writes `models/lgb_model.txt`.
   It also saves predictions to `data/predictions.csv`.

**Notes**
- This project uses a *structured-input* approach: `skills` are stored as comma-separated canonical tokens.
- It's a starter: replace sample CSV with your real data (same columns) and tune features / model.
