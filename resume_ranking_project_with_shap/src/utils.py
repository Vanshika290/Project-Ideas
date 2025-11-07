import pandas as pd
import os

def read_csv(path):
    return pd.read_csv(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
