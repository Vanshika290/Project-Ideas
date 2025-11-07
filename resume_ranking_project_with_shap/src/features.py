import pandas as pd
import numpy as np

SKILL_VOCAB = ['python','java','react','aws','ml','html','css','js']

def canonicalize_skills(skills_cell):
    # Accepts comma or space separated skills; returns list of cleaned tokens
    if pd.isna(skills_cell):
        return []
    # ensure string
    s = str(skills_cell)
    # replace common separators with comma, lowercase, split
    for ch in [';', '|', '/']:
        s = s.replace(ch, ',')
    toks = [t.strip().lower() for t in s.replace(',', ' ').split() if t.strip()]
    return toks

def compute_features(df, required_skills=None):
    df = df.copy()
    # canonicalize skills
    df['skill_tokens'] = df['skills'].apply(canonicalize_skills)

    if required_skills is None:
        required_skills = []

    def count_matched(tokens):
        if not required_skills:
            return 0
        return sum(1 for t in tokens if t in required_skills)

    df['num_skills_total'] = df['skill_tokens'].apply(len)
    df['num_skills_matched'] = df['skill_tokens'].apply(lambda toks: count_matched(toks))
    df['pct_skills_matched'] = df.apply(
        lambda r: (r['num_skills_matched'] / len(required_skills)) if required_skills else 0.0, axis=1
    )

    # numeric features
    df['years_experience'] = pd.to_numeric(df['years_experience'], errors='coerce').fillna(0)
    df['months_gap'] = pd.to_numeric(df.get('months_gap', 0), errors='coerce').fillna(0)

    # education_level: map text to ordinal (HighSchool=0, Bachelors=1, Masters=2, PhD=3)
    edu_map = {'HighSchool': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
    df['education_level_num'] = df['education_level'].map(edu_map).fillna(0).astype(int)

    # basic derived features
    df['experience_score'] = df['years_experience'].clip(0,30) / 30.0
    df['gap_penalty'] = df['months_gap'].apply(lambda x: min(x,24)/24.0)

    # final feature set
    feature_cols = ['years_experience','num_skills_total','num_skills_matched',
                    'pct_skills_matched','education_level_num','experience_score','gap_penalty']
    return df, feature_cols
