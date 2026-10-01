"""
Train a salary prediction model and save it as model.pkl.
Run this once before launching the Streamlit app:
    py -3 train_model.py
"""

import os
import re
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Load data ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Salary_Data.csv")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

# ── 2. Clean / normalise ─────────────────────────────────────────────────────
df = df.dropna(subset=["Salary"])
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df = df.dropna(subset=["Salary"])
df = df[df["Salary"] > 1000]  # remove obvious data-entry errors

# Normalise education level labels
edu_map = {
    "bachelor's degree": "Bachelor's",
    "master's degree":   "Master's",
    "phd":               "PhD",
}
df["Education Level"] = (
    df["Education Level"]
    .fillna("")
    .str.strip()
    .apply(lambda x: edu_map.get(x.lower(), x) if x else x)
)

# Drop rows still missing features
FEATURES = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
df = df.dropna(subset=FEATURES)

# ── 3. Encode ─────────────────────────────────────────────────────────────────
edu_order = ["High School", "Bachelor's", "Master's", "PhD"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "edu_ord",
            OrdinalEncoder(
                categories=[edu_order],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ["Education Level"],
        ),
        (
            "gender_ord",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ["Gender"],
        ),
        (
            "job_ord",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ["Job Title"],
        ),
        ("num_scaler", StandardScaler(), ["Age", "Years of Experience"]),
    ],
    remainder="drop",
)

# ── 4. Pipeline ───────────────────────────────────────────────────────────────
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ]
)

X = df[FEATURES]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE : ${mae:,.0f}")
print(f"R²  : {r2:.4f}")

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
# Save unique sorted values for UI dropdowns
meta = {
    "job_titles":        sorted(df["Job Title"].unique().tolist()),
    "education_levels":  edu_order,
    "genders":           sorted(df["Gender"].unique().tolist()),
    "age_min":           int(df["Age"].min()),
    "age_max":           int(df["Age"].max()),
    "exp_min":           int(df["Years of Experience"].min()),
    "exp_max":           int(df["Years of Experience"].max()),
    "mae":               round(mae, 2),
    "r2":                round(r2, 4),
}

out_path = os.path.join(BASE_DIR, "model.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"model": model, "meta": meta}, f)

print(f"Model saved -> {out_path}")
