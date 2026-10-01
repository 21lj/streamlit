"""
Salary Prediction App — Streamlit UI
Run: streamlit run salary_prediction/app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
)

# ── Load model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artifact = load_model()
model = artifact["model"]
meta = artifact["meta"]

# ── Load raw data for charts ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "dataset", "Salary_Data.csv")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Salary"])
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary"])
    df = df[df["Salary"] > 1000]
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
    return df


df = load_data()

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 1.8rem; color: #1f2328; }
    .metric-card p  { margin: 0; font-size: 0.85rem; color: #57606a; }
    .predict-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
        color: white;
    }
    .predict-box h1 { font-size: 3rem; margin: 0; }
    .predict-box p  { margin: 4px 0 0; font-size: 1rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💼 Salary Prediction App")
st.markdown(
    "Predict your expected salary based on your profile. "
    "Trained on **{:,} records** with **R² = {:.2f}** and **MAE ≈ ${:,.0f}**.".format(
        len(df), meta["r2"], meta["mae"]
    )
)
st.divider()

# ── Layout ─────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

# ═══ INPUT PANEL ══════════════════════════════════════════════════════════════
with left_col:
    st.subheader("Your Profile")

    age = st.slider(
        "Age",
        min_value=meta["age_min"],
        max_value=meta["age_max"],
        value=28,
        step=1,
    )

    gender = st.selectbox("Gender", options=meta["genders"])

    education = st.selectbox("Education Level", options=meta["education_levels"])

    job_title = st.selectbox("Job Title", options=meta["job_titles"])

    experience = st.slider(
        "Years of Experience",
        min_value=meta["exp_min"],
        max_value=meta["exp_max"],
        value=3,
        step=1,
    )

    predict_btn = st.button("Predict Salary", type="primary", use_container_width=True)

# ═══ OUTPUT PANEL ═════════════════════════════════════════════════════════════
with right_col:
    # Prediction result
    if predict_btn:
        input_df = pd.DataFrame(
            [{
                "Age": age,
                "Gender": gender,
                "Education Level": education,
                "Job Title": job_title,
                "Years of Experience": experience,
            }]
        )
        predicted = model.predict(input_df)[0]
        st.markdown(
            f"""
            <div class="predict-box">
                <p>Estimated Annual Salary</p>
                <h1>${predicted:,.0f}</h1>
                <p>± ${meta['mae']:,.0f} model error</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Profile: {age} yrs old · {gender} · {education} · {job_title} · {experience} yrs exp.")

        # Gauge chart
        sal_min, sal_max = df["Salary"].min(), df["Salary"].max()
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted,
                number={"prefix": "$", "valueformat": ",.0f"},
                title={"text": "Predicted vs Dataset Range"},
                gauge={
                    "axis": {"range": [sal_min, sal_max]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [sal_min, np.percentile(df["Salary"], 25)], "color": "#fee2e2"},
                        {"range": [np.percentile(df["Salary"], 25), np.percentile(df["Salary"], 75)], "color": "#dbeafe"},
                        {"range": [np.percentile(df["Salary"], 75), sal_max], "color": "#d1fae5"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": predicted,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.info("Fill in your profile on the left and click **Predict Salary**.")

# ── Dataset Insights ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Dataset Insights")

tab1, tab2, tab3 = st.tabs(["Salary Distribution", "By Education", "Top Job Titles"])

with tab1:
    fig_hist = px.histogram(
        df, x="Salary", nbins=60,
        title="Salary Distribution across all employees",
        labels={"Salary": "Annual Salary ($)"},
        color_discrete_sequence=["#2563eb"],
    )
    fig_hist.update_layout(bargap=0.05, height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    edu_df = df[df["Education Level"].isin(edu_order)]
    fig_box = px.box(
        edu_df,
        x="Education Level",
        y="Salary",
        color="Education Level",
        category_orders={"Education Level": edu_order},
        title="Salary by Education Level",
        labels={"Salary": "Annual Salary ($)"},
    )
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    top_jobs = (
        df.groupby("Job Title")["Salary"]
        .median()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    fig_bar = px.bar(
        top_jobs,
        x="Salary",
        y="Job Title",
        orientation="h",
        title="Top 15 Job Titles by Median Salary",
        labels={"Salary": "Median Annual Salary ($)", "Job Title": ""},
        color="Salary",
        color_continuous_scale="Blues",
    )
    fig_bar.update_layout(height=480, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Gradient Boosting Regressor | Dataset: 6,699 records | Features: Age, Gender, Education, Job Title, Experience")
