"""
Drug Review Sentiment Analyzer - Streamlit application
======================================================
Loads the artifacts saved by the notebook (models/) and predicts sentiment.
It never retrains the model.

Usage (from the project root):
    streamlit run app.py
"""

import os
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse
import sklearn
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (app.py lives in the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

MODEL_PATH           = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH      = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')       # concat mode
COL_VECTORIZERS_PATH = os.path.join(MODELS_DIR, 'col_tfidf_vectorizers.pkl')  # union mode
METADATA_PATH        = os.path.join(MODELS_DIR, 'model_metadata.pkl')
TRAIN_PATH           = os.path.join(DATA_DIR, 'train.tsv')
TEST_PATH            = os.path.join(DATA_DIR, 'test.tsv')

REVIEW_COLS = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
FIELD_LABELS = {
    'benefitsReview':    'Benefits',
    'sideEffectsReview': 'Side effects',
    'commentsReview':    'Comments',
}
SENTIMENT_ORDER = ['Positive', 'Neutral', 'Negative']
SENTIMENT_COLORS = {'Positive': '#27ae60', 'Neutral': '#2980b9', 'Negative': '#e74c3c'}
SENTIMENT_EMOJI = {'Positive': '✅', 'Neutral': '🔵', 'Negative': '❌'}


# ---------------------------------------------------------------------------
# Text preprocessing - must stay identical to clean_text() in the notebook
# ---------------------------------------------------------------------------
def clean_text(text) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def rating_to_sentiment(rating):
    if pd.isna(rating):
        return None
    r = int(rating)
    if r <= 4:
        return 'Negative'
    if r <= 6:
        return 'Neutral'
    return 'Positive'


# ---------------------------------------------------------------------------
# Artifact + data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Return (model, vectorizer_or_dict, metadata, problems). problems is a list of str."""
    if not os.path.exists(MODEL_PATH):
        return None, None, {}, [f'Missing file: {MODEL_PATH}']
    try:
        metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
        strategy = metadata.get('feature_strategy')
        if strategy is None:  # no metadata: infer from the files that exist
            strategy = 'union' if os.path.exists(COL_VECTORIZERS_PATH) else 'concat'
            metadata['feature_strategy'] = strategy
        vec_path = COL_VECTORIZERS_PATH if strategy == 'union' else VECTORIZER_PATH
        if not os.path.exists(vec_path):
            return None, None, metadata, [f'Missing file: {vec_path}']
        return joblib.load(MODEL_PATH), joblib.load(vec_path), metadata, []
    except Exception as exc:  # corrupt or incompatible pickle
        return None, None, {}, [f'Could not load model files ({exc}). Re-run the notebook.']


@st.cache_data(show_spinner=False)
def load_dataset():
    """Return (train+test DataFrame with a 'sentiment' column, error message or None)."""
    required = {'urlDrugName', 'rating', 'condition'} | set(REVIEW_COLS)
    frames = []
    for split, path in (('train', TRAIN_PATH), ('test', TEST_PATH)):
        if not os.path.exists(path):
            return None, f'Dataset file not found: {path}'
        df = pd.read_csv(path, sep='\t', index_col=0)
        if not required.issubset(df.columns):
            return None, f'{os.path.basename(path)} is missing columns: {sorted(required - set(df.columns))}'
        df['split'] = split
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data['sentiment'] = data['rating'].apply(rating_to_sentiment)
    return data.dropna(subset=['sentiment']), None


def build_features(texts: dict, vectorizer, strategy: str):
    """Vectorize the three review fields exactly as the notebook did in training.

    union  : one vectorizer per column, matrices hstacked.
    concat : non-empty fields joined with a space, cleaned, one vectorizer.
    Returns (feature matrix, text shown to the user as "cleaned text").
    """
    if strategy == 'union':
        cleaned = {c: clean_text(texts[c]) for c in REVIEW_COLS}
        parts = [vectorizer[c].transform([cleaned[c]]) for c in REVIEW_COLS]
        shown = '\n'.join(f'[{FIELD_LABELS[c]}] {cleaned[c]}' for c in REVIEW_COLS if cleaned[c])
        return scipy.sparse.hstack(parts, format='csr'), shown
    combined = clean_text(' '.join(t for t in texts.values() if t.strip()))
    return vectorizer.transform([combined]), combined


# ---------------------------------------------------------------------------
# Sample reviews (one text per field)
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = {
    'Positive example': {
        'benefitsReview': 'This medication worked wonderfully for me. After just two weeks my symptoms improved significantly.',
        'sideEffectsReview': 'The side effects were minimal and manageable.',
        'commentsReview': 'I feel much better and would highly recommend it to others with the same condition.',
    },
    'Neutral example': {
        'benefitsReview': 'It seems to help a little, but not as much as I had hoped.',
        'sideEffectsReview': 'The side effects are there but tolerable.',
        'commentsReview': 'I have taken it for about a month. Not sure if I will continue using it.',
    },
    'Negative example': {
        'benefitsReview': 'It did not help with my symptoms at all.',
        'sideEffectsReview': 'I had severe side effects including dizziness and nausea.',
        'commentsReview': 'I do not recommend this drug at all. It made my condition worse.',
    },
}


def fill_sample(name: str):
    for col in REVIEW_COLS:
        st.session_state[f'in_{col}'] = SAMPLE_REVIEWS[name][col]


# ---------------------------------------------------------------------------
# Page configuration + sidebar
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='Drug Review Sentiment Analyzer',
    page_icon='💊',
    layout='wide',
    initial_sidebar_state='expanded',
)

model, vectorizer, metadata, problems = load_artifacts()
strategy = metadata.get('feature_strategy', 'concat')

with st.sidebar:
    st.title('💊 Drug Review Sentiment Analyzer')
    st.markdown('---')
    st.subheader('About This Project')
    st.markdown(
        'A machine-learning model trained on the **Drug Reviews (Druglib.com)** '
        'dataset predicts the sentiment expressed in drug reviews.'
    )
    st.subheader('Sentiment Classes')
    st.markdown(
        '- 🟢 **Positive** — Rating 7–10  \n'
        '- 🔵 **Neutral** — Rating 5–6  \n'
        '- 🔴 **Negative** — Rating 1–4'
    )
    if metadata:
        st.subheader('Model')
        st.markdown(f'**{metadata.get("model_name", "N/A")}**')
        st.markdown(
            f'Feature strategy: `{strategy}`  \n'
            f'Test accuracy: `{metadata.get("test_accuracy", "N/A")}`  \n'
            f'Test F1 (weighted): `{metadata.get("test_f1_weighted", "N/A")}`  \n'
            f'Test F1 (macro): `{metadata.get("test_f1_macro", "N/A")}`'
        )
    st.markdown('---')
    st.subheader('⚠️ Disclaimer')
    st.warning(
        'This application analyses sentiment expressed in user reviews. It is **not a medical '
        'diagnostic tool** and must not be used to judge medication safety, effectiveness, '
        'or treatment decisions.'
    )
    st.markdown('---')
    page = st.radio(
        'Navigate',
        ['🔍 Sentiment Analyzer', '📊 Dataset Analytics', '💊 Drug Analysis', 'ℹ️ Model Information'],
    )


def require_model():
    """Stop the page with a clear message if the artifacts could not be loaded."""
    if problems:
        st.error('\n\n'.join(problems) + '\n\nRun the Jupyter notebook first to train and save the model.')
        st.stop()
    trained_with = metadata.get('sklearn_version')
    if trained_with and trained_with != sklearn.__version__:
        st.warning(
            f'The model was trained with scikit-learn {trained_with}, but {sklearn.__version__} is installed. '
            'Predictions may differ. Re-run the notebook to rebuild the model with your version.'
        )


def require_data():
    data, error = load_dataset()
    if error:
        st.error(error)
        st.stop()
    return data


# ============================================================
# PAGE 1 - SENTIMENT ANALYZER
# ============================================================
if page == '🔍 Sentiment Analyzer':
    st.title('Drug Review Sentiment Analyzer')
    st.markdown('*Machine Learning Based Sentiment Analysis of Drug Reviews*')
    st.info(
        'This application analyses sentiment expressed in user reviews. It is not a medical '
        'diagnostic tool and must not be used to judge medication safety, effectiveness, '
        'or treatment decisions.'
    )
    require_model()

    st.subheader('Quick Examples')
    for col, name in zip(st.columns(3), SAMPLE_REVIEWS):
        col.button(name, on_click=fill_sample, args=(name,))

    st.subheader('Enter a Drug Review')
    st.caption('Fill in at least one box. Empty boxes are fine.')
    texts = {}
    for col, (field, label) in zip(st.columns(3), FIELD_LABELS.items()):
        texts[field] = col.text_area(label, key=f'in_{field}', height=170)

    if st.button('Analyze Sentiment', type='primary'):
        if not any(clean_text(t) for t in texts.values()):
            st.warning('Please enter some review text before clicking Analyze.')
        else:
            try:
                X, cleaned_shown = build_features(texts, vectorizer, strategy)
                label = str(model.predict(X)[0])
                st.markdown('---')
                left, right = st.columns([1, 2])

                with left:
                    st.markdown(
                        f'<div style="background:{SENTIMENT_COLORS.get(label, "#7f8c8d")};padding:20px;'
                        f'border-radius:10px;text-align:center;">'
                        f'<h2 style="color:white;margin:0;">{SENTIMENT_EMOJI.get(label, "")} {label.upper()}</h2>'
                        f'<p style="color:white;margin:4px 0 0;">Predicted Sentiment</p></div>',
                        unsafe_allow_html=True,
                    )
                    if hasattr(model, 'predict_proba'):
                        st.markdown('##### Prediction Confidence')
                        for cls, prob in sorted(zip(model.classes_, model.predict_proba(X)[0]), key=lambda t: -t[1]):
                            st.markdown(
                                f'<div style="margin:4px 0;"><span style="display:inline-block;width:85px;">{cls}</span>'
                                f'<div style="display:inline-block;width:{int(prob * 180)}px;height:12px;'
                                f'background:{SENTIMENT_COLORS.get(cls, "#7f8c8d")};border-radius:4px;'
                                f'vertical-align:middle;"></div> <b>{prob:.1%}</b></div>',
                                unsafe_allow_html=True,
                            )
                    else:  # LinearSVC has no probabilities
                        st.markdown('##### Model Decision Scores')
                        st.caption('Raw scores. They are not probabilities.')
                        for cls, score in sorted(zip(model.classes_, model.decision_function(X)[0]), key=lambda t: -t[1]):
                            st.markdown(f'- **{cls}**: `{score:.3f}`')

                with right:
                    full_text = ' '.join(t for t in texts.values() if t.strip())
                    words = len(full_text.split())
                    length_cat = 'Short' if words < 30 else 'Medium' if words < 100 else 'Long'
                    st.markdown('##### Review Statistics')
                    s1, s2, s3 = st.columns(3)
                    s1.metric('Words', words)
                    s2.metric('Characters', len(full_text))
                    s3.metric('Length', length_cat)
                    with st.expander('Cleaned text used for prediction'):
                        st.code(cleaned_shown, language=None)
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')


# ============================================================
# PAGE 2 - DATASET ANALYTICS
# ============================================================
elif page == '📊 Dataset Analytics':
    st.title('📊 Dataset Analytics')
    data = require_data()

    st.subheader('Summary')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Reviews', f'{len(data):,}')
    c2.metric('Unique Drugs', f'{data["urlDrugName"].nunique():,}')
    c3.metric('Unique Conditions', f'{data["condition"].nunique():,}')
    c4.metric('Average Rating', f'{data["rating"].mean():.2f}')

    st.markdown('---')
    left, right = st.columns(2)

    with left:
        st.subheader('Sentiment Distribution')
        counts = data['sentiment'].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=counts.index, colors=[SENTIMENT_COLORS[s] for s in counts.index],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Sentiment Distribution (All Data)')
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader('Rating Distribution')
        ratings = data['rating'].astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(ratings.index.astype(str), ratings.values, color='steelblue', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Rating Distribution (All Data)')
        st.pyplot(fig)
        plt.close(fig)

    st.subheader('Top 15 Drugs by Review Count')
    top = data['urlDrugName'].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top.index, top.values, color='teal', edgecolor='black')
    ax.set_xlabel('Number of Reviews')
    ax.set_title('Top 15 Most Reviewed Drugs')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PAGE 3 - DRUG ANALYSIS
# ============================================================
elif page == '💊 Drug Analysis':
    st.title('💊 Drug Analysis')
    st.caption('Statistics come from the dataset and show patient-reported experiences, not medical recommendations.')
    data = require_data()

    drug = st.selectbox('Select a Drug', sorted(data['urlDrugName'].dropna().unique()))
    sub = data[data['urlDrugName'] == drug]
    pct = sub['sentiment'].value_counts(normalize=True).reindex(SENTIMENT_ORDER, fill_value=0) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Reviews', len(sub))
    c2.metric('Avg Rating', f'{sub["rating"].mean():.1f}')
    c3.metric('Positive %', f'{pct["Positive"]:.1f}%')
    c4.metric('Neutral %', f'{pct["Neutral"]:.1f}%')
    c5.metric('Negative %', f'{pct["Negative"]:.1f}%')

    left, right = st.columns(2)
    with left:
        st.subheader('Sentiment Breakdown')
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(pct.values, labels=pct.index, colors=[SENTIMENT_COLORS[s] for s in pct.index],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Sentiment: {drug}')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader('Rating Distribution')
        ratings = sub['rating'].astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(ratings.index.astype(str), ratings.values, color='slateblue', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title(f'Ratings: {drug}')
        st.pyplot(fig)
        plt.close(fig)

    st.info('These statistics reflect patient-reported opinions. They are not clinical evidence '
            'and must not influence treatment decisions.')


# ============================================================
# PAGE 4 - MODEL INFORMATION
# ============================================================
elif page == 'ℹ️ Model Information':
    st.title('ℹ️ Model Information')
    require_model()

    steps = {
        '1. Data Loading': (
            '`train.tsv` and `test.tsv` are loaded. Each record has three free-text review columns: '
            '`benefitsReview`, `sideEffectsReview`, and `commentsReview`. A missing column counts as empty text.'
        ),
        '2. Sentiment Labeling': (
            'The numeric `rating` (1–10) is mapped to a label:  \n'
            '- Rating 1–4 → **Negative**  \n- Rating 5–6 → **Neutral**  \n- Rating 7–10 → **Positive**  \n'
            'The rating is **not** a model feature.'
        ),
        '3. Text Preprocessing': (
            'Lowercase, HTML tag removal, URL removal, punctuation removal (apostrophes kept), '
            'whitespace normalization. No stop-word removal, so negations (not, no, never) stay in the text.'
        ),
        '4. Duplicate Removal': (
            'Exact duplicate reviews are dropped from the training data. Test reviews that also appear '
            'in the training data are dropped from the test data, so no text sits on both sides.'
        ),
        '5. TF-IDF - Baseline (Concatenated Text)': (
            'The three columns are joined into one string and one TF-IDF vectorizer is fitted on it.  \n'
            'Parameters: `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=50000`.'
        ),
        '6. TF-IDF - Per-Column Feature Union': (
            'One TF-IDF vectorizer per column (`max_features=20000` each). The three sparse matrices are '
            'stacked with `scipy.sparse.hstack`, so a word keeps a separate weight per column.'
        ),
        '7. Feature Strategy Selection': (
            'Both strategies are scored with all three classifiers on the validation set. The strategy with the '
            'highest weighted F1 is used.  \n'
            f'**Strategy in this deployment: `{strategy}`**'
        ),
        '8. Model Training & Selection': (
            'Logistic Regression, Multinomial Naive Bayes and Linear SVM are trained on a stratified 80/20 split. '
            'The best weighted F1 wins and is retrained on the full training data. All vectorizers are fitted on training text only.'
        ),
        '9. Final Evaluation': 'The selected model is scored once on the held-out test set.',
        '10. Artifact Saving': (
            'The model, vectorizer(s) and metadata are saved with `joblib` in `models/`.  \n'
            '- Concat mode: `sentiment_model.pkl` + `tfidf_vectorizer.pkl`  \n'
            '- Union mode: `sentiment_model.pkl` + `col_tfidf_vectorizers.pkl`'
        ),
    }
    st.subheader('Machine Learning Pipeline')
    for title, desc in steps.items():
        with st.expander(title):
            st.markdown(desc)

    st.markdown('---')
    st.subheader('Selected Model & Performance')
    st.markdown(f'**Algorithm:** {metadata.get("model_name", "N/A")}')
    st.markdown(f'**Feature strategy:** `{strategy}`')
    if 'n_train_samples' in metadata:
        st.markdown(f'**Trained on:** {metadata["n_train_samples"]:,} reviews  \n'
                    f'**Tested on:** {metadata["n_test_samples"]:,} reviews')
    c1, c2, c3 = st.columns(3)
    c1.metric('Test Accuracy', metadata.get('test_accuracy', 'N/A'))
    c2.metric('Test F1 (Weighted)', metadata.get('test_f1_weighted', 'N/A'))
    c3.metric('Test F1 (Macro)', metadata.get('test_f1_macro', 'N/A'))
    st.caption('The Neutral class (rating 5–6) is the weakest: most Neutral reviews are predicted as another class.')

    st.markdown('---')
    st.subheader('Limitations')
    st.markdown(
        '- Bag-of-words / TF-IDF cannot fully capture context, sarcasm, or complex negation.  \n'
        '- Labels come from ratings, which are subjective and imperfect proxies.  \n'
        '- The model has no medical knowledge. It analyses language patterns only.  \n'
        '- Reviews may contain bias, errors, or unusual language.  \n'
        '- This system is **not a medical diagnostic tool** and must not be used for clinical decisions.'
    )
    st.subheader('Future Improvements')
    st.markdown(
        '- Transformer models (BERT, BioBERT) for richer language understanding.  \n'
        '- Aspect-based sentiment analysis for specific drug attributes.  \n'
        '- Better negation handling with dependency parsing.  \n'
        '- Multilingual reviews.  \n'
        '- Explainability (LIME / SHAP).  \n'
        '- Larger and more diverse datasets.'
    )
