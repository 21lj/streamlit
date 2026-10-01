"""
Drug Review Sentiment Analyzer — Streamlit Application
=======================================================
Loads pre-trained artifacts from models/ and performs inference.
Does NOT retrain the model.

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import re
import sys

import scipy.sparse
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------------------------
# Path setup — resolve project root regardless of where the app is launched
# ---------------------------------------------------------------------------
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

MODEL_PATH           = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH      = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
COL_VECTORIZERS_PATH = os.path.join(MODELS_DIR, 'col_tfidf_vectorizers.pkl')
METADATA_PATH        = os.path.join(MODELS_DIR, 'model_metadata.pkl')
TRAIN_PATH           = os.path.join(DATA_DIR,   'train.tsv')
TEST_PATH            = os.path.join(DATA_DIR,   'test.tsv')

REVIEW_COLS = ['benefitsReview', 'sideEffectsReview', 'commentsReview']

# ---------------------------------------------------------------------------
# Text preprocessing (must mirror the notebook pipeline)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Model loading (cached so artifacts are loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model + vectorizer(s). Returns (model, vectorizer_or_dict, metadata, missing_files)."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, [MODEL_PATH]

    metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    strategy = metadata.get('feature_strategy', 'concat')

    model = joblib.load(MODEL_PATH)

    if strategy == 'union':
        if not os.path.exists(COL_VECTORIZERS_PATH):
            return None, None, metadata, [COL_VECTORIZERS_PATH]
        vectorizer = joblib.load(COL_VECTORIZERS_PATH)  # dict: col -> TfidfVectorizer
    else:
        if not os.path.exists(VECTORIZER_PATH):
            return None, None, metadata, [VECTORIZER_PATH]
        vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer, metadata, []


def transform_input(text: str, vectorizer, strategy: str):
    """Vectorize a raw review string using the saved feature strategy.

    For the 'concat' strategy, the single combined review text is transformed.
    For the 'union' strategy, the same text is used for all three column
    vectorizers (since there is only one input box in the app — we cannot
    separate it back into benefit/side-effects/comments at inference time).
    The three resulting sparse vectors are hstacked exactly as during training.
    """
    cleaned = clean_text(text)
    if strategy == 'union':
        parts = [vectorizer[col].transform([cleaned]) for col in REVIEW_COLS]
        return scipy.sparse.hstack(parts, format='csr')
    else:
        return vectorizer.transform([cleaned])


@st.cache_data(show_spinner=False)
def load_dataset():
    missing = []
    for p in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        return None, None, missing

    def _read(path):
        df = pd.read_csv(path, sep='\t', index_col=0)
        # Validate expected columns
        required = {'urlDrugName', 'rating', 'condition',
                    'benefitsReview', 'sideEffectsReview', 'commentsReview'}
        found = set(df.columns)
        if not required.issubset(found):
            st.error(
                f"Unexpected columns in {os.path.basename(path)}.\n"
                f"Expected: {required}\nFound: {found}"
            )
            return None
        df['review'] = (
            df[['benefitsReview', 'sideEffectsReview', 'commentsReview']]
            .fillna('')
            .apply(lambda r: ' '.join(v for v in r.values if v.strip()), axis=1)
        )
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['sentiment'] = df['rating'].apply(rating_to_sentiment)
        df = df.dropna(subset=['sentiment'])
        return df

    train_df = _read(TRAIN_PATH)
    test_df  = _read(TEST_PATH)
    return train_df, test_df, []


def rating_to_sentiment(rating):
    try:
        r = int(rating)
    except (ValueError, TypeError):
        return None
    if r <= 4:
        return 'Negative'
    elif r <= 6:
        return 'Neutral'
    else:
        return 'Positive'


def sentiment_color(label: str) -> str:
    return {'Positive': '#27ae60', 'Neutral': '#2980b9', 'Negative': '#e74c3c'}.get(label, '#7f8c8d')


def sentiment_emoji(label: str) -> str:
    return {'Positive': '✅', 'Neutral': '🔵', 'Negative': '❌'}.get(label, '❓')


# ---------------------------------------------------------------------------
# Sample reviews
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = {
    'Positive example': (
        "This medication worked wonderfully for me. After just two weeks, my symptoms "
        "improved significantly. The side effects were minimal and manageable. "
        "I feel much better and would highly recommend it to others dealing with the same condition."
    ),
    'Neutral example': (
        "I have been taking this drug for about a month. It seems to help a little, "
        "but not as much as I had hoped. The side effects are there but tolerable. "
        "Not sure if I will continue using it."
    ),
    'Negative example': (
        "This medication was a nightmare for me. Not only did it not help with my symptoms, "
        "but I experienced severe side effects including dizziness and nausea. "
        "I do not recommend this drug at all. It made my condition worse."
    ),
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='Drug Review Sentiment Analyzer',
    page_icon='💊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title('💊 Drug Review Sentiment Analyzer')
    st.markdown('---')

    st.subheader('About This Project')
    st.markdown(
        'This application uses a machine-learning model trained on the '
        '**Drug Reviews (Druglib.com)** dataset to predict the sentiment '
        'expressed in drug reviews.'
    )

    st.subheader('Sentiment Classes')
    st.markdown(
        '- 🟢 **Positive** — Rating 7–10  \n'
        '- 🔵 **Neutral** — Rating 5–6  \n'
        '- 🔴 **Negative** — Rating 1–4'
    )

    _, _, meta, _ = load_artifacts()
    if meta:
        st.subheader('Model')
        st.markdown(f'**{meta.get("model_name", "N/A")}**')
        st.markdown(
            f'Feature Strategy: `{meta.get("feature_strategy", "concat")}`  \n'
            f'Test Accuracy: `{meta.get("test_accuracy", "N/A")}`  \n'
            f'Test F1 (Weighted): `{meta.get("test_f1_weighted", "N/A")}`  \n'
            f'Test F1 (Macro): `{meta.get("test_f1_macro", "N/A")}`'
        )

    st.markdown('---')
    st.subheader('⚠️ Disclaimer')
    st.warning(
        'This application analyses sentiment expressed in user reviews. '
        'It is **not a medical diagnostic tool** and should not be used to determine '
        'medication safety, effectiveness, or treatment decisions.'
    )

    st.markdown('---')
    page = st.radio(
        'Navigate',
        ['🔍 Sentiment Analyzer', '📊 Dataset Analytics', '💊 Drug Analysis', 'ℹ️ Model Information'],
    )

# ---------------------------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------------------------
model, vectorizer, metadata, missing_files = load_artifacts()

# ============================================================
# PAGE 1 — SENTIMENT ANALYZER
# ============================================================
if page == '🔍 Sentiment Analyzer':
    st.title('Drug Review Sentiment Analyzer')
    st.markdown('*Machine Learning Based Sentiment Analysis of Drug Reviews*')
    st.info(
        '⚠️ This application analyses sentiment expressed in user reviews. '
        'It is not a medical diagnostic tool and should not be used to determine '
        'medication safety, effectiveness, or treatment decisions.'
    )

    if missing_files:
        st.error(
            f'Model artifacts not found: {", ".join(missing_files)}\n\n'
            'Please run the Jupyter Notebook first to train and save the model.'
        )
        st.stop()

    # Sample review buttons
    st.subheader('Quick Examples')
    cols = st.columns(3)
    for col, (label, review_text) in zip(cols, SAMPLE_REVIEWS.items()):
        if col.button(label):
            st.session_state['review_input'] = review_text

    # Text input
    st.subheader('Enter a Drug Review')
    default_text = st.session_state.get('review_input', '')
    review_input = st.text_area(
        'Drug review text',
        value=default_text,
        height=180,
        placeholder='I have been using this medication for several weeks and it has helped improve my symptoms.',
        label_visibility='collapsed',
    )

    analyze_btn = st.button('Analyze Sentiment', type='primary', use_container_width=True)

    if analyze_btn:
        if not review_input.strip():
            st.warning('Please enter a drug review before clicking Analyze.')
        else:
            cleaned = clean_text(review_input)
            if not cleaned:
                st.error('The review could not be processed. Please enter valid text.')
            else:
                try:
                    strategy = metadata.get('feature_strategy', 'concat')
                    X = transform_input(review_input, vectorizer, strategy)
                    predicted_label = model.predict(X)[0]

                    st.markdown('---')
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        color = sentiment_color(predicted_label)
                        emoji = sentiment_emoji(predicted_label)
                        st.markdown(
                            f'<div style="background:{color};padding:20px;border-radius:10px;text-align:center;">'
                            f'<h2 style="color:white;margin:0;">{emoji} {predicted_label.upper()}</h2>'
                            f'<p style="color:white;margin:4px 0 0;">Predicted Sentiment</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Confidence / decision score
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X)[0]
                            classes = model.classes_
                            st.markdown('##### Prediction Confidence')
                            for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
                                bar_color = sentiment_color(cls)
                                st.markdown(
                                    f'<div style="margin:4px 0;">'
                                    f'<span style="display:inline-block;width:85px;">{cls}</span>'
                                    f'<div style="display:inline-block;width:{int(prob*180)}px;height:12px;'
                                    f'background:{bar_color};border-radius:4px;vertical-align:middle;"></div>'
                                    f' <b>{prob:.1%}</b>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            # LinearSVC: show decision scores, not probabilities
                            scores = model.decision_function(X)[0]
                            classes = model.classes_
                            st.markdown('##### Model Decision Scores')
                            st.caption('(Not calibrated probabilities)')
                            for cls, sc in sorted(zip(classes, scores), key=lambda x: -x[1]):
                                st.markdown(f'- **{cls}**: `{sc:.3f}`')

                    with col2:
                        words  = len(review_input.split())
                        chars  = len(review_input)
                        if words < 30:
                            length_cat = 'Short'
                        elif words < 100:
                            length_cat = 'Medium'
                        else:
                            length_cat = 'Long'

                        st.markdown('##### Review Statistics')
                        s1, s2, s3 = st.columns(3)
                        s1.metric('Words', words)
                        s2.metric('Characters', chars)
                        s3.metric('Length', length_cat)

                        with st.expander('Cleaned text used for prediction'):
                            st.code(cleaned, language=None)

                except Exception as exc:
                    st.error(f'An error occurred during prediction. Please try again.\n\nDetails: {exc}')


# ============================================================
# PAGE 2 — DATASET ANALYTICS
# ============================================================
elif page == '📊 Dataset Analytics':
    st.title('📊 Dataset Analytics')

    train_df, test_df, ds_missing = load_dataset()

    if ds_missing:
        st.error(f'Dataset files not found: {", ".join(ds_missing)}')
        st.stop()

    if train_df is None or test_df is None:
        st.error('Could not load the dataset. Check error messages above.')
        st.stop()

    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Summary metrics
    st.subheader('Summary')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Reviews',   f'{len(combined):,}')
    c2.metric('Unique Drugs',     f'{combined["urlDrugName"].nunique():,}')
    c3.metric('Unique Conditions',f'{combined["condition"].nunique():,}')
    c4.metric('Average Rating',   f'{combined["rating"].mean():.2f}')

    st.markdown('---')
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader('Sentiment Distribution')
        sent_counts = combined['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'])
        colors_pie = ['#27ae60', '#2980b9', '#e74c3c']
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.pie(
            sent_counts.values,
            labels=sent_counts.index,
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
        )
        ax1.set_title('Sentiment Distribution (All Data)')
        st.pyplot(fig1)
        plt.close(fig1)

    with col_right:
        st.subheader('Rating Distribution')
        rating_counts = combined['rating'].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(rating_counts.index.astype(str), rating_counts.values, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Count')
        ax2.set_title('Rating Distribution (All Data)')
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader('Top 15 Drugs by Review Count')
    top_drugs = combined['urlDrugName'].value_counts().head(15).sort_values()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.barh(top_drugs.index, top_drugs.values, color='teal', edgecolor='black')
    ax3.set_xlabel('Number of Reviews')
    ax3.set_title('Top 15 Most Reviewed Drugs')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ============================================================
# PAGE 3 — DRUG ANALYSIS
# ============================================================
elif page == '💊 Drug Analysis':
    st.title('💊 Drug Analysis')
    st.caption('Statistics are derived from the dataset and represent patient-reported experiences — not medical recommendations.')

    train_df, test_df, ds_missing = load_dataset()

    if ds_missing:
        st.error(f'Dataset files not found: {", ".join(ds_missing)}')
        st.stop()

    if train_df is None or test_df is None:
        st.error('Could not load the dataset.')
        st.stop()

    combined = pd.concat([train_df, test_df], ignore_index=True)

    drug_list = sorted(combined['urlDrugName'].dropna().unique().tolist())
    selected_drug = st.selectbox('Select a Drug', drug_list)

    drug_df = combined[combined['urlDrugName'] == selected_drug]

    if drug_df.empty:
        st.warning('No records found for the selected drug.')
    else:
        n_reviews = len(drug_df)
        avg_rating = drug_df['rating'].mean()
        sent_pcts  = drug_df['sentiment'].value_counts(normalize=True) * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Reviews', n_reviews)
        c2.metric('Avg Rating', f'{avg_rating:.1f}')
        c3.metric('Positive %', f'{sent_pcts.get("Positive", 0):.1f}%')
        c4.metric('Neutral %',  f'{sent_pcts.get("Neutral",  0):.1f}%')
        c5.metric('Negative %', f'{sent_pcts.get("Negative", 0):.1f}%')

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader('Sentiment Breakdown')
            sent_vals = [sent_pcts.get(s, 0) for s in ['Positive', 'Neutral', 'Negative']]
            fig_d1, ax_d1 = plt.subplots(figsize=(5, 4))
            ax_d1.pie(
                sent_vals,
                labels=['Positive', 'Neutral', 'Negative'],
                colors=['#27ae60', '#2980b9', '#e74c3c'],
                autopct='%1.1f%%',
                startangle=90,
            )
            ax_d1.set_title(f'Sentiment: {selected_drug}')
            st.pyplot(fig_d1)
            plt.close(fig_d1)

        with col_b:
            st.subheader('Rating Distribution')
            r_counts = drug_df['rating'].value_counts().sort_index()
            fig_d2, ax_d2 = plt.subplots(figsize=(5, 4))
            ax_d2.bar(r_counts.index.astype(str), r_counts.values, color='slateblue', edgecolor='black')
            ax_d2.set_xlabel('Rating')
            ax_d2.set_ylabel('Count')
            ax_d2.set_title(f'Ratings: {selected_drug}')
            st.pyplot(fig_d2)
            plt.close(fig_d2)

        st.info(
            '⚠️ These statistics reflect patient-reported opinions from the dataset. '
            'They are not clinical evidence and should not influence treatment decisions.'
        )


# ============================================================
# PAGE 4 — MODEL INFORMATION
# ============================================================
elif page == 'ℹ️ Model Information':
    st.title('ℹ️ Model Information')

    if missing_files:
        st.error(
            f'Model artifacts not found: {", ".join(missing_files)}\n\n'
            'Run the Jupyter Notebook to generate the model artifacts.'
        )
        st.stop()

    st.subheader('Machine Learning Pipeline')

    feat_strategy = metadata.get('feature_strategy', 'concat') if metadata else 'concat'

    steps = {
        '1. Data Loading': (
            'The raw `train.tsv` and `test.tsv` files are loaded. '
            'Each record has three free-text review columns: '
            '`benefitsReview`, `sideEffectsReview`, and `commentsReview`.'
        ),
        '2. Sentiment Labeling': (
            'The numeric `rating` column (1–10) is mapped to sentiment labels:  \n'
            '- Rating 1–4 → **Negative**  \n'
            '- Rating 5–6 → **Neutral**  \n'
            '- Rating 7–10 → **Positive**  \n'
            'The rating is **not** used as a model feature.'
        ),
        '3. Text Preprocessing': (
            'Reviews are cleaned with: lowercase conversion, HTML tag removal, '
            'URL removal, punctuation removal (preserving apostrophes), '
            'and whitespace normalization. Negation words (not, no, never, etc.) are preserved.'
        ),
        '4. TF-IDF Vectorization — Baseline (Concatenated Text)': (
            'Approach A: The three review columns are concatenated into one string and a single '
            'TF-IDF vectorizer is fitted on it.  \n'
            'Parameters: `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, '
            '`max_features=50000`.  \n'
            'The vectorizer is fitted **only on training data** to prevent data leakage.'
        ),
        '5. TF-IDF Vectorization — Per-Column Feature Union': (
            'Approach B: A separate TF-IDF vectorizer is fitted on each of the three review columns '
            'individually (`benefitsReview`, `sideEffectsReview`, `commentsReview`).  \n'
            'The three resulting sparse matrices are horizontally stacked (`scipy.sparse.hstack`) '
            'to form one combined feature matrix.  \n'
            'This preserves column-specific vocabulary patterns — e.g. words that appear '
            'predominantly in side-effects reviews carry different weight than the same words '
            'in benefits reviews.'
        ),
        '6. Feature Strategy Selection': (
            'Both feature strategies are evaluated with all three classifiers on the validation set.  \n'
            'The strategy whose best model achieves the highest weighted F1-score is carried forward.  \n'
            f'**Strategy used in this deployment: `{feat_strategy}`**'
        ),
        '7. Model Training & Selection': (
            'Three classifiers are trained and compared on a stratified 80/20 validation split:  \n'
            '- Logistic Regression  \n'
            '- Multinomial Naive Bayes  \n'
            '- Linear SVM (LinearSVC)  \n'
            'The model with the highest weighted F1-score on the validation set is selected and '
            'retrained on the full training data.'
        ),
        '8. Final Evaluation': (
            'The selected model is evaluated **once** on the held-out test set (`test.tsv`). '
            'Accuracy, Precision, Recall, and F1-scores are reported.'
        ),
        '9. Artifact Saving': (
            'The trained model and TF-IDF vectorizer(s) are saved using `joblib` as `.pkl` files '
            'in the `models/` directory. This Streamlit app loads these files for inference.  \n'
            '- Concat mode: `sentiment_model.pkl` + `tfidf_vectorizer.pkl`  \n'
            '- Union mode: `sentiment_model.pkl` + `col_tfidf_vectorizers.pkl`'
        ),
    }

    for title, desc in steps.items():
        with st.expander(title, expanded=False):
            st.markdown(desc)

    st.markdown('---')
    st.subheader('Selected Model & Performance')

    if metadata:
        st.markdown(f'**Algorithm:** {metadata.get("model_name", "N/A")}')
        st.markdown(f'**Feature Strategy:** `{feat_strategy}`')
        col1, col2, col3 = st.columns(3)
        col1.metric('Test Accuracy',     metadata.get('test_accuracy',    'N/A'))
        col2.metric('Test F1 (Weighted)',metadata.get('test_f1_weighted', 'N/A'))
        col3.metric('Test F1 (Macro)',   metadata.get('test_f1_macro',    'N/A'))

    st.markdown('---')
    st.subheader('Limitations')
    st.markdown(
        '- Bag-of-words / TF-IDF cannot fully capture context, sarcasm, or complex negation.  \n'
        '- Sentiment labels are derived from ratings, which are subjective and imperfect proxies.  \n'
        '- The model has no medical knowledge — it analyses linguistic patterns only.  \n'
        '- Reviews in the dataset may contain bias, errors, or atypical language.  \n'
        '- This system is **not a medical diagnostic tool** and must not be used for clinical decisions.'
    )

    st.subheader('Future Improvements')
    st.markdown(
        '- Use transformer-based models (BERT, BioBERT) for richer language understanding.  \n'
        '- Implement aspect-based sentiment analysis to target specific drug attributes.  \n'
        '- Improve negation handling with dependency parsing.  \n'
        '- Support multilingual reviews.  \n'
        '- Add explainability (LIME / SHAP) to highlight influential words.  \n'
        '- Train on larger, more diverse datasets.'
    )
