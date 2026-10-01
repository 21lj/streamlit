# Drug Review Sentiment Analysis Using Machine Learning

> ⚠️ **Disclaimer:** This project analyses user-written opinions expressed in drug reviews. It does **not** determine whether a medication is medically safe, effective, or clinically appropriate for any patient. The outputs of this project must not be used to make any health or treatment decisions.

---

## Overview

This project is a complete, end-to-end machine-learning system for **sentiment classification of patient drug reviews**. Given a written drug review, the system predicts whether the expressed sentiment is **Positive**, **Neutral**, or **Negative**.

The project was built as a 1-week internship exercise using traditional NLP techniques and classical machine-learning algorithms — deliberately avoiding deep learning in favour of interpretable, fast-to-train models that are well-suited to a constrained timeline.

The work is divided into two main components:

1. **A Jupyter Notebook** — responsible for all data exploration, preprocessing, model training, evaluation, and artifact saving.
2. **A Streamlit Web Application** — a multi-page interactive frontend that loads the saved artifacts and provides sentiment predictions, dataset analytics, and model documentation.

---

## Objectives

- Understand the distribution of drug reviews across sentiment classes, drugs, and medical conditions.
- Derive three-class sentiment labels (Positive / Neutral / Negative) from numeric patient ratings.
- Build a text-preprocessing pipeline that preserves sentiment-critical words such as negations.
- Experiment with two TF-IDF feature strategies — concatenated text vs. per-column feature union — and automatically select the better one.
- Train and compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM classifiers.
- Select the best model on a held-out validation set, retrain it on the full training data, and evaluate it once on the test set.
- Expose the trained model through a clean, production-style Streamlit application.

---

## Dataset

The project uses the **Drug Reviews (Druglib.com)** dataset. Two TSV files are provided:

| File | Role | Size |
|------|------|------|
| `data/train.tsv` | Used for all training and validation | 3,107 records |
| `data/test.tsv` | Held out until final evaluation only | 1,036 records |

### Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `urlDrugName` | Text | Name of the drug being reviewed |
| `rating` | Integer (1–10) | Overall patient satisfaction score |
| `effectiveness` | Categorical | Perceived effectiveness (e.g. Highly Effective) |
| `sideEffects` | Categorical | Perceived severity of side effects |
| `condition` | Text | Medical condition the drug was used to treat |
| `benefitsReview` | Free text | Patient's description of the drug's benefits |
| `sideEffectsReview` | Free text | Patient's description of experienced side effects |
| `commentsReview` | Free text | General comments and overall opinion |

The three free-text columns — `benefitsReview`, `sideEffectsReview`, and `commentsReview` — are the primary inputs to the sentiment model. They capture distinct aspects of the patient experience and are treated separately in the per-column feature union approach.

---

## Sentiment Label Derivation

Sentiment labels are not collected directly from patients. Instead, they are derived from the numeric `rating` field using a fixed mapping:

| Rating Range | Sentiment Label |
|---|---|
| 1 – 4 | Negative |
| 5 – 6 | Neutral |
| 7 – 10 | Positive |

This mapping is a deliberate design choice: the `rating` field is a clean, consistent proxy for overall satisfaction. Using it as the labelling source avoids the need for manual annotation.

**Important:** The `rating` is used **only to create labels** — it is never given to the model as an input feature. The model learns to predict sentiment purely from the review text.

The training set is heavily imbalanced: approximately 69% Positive, 21% Negative, and 10% Neutral. All evaluation metrics therefore use both weighted and macro averaging to give a fair picture of per-class performance.

---

## Text Preprocessing

The preprocessing pipeline is designed to be lightweight and sentiment-preserving:

1. **Lowercase conversion** — normalises capitalisation.
2. **HTML tag removal** — strips any inline HTML that may appear in scraped reviews.
3. **URL removal** — removes web addresses that carry no sentiment signal.
4. **Punctuation removal** — removes symbols while retaining apostrophes (needed for contractions).
5. **Whitespace normalisation** — collapses multiple spaces and trims leading/trailing spaces.

**Negation words are explicitly preserved.** Words such as *not*, *no*, *never*, *didn't*, *doesn't* are excluded from any stop-word filtering. This is critical because removing them reverses the polarity of a statement — for example, "did not help" would become "did help" after aggressive stop-word removal, completely flipping the sentiment.

---

## Feature Engineering — Two TF-IDF Strategies

A key experimental feature of this project is the comparison of two different ways of converting review text into numerical features.

### Strategy A — Concatenated Text (Baseline)

All three review columns are joined into a single string per record. One `TfidfVectorizer` is fitted on this combined text.

- **Parameters:** `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=50000`
- **Advantage:** Simple; all vocabulary is visible to a single model.
- **Disadvantage:** Loses the information about *which part* of the review a word came from.

### Strategy B — Per-Column Feature Union

A separate `TfidfVectorizer` is fitted on each of the three columns individually. The three resulting sparse matrices are then horizontally stacked using `scipy.sparse.hstack` to form one combined feature matrix.

- **Parameters per vectorizer:** `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=20000`
- **Advantage:** Captures column-specific vocabulary patterns. A word appearing in `sideEffectsReview` carries a different positional signal than the same word in `benefitsReview`.
- **Disadvantage:** Slightly more complex; the total feature space is larger.

### Automatic Selection

Both strategies are evaluated with all three classifiers on the validation set. The strategy whose best model achieves the highest weighted F1-score is automatically selected and used for all subsequent steps (final retraining, test evaluation, and Streamlit inference).

---

## Machine Learning Models

Three classical supervised classifiers are trained and compared:

| Model | Notes |
|-------|-------|
| **Logistic Regression** | `max_iter=1000`, `class_weight='balanced'` |
| **Multinomial Naive Bayes** | Default parameters; requires non-negative TF-IDF values |
| **Linear SVM** | `LinearSVC`, `max_iter=2000`, `class_weight='balanced'` |

All models are trained on the same feature matrix (whichever strategy was selected) and evaluated on the same validation split.

### Evaluation Metrics

- **Accuracy** — overall fraction of correct predictions.
- **Weighted Precision / Recall / F1** — averages weighted by class support; appropriate for imbalanced datasets.
- **Macro F1** — unweighted average across classes; highlights performance on minority classes (especially Neutral).
- **Confusion matrix** — shows exactly which classes are being confused.

The model with the highest **weighted F1** on the validation set is selected as the final model. Accuracy alone is not used as the selection criterion because the class imbalance makes it misleading.

---

## Training Methodology

The notebook follows a strict data-separation discipline to ensure unbiased evaluation:

1. `train.tsv` is loaded and split 80/20 (stratified) into `X_train` and `X_validation`.
2. The TF-IDF vectorizer(s) are fitted **only on `X_train`** — never on validation or test data. This prevents data leakage, where information from unseen data would otherwise inflate vocabulary statistics and performance scores.
3. All three classifiers are trained on `X_train` and evaluated on `X_validation`.
4. The best model is identified. It is then instantiated fresh and retrained on the **full `train.tsv`** (all 3,107 records) so that no training data is wasted.
5. The TF-IDF vectorizer(s) are also refit on the full training set at this point.
6. The final model is evaluated **exactly once** on `test.tsv`. This result is the reported test performance.

---

## Project Structure

```
drug-review-sentiment/
│
├── data/
│   ├── train.tsv                         ← training data (3,107 records)
│   └── test.tsv                          ← held-out test data (1,036 records)
│
├── notebooks/
│   └── drug_review_sentiment_analysis.ipynb   ← full ML pipeline
│
├── models/                               ← generated by running the notebook
│   ├── sentiment_model.pkl               ← trained classifier
│   ├── tfidf_vectorizer.pkl              ← concat-mode vectorizer (if used)
│   ├── col_tfidf_vectorizers.pkl         ← per-column vectorizers dict (if union mode)
│   └── model_metadata.pkl               ← model name, metrics, feature strategy
│
├── app/
│   └── streamlit_app.py                  ← Streamlit web application
│
├── requirements.txt
└── README.md
```

---

## Notebook Structure

The notebook is organised into clearly labelled sections:

| Section | Content |
|---------|---------|
| 1 | Project introduction, objectives, and disclaimer |
| 2 | Library imports |
| 3 | Dataset loading and schema inspection |
| 4 | Exploratory data analysis (distributions, top drugs, review lengths) |
| 5 | Data cleaning and preprocessing pipeline |
| 6 | Sentiment label creation from ratings |
| 7 | Train / validation split |
| 8 | Baseline TF-IDF vectorization (concatenated text) |
| 9 | Per-column TF-IDF feature union experiment and strategy selection |
| 10 | Model training and comparison on validation set |
| 11 | Model selection |
| 12 | Final evaluation on the held-out test set |
| 13 | Saving model artifacts |

---

## Streamlit Application

The web application has four pages, accessible from the sidebar:

### 🔍 Sentiment Analyzer
The main prediction interface. Enter any drug review text and the model predicts its sentiment (Positive, Neutral, or Negative). The result is displayed with colour coding (green / blue / red). If the model supports probability estimates, a confidence bar chart is shown; otherwise the raw decision scores from the SVM are displayed (clearly labelled as scores, not probabilities). Basic review statistics (word count, character count, length category) are also shown. Three sample review buttons allow quick testing with pre-written examples.

### 📊 Dataset Analytics
Displays aggregate statistics across both train and test data: total reviews, unique drugs, unique conditions, average rating, sentiment distribution, rating distribution, and a chart of the top 15 most-reviewed drugs.

### 💊 Drug Analysis
Allows the user to select any drug from the dataset and view its individual statistics: number of reviews, average rating, and the percentage of Positive / Neutral / Negative reviews. All statistics are labelled as patient-reported data, not medical recommendations.

### ℹ️ Model Information
Documents the full machine-learning pipeline in plain language, explains both TF-IDF feature strategies, shows the selected model name and feature strategy, and displays the final test set performance metrics. Also includes a limitations section and a list of suggested future improvements.

---

## Installation

### 1. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open the notebook from the project root and run all cells in order:

```bash
jupyter notebook notebooks/drug_review_sentiment_analysis.ipynb
```

Or with JupyterLab:

```bash
jupyter lab
```

Select **Kernel → Restart & Run All** to execute every cell from top to bottom. This must be done before launching the Streamlit app, as it generates the saved model artifacts.

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. All inference is performed using the pre-trained artifacts from `models/` — the model is never retrained when the app starts.

---

## Results

The table below will be populated after running the notebook. Placeholder values are intentionally left blank.

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | — | — |
| Precision (Weighted) | — | — |
| Recall (Weighted) | — | — |
| F1 (Weighted) | — | — |
| F1 (Macro) | — | — |

The `model_metadata.pkl` file saved by the notebook records the exact test metrics and the feature strategy used.

---

## Limitations

- **Reviews represent individual patient opinions**, not clinical evidence. A high sentiment score does not mean a drug is effective or safe.
- **Ratings are imperfect proxy labels.** A patient may write a predominantly positive review but assign a mid-range rating, causing a labelling mismatch.
- **The Neutral class is structurally difficult.** It occupies a narrow rating band (5–6) and is the most ambiguous in text — models consistently underperform on it.
- **Class imbalance** (≈69% Positive) biases the model toward the majority class. Weighted loss and weighted evaluation metrics partially compensate for this.
- **Bag-of-words cannot capture context.** TF-IDF represents text as an unordered set of tokens. Sarcasm, complex negation ("not entirely unhelpful"), and domain-specific abbreviations are not handled well.
- **The dataset may contain inherent bias.** Online self-reported reviewers are not a random sample of all patients — they tend to have stronger opinions (very positive or very negative), and certain conditions or demographics may be over- or under-represented.
- **This system is not a medical diagnostic tool** and must not be used to guide any clinical or treatment decision.

---

## Future Improvements

- **Transformer-based models** (BERT, BioBERT, RoBERTa) would capture contextual meaning and handle negation and sarcasm far better than TF-IDF.
- **Aspect-based sentiment analysis** could separate opinions on efficacy, side-effect severity, cost, and convenience rather than producing a single overall label.
- **Better negation handling** using dependency parsing (e.g. spaCy) could identify the exact scope of negation in a sentence.
- **Calibrated probabilities** for LinearSVC via `CalibratedClassifierCV` would allow genuine confidence scores to be displayed in the app.
- **Multilingual support** to handle reviews written in languages other than English.
- **Explainability** using LIME or SHAP to highlight the specific words that most influenced each prediction.
- **Larger and more diverse datasets** to improve generalisation across drug categories and patient demographics.
