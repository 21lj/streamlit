# 💼 Salary Prediction App

A machine learning web app that predicts annual salary based on employee profile using a Gradient Boosting model, served via a Streamlit UI.

---

## Dataset

- **File**: `dataset/Salary_Data.csv`
- **Records**: 6,699
- **Features**: Age, Gender, Education Level, Job Title, Years of Experience
- **Target**: Salary

---

## Model

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| R² Score | 0.9784 |
| MAE | ~$4,594 |

---

## Project Structure

```
salary_prediction/
├── dataset/
│   └── Salary_Data.csv
├── app.py              # Streamlit UI
├── train_model.py      # Model training script
├── model.pkl           # Saved model (generated)
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (run once)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

App runs at **http://localhost:8501**

---

## App Features

- Predict salary from Age, Gender, Education Level, Job Title, and Years of Experience
- Gauge chart showing predicted salary vs. dataset range
- Dataset insights: salary distribution, salary by education, top job titles by pay

---

*Developed by [IBM Bob](https://www.ibm.com/products/bob)*
