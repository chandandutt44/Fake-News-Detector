# Fake News Detection Using Machine Learning

## Overview
Lightweight fake news detector using TF-IDF + Logistic Regression and a streamlit UI.

## Structure
- `src/` : merging datasets, data prep, training
- `app/` : Streamlit app
- `data/` : fake_or_real_news.csv
- `confusuion matrix.csv/`: confusion matrix

## How to run (local)
1. Create venv and activate:
    - `python -m venv venv`
    - `venv\Scripts\activate` (Windows)
2. Install:
    - `pip install -r requirements.txt`
3. Prepare data:
    - Download dataset
4. Train:
    - `python src/train_model.py`
5. Run app:
    - `streamlit run app/streamlit_app.py`

## Deployment
- Deploy to Dtreamlit Community Cloud or Hugging Face Spaces.
