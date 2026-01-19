import streamlit as st
import pandas as pd
import joblib

st.title("fake News Detector (Student Tool)")

model = joblib.load("model.joblib")

uploaded = st.file_uploader("Upload CSV with 'text' column (optional) or paste text below:", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns and 'text_clean' not in df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        texts = df.get('text', df.get('text_clean'))
        preds = model.predict(texts)
        df['prediction'] = preds
        st.write(df[['text','prediction']].head())
else:
    txt = st.text_area("Paste a news paragraph")
    if st.button("Check"):
        if not  txt.strip():
            st.warning("Paste some text first.")
        else:
            pred = model.predict([txt])[0]
            st.write("Prediction:", "Real" if pred==1 else "Fake")