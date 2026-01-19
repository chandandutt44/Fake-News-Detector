import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

STOP = set(stopwords.words('english'))

def clean_text(s):
    if pd.isna(s): return ""
    s = re.sub(r'\s+',' ',s)
    s = re.sub(r'http\S+','',s)
    s = re.sub(r'[^A-Za-z0-9 .,!?]',' ',s)
    s = s.lower()

    tokens = [w for w in s.split() if w not in STOP]
    return " ".join(tokens)
def load_dataset(path="data/fake_or_real_news.csv"):
    df = pd.read_csv(path, encoding="utf-8")
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('')
    else:
        for alt in ['article', 'content', 'body']:
            if alt in df.columns:
                df['text'] = df[alt].fillna('')
                break
        else:
            raise ValueError("merged CSV must contain a 'text' column. Found: " + ", ".join(df.columns))
        
    df['text_clean'] = df['text'].apply(clean_text)
    if 'label' not in df.columns:
        raise ValueError("Merged CSV must contain a 'label' column (values: 'fake'/'real').")
    df['label'] = df['label'].map(lambda x: 1 if str(x).lower() in ['real', 'true', '1'] else 0)
    df = df[['text_clean','label']].dropna().reset_index(drop=True)
    
    return df
def split(df, testsize=0.2, rando_state=42):
    X = df['text_clean']
    y = df['label']
    return train_test_split(X, y, test_size=testsize, random_state=rando_state, stratify=y)

if __name__=="__main__":
    df = load_dataset("data/fake_or_real_news.csv")
    X_train, X_test, y_train, y_test = split(df)
    print("Loaded rows:", len(df))
    print("Train shape:", X_train.shape)
    print(df.head(2))