import pandas as pd
from pathlib import Path

DATA_DIR = Path("C:\\Users\\chand\\OneDrive\\Documents\\data")
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
OUT_PATH = DATA_DIR / "fake_or_real_news.csv"

def load_and_prepare(path, label_name):
    df = pd.read_csv(path)
    title = df.columns[df.columns.str.lower() == "titel"]
    text = df.columns[df.columns.str.lower() == "text"]
    tcol = title[0] if len(title)>0 else ("title" if "title" in df.columns else None)
    txtcol = text[0] if len(text)>0 else ("text" if "text" in df.columns else None)

    if tcol and txtcol:
        df["text"] = df[tcol].fillna("") + ". "+df[txtcol].fillna("")
    elif txtcol:
        df["text"] = df[txtcol].fillna("")
    elif tcol:
        df["text"] = df[tcol].fillna("")
    else:
        str_cols = df.select_dtypes(inclde="object").columns
        if len(str_cols) == 0:
            raise ValueError(f"No text_like columns found in {path}")
        df ["text"] = df[str_cols[0]].fillna("")
    
    df = df[["text"]].copy()
    df["label"] = label_name
    return df
def main():
    fake = load_and_prepare(FAKE_PATH, "fake")
    real = load_and_prepare(TRUE_PATH, "real")

    combined = pd.concat([fake, real], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Class counts:\n", combined["label"].value_counts())

    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved combined dataset to: {OUT_PATH} (rows: {len(combined)})")

if __name__=="__main__":
    main()