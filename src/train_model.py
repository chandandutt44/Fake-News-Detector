import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataprep import load_dataset, split
import pandas as pd

if __name__=="__main__":
    df = load_dataset("data/fake_or_real_news.csv")
    X_train, X_test, y_train, y_test = split(df)

    model = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Acuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\\n", cm)

    joblib.dump(model, "model.joblib")
    pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)