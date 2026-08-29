"""Train and score a TF-IDF/logistic-regression baseline."""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

root = Path(__file__).parent
data = pd.read_csv(root / "dataset.csv")
x_train, x_test, y_train, y_test = train_test_split(data.text, data.label, test_size=0.25,
                                                      random_state=42, stratify=data.label)
vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(x_train)
model = LogisticRegression(max_iter=1000).fit(vectorizer.transform(x_train), y_train)
score = f1_score(y_test, model.predict(vectorizer.transform(x_test)), average="macro")
(root / "baseline_f1.txt").write_text(f"macro_f1={score:.4f}\n", encoding="utf-8")
print(f"Macro F1: {score:.4f}")
