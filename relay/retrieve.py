from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from relay.data import corpus


def index(rows=None):
    rows = rows if rows is not None else corpus()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform([r["customer"] for r in rows])
    return rows, vectorizer, matrix


def similar(text, k=5, built=None):
    rows, vectorizer, matrix = built if built else index()
    scores = linear_kernel(vectorizer.transform([text]), matrix)[0]
    top = sorted(range(len(scores)), key=lambda i: (-scores[i], rows[i]["id"]))[:k]
    return [
        {
            "id": rows[i]["id"],
            "customer": rows[i]["customer"],
            "brand_reply": rows[i]["brand_reply"],
            "score": round(float(scores[i]), 4),
        }
        for i in top
    ]
