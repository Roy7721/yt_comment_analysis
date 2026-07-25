from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def predict_sentiment(comments: list[str]) -> list[dict]:
    results = classifier(comments)
    return [{"text": c, "label": r["label"], "score": r["score"]} for c, r in zip(comments, results)]   