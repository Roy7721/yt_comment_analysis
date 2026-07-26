import json
import mlflow
from dotenv import load_dotenv

# Creds: locally from .env, in CI from injected secrets. No hardcoding.
load_dotenv()
mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")

# dvc repro rewrites this file each run — it holds the FRESH model's URI.
# (Run from the repo root so this relative path resolves.)
MODEL_INFO_PATH = "reports/model_info.json"
METRICS_PATH = "reports/metrics.json"

def main():
    # Load the model this run just trained — NOT the registered @staging one
    with open(MODEL_INFO_PATH) as f:
        model_uri = json.load(f)["model_uri"]        # e.g. models:/m-728642...

    model = mlflow.pyfunc.load_model(model_uri)

    comments = [
        "this video was absolutely terrible and a complete waste of time",
        "nothing special here",
        "amazing work, i loved every second of this",
    ]
    preds = model.predict(comments)

    # Gate 1: one label per input
    assert len(preds) == len(comments), f"expected {len(comments)} preds, got {len(preds)}"
    # Gate 2: every label is a valid class
    assert all(int(p) in {-1, 0, 1} for p in preds), f"invalid labels: {list(preds)}"

    print(f"✅ pre-promotion gate passed for {model_uri}:", list(preds))

    # Gate 3: quality floor — read the metrics this run produced
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    ACC_FLOOR = 0.80
    assert metrics["accuracy"] >= ACC_FLOOR, \
        f"accuracy {metrics['accuracy']:.4f} below floor {ACC_FLOOR}"

if __name__ == "__main__":
    main()