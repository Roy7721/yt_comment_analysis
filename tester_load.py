import os
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"] = "Roy7721"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bc912b7d58bd2aca05abdc192e010414493a3886"
mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")

# Load the model straight from the registry by its alias
model = mlflow.pyfunc.load_model("models:/yt_chrome_plugin_model#4@staging")

comments = [
    "this video was absolutely terrible and a complete waste of time",
    "it was okay, nothing special really",
    "amazing work, i loved every second of this",
]
print(model.predict(comments))