import mlflow
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "Roy7721"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bc912b7d58bd2aca05abdc192e010414493a3886"

mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")

with mlflow.start_run(run_name="test-connection"):
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 0.99)

print("Run logged successfully!")