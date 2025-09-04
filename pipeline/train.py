import mlflow
import duckdb as ddb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

TARGETS = ["gastrointestinal","neurological","dermatological","cardiovascular","psychiatric"]

def load_training(ddb_path: str):
    con = ddb.connect(ddb_path, read_only=False)
    df = con.execute("""
        select review_id, drug, review_text, rating, created_at,
               gastrointestinal, neurological, dermatological, cardiovascular, psychiatric
        from 'data/silver/labels.parquet'
    """).df()
    con.close()
    return df

def train_model(df: pd.DataFrame, experiment_name: str = "side_effects_demo"):
    X = df["review_text"]
    Y = df[TARGETS].astype(int)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        pipe.fit(X, Y)
        preds = pipe.predict(X)
        micro_f1 = f1_score(Y, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(Y, preds, average="macro", zero_division=0)

        mlflow.log_metric("micro_f1", micro_f1)
        mlflow.log_metric("macro_f1", macro_f1)

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        report = classification_report(Y, preds, target_names=Y.columns, zero_division=0, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        model_uri = f"runs:/{run.info.run_id}/model"
        return model_uri, micro_f1, macro_f1
