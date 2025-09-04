import os
import duckdb as ddb
import pandas as pd
from prefect import flow, task
from pipeline.ingest import ingest_to_csv
from pipeline.label import distant_supervision
from pipeline.train import load_training, train_model
from pipeline.score import score_new_reviews
from pipeline.signal import compute_signals, persist_to_duckdb

DUCKDB_PATH = "side_effects.duckdb"

@task
def ingest():
    os.makedirs("data/bronze", exist_ok=True)
    path = "data/bronze/reviews.csv"
    return ingest_to_csv(path, n=2000)

@task
def label(path_csv: str):
    df = pd.read_csv(path_csv, parse_dates=["created_at"])
    df = distant_supervision(df)
    os.makedirs("data/silver", exist_ok=True)
    df.to_parquet("data/silver/labels.parquet", index=False)
    return "data/silver/labels.parquet"

@task
def train(ddb_path: str):
    df = load_training(ddb_path)
    model_uri, micro_f1, macro_f1 = train_model(df)
    return model_uri

@task
def score(ddb_path: str, model_uri: str):
    df_scores = score_new_reviews(ddb_path, model_uri)
    return df_scores

@task
def signalize(ddb_path: str, df_scores: pd.DataFrame):
    df_labels = pd.read_parquet("data/silver/labels.parquet")
    df_signals = compute_signals(df_scores)
    persist_to_duckdb(ddb_path, df_labels, df_scores, df_signals)
    return "done"

@flow(name="side_effects_pipeline")
def main():
    ddb_path = DUCKDB_PATH
    csv_path = ingest()
    _ = label(csv_path)
    model_uri = train(ddb_path)
    df_scores = score(ddb_path, model_uri)
    _ = signalize(ddb_path, df_scores)
    print("Flow complete. Data saved in data/ and signals in DuckDB/Parquet.")

if __name__ == "__main__":
    main()
