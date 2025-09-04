import duckdb as ddb
import pandas as pd
from .utils import ensure_datetime_week

TARGETS = ["gastrointestinal","neurological","dermatological","cardiovascular","psychiatric"]

def compute_signals(df_scores: pd.DataFrame, threshold_z: float = 2.0):
    df = df_scores.copy()
    df = ensure_datetime_week(df, "created_at")
    long = df.melt(id_vars=["review_id","drug","created_at","week_start"],
                   value_vars=TARGETS, var_name="category", value_name="proba")
    agg = (long.groupby(["drug","category","week_start"], as_index=False)
                .agg(soft_count=("proba","sum")))
    agg["z"] = agg.groupby(["drug","category"])["soft_count"].transform(lambda s: (s - s.mean())/ (s.std(ddof=1)+1e-6))
    agg["is_burst"] = (agg["z"] >= threshold_z).astype(int)
    return agg

def persist_to_duckdb(ddb_path: str, df_labels: pd.DataFrame, df_scores: pd.DataFrame, df_signals: pd.DataFrame):
    con = ddb.connect(ddb_path, read_only=False)
    con.execute("create schema if not exists data")
    df_labels.to_parquet("data/silver/labels.parquet", index=False)
    df_scores.to_parquet("data/gold/scores.parquet", index=False)
    df_signals.to_parquet("data/gold/signals.parquet", index=False)
    con.close()
