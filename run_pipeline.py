import pandas as pd
from pipeline.ingest import ingest_to_csv
from pipeline.label import distant_supervision
from pipeline.train import load_training, train_model
from pipeline.score import score_new_reviews
from pipeline.signal import compute_signals, persist_to_duckdb

DUCKDB_PATH = "side_effects.duckdb"

# 1) Ingest
csv_path = ingest_to_csv("data/bronze/reviews.csv", n=2000)

# 2) Label
df = pd.read_csv(csv_path, parse_dates=["created_at"])
df = distant_supervision(df)
df.to_parquet("data/silver/labels.parquet", index=False)

# 3) Train
train_df = load_training(DUCKDB_PATH)
model_uri, micro_f1, macro_f1 = train_model(train_df)
print(f"Training done. micro_f1={micro_f1:.3f} macro_f1={macro_f1:.3f}")

# 4) Score
df_scores = score_new_reviews(DUCKDB_PATH, model_uri)

# 5) Signals
df_labels = pd.read_parquet("data/silver/labels.parquet")
df_signals = compute_signals(df_scores)
persist_to_duckdb(DUCKDB_PATH, df_labels, df_scores, df_signals)
print("Pipeline complete. Parquet written to data/silver and data/gold.")
