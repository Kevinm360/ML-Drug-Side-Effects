import streamlit as st
import pandas as pd
import duckdb as ddb
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Side-Effects Signal Monitor", layout="wide")

DATA_DIR = Path("data")
SIGNALS = DATA_DIR / "gold" / "signals.parquet"
SCORES  = DATA_DIR / "gold" / "scores.parquet"
LABELS  = DATA_DIR / "silver" / "labels.parquet"

st.title("💊 Side-Effects Signal Monitor")
st.write("Local demo with synthetic data. Run the Prefect flow first to populate data.")

if not SIGNALS.exists():
    st.warning("No signals found. Please run: `python flows/side_effects_flow.py`")
    st.stop()

signals = pd.read_parquet(SIGNALS)
scores  = pd.read_parquet(SCORES)
labels  = pd.read_parquet(LABELS)

drugs = sorted(signals["drug"].unique().tolist())
cats  = sorted(signals["category"].unique().tolist())

with st.sidebar:
    st.header("Filters")
    drug = st.selectbox("Drug", drugs, index=0)
    category = st.selectbox("Side-effect category", cats, index=0)
    z_thresh = st.slider("Burst z-threshold", 1.0, 4.0, 2.0, 0.1)

subset = signals[(signals["drug"]==drug) & (signals["category"]==category)].copy()
subset["burst"] = subset["is_burst"].where(subset["z"]>=z_thresh, 0)

left, right = st.columns([2,1], gap="large")

with left:
    st.subheader(f"Weekly soft counts — {drug} · {category}")
    base = alt.Chart(subset).mark_line().encode(
        x="week_start:T", y=alt.Y("soft_count:Q", title="soft count (sum of probabilities)")
    )
    points = alt.Chart(subset[subset["burst"]==1]).mark_point(size=80, shape="triangle-up").encode(
        x="week_start:T", y="soft_count:Q", tooltip=["week_start","soft_count","z"]
    )
    st.altair_chart(base + points, use_container_width=True)

with right:
    st.subheader("Summary")
    st.metric("Weeks", len(subset))
    st.metric("Bursts", int(subset["burst"].sum()))
    st.dataframe(subset.sort_values("week_start", ascending=False).head(12))

st.markdown("---")
st.subheader("Recent reviews (sample)")
sample = labels[labels["drug"]==drug].sort_values("created_at", ascending=False).head(10)
st.dataframe(sample[["created_at","rating","review_text","gastrointestinal","neurological","dermatological","cardiovascular","psychiatric"]])
