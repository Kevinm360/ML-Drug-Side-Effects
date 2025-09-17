# app/pages/5_Narratives_NLP.py
import streamlit as st
import pandas as pd
import io, os, zipfile, tempfile, requests
import sys, pathlib

# Ensure we can import sibling app modules (e.g., nlp_narratives.py)
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from nlp_narratives import analyze_narratives, NLPConfig

st.set_page_config(page_title="Narratives NLP — FAERS/MAUDE", layout="wide")
st.title("🧪 Narratives NLP — keyword & clinical flagging")

st.caption(
    "Option A: Upload a CSV with a **narrative** column.  "
    "Option B: Fetch a sample from **FDA MAUDE** Text Data (device event narratives).  "
    "Note: FAERS case narratives are not provided via openFDA; VAERS CSVs require a human CAPTCHA."
)

# ------------------------
# Helpers for MAUDE fetch
# ------------------------
def _guess_narrative_col(df: pd.DataFrame, sample_rows: int = 1000) -> int:
    #\"\"\"Pick the column with the largest median string length across a sample.\"\"\"
    sample = df.head(sample_rows).astype(str)
    med_lens = sample.apply(lambda s: s.str.len().median())
    return int(med_lens.fillna(0).astype(float).idxmax())

@st.cache_data(show_spinner=False)
def fetch_maude_zip(source: str) -> bytes:
    base = "https://www.accessdata.fda.gov/MAUDE/ftparea"
    url = f"{base}/foitextadd.zip" if source == "current" else f"{base}/foitext{source}.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def fetch_maude_narratives(limit: int = 10000, source: str = "current") -> pd.DataFrame:
    #\"\"\"Download MAUDE Text Data zip and parse narratives to a DataFrame.\"\"\"
    content = fetch_maude_zip(source)
    with zipfile.ZipFile(io.BytesIO(content), "r") as z:
        txt_name = next((n for n in z.namelist() if n.lower().endswith(".txt")), None)
        if not txt_name:
            raise RuntimeError("No .txt file found inside the FDA foitext zip.")
        with z.open(txt_name) as fh:
            # MAUDE text files are pipe-delimited, typically no header
            df = pd.read_csv(
                fh,
                sep="|",
                header=None,
                dtype=str,
                engine="python",
                on_bad_lines="skip"
            )
    narr_col = _guess_narrative_col(df)
    id_col = 0 if 0 in df.columns else df.columns[0]
    out = pd.DataFrame({
        "safetyreportid": df[id_col].astype(str).str.strip(),
        "narrative": df[narr_col].astype(str).str.strip()
    })
    out = out[out["narrative"].str.len() > 0].dropna().head(limit).reset_index(drop=True)
    return out

# ------------------------
# UI
# ------------------------
mode = st.radio("Data source", ["Upload CSV", "Fetch FDA (MAUDE) sample"], horizontal=True)

df = None
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload narratives CSV (must contain a 'narrative' column)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
else:
    src_col = st.columns(3)
    source = src_col[0].selectbox("Which file?", ["current (this month, smaller)", "2024", "2023", "2019"], index=0)
    source_val = "current" if source.startswith("current") else source
    limit = int(src_col[1].number_input("Max rows", 1000, 200000, 10000, step=1000))
    if src_col[2].button("⬇️ Fetch from FDA"):
        with st.spinner("Downloading and parsing FDA MAUDE narratives..."):
            try:
                df = fetch_maude_narratives(limit=limit, source=source_val)
                st.success(f"Fetched {len(df):,} narratives from FDA MAUDE ({source}).")
                st.caption("Source: FDA MAUDE Text Data — foitext*.zip")
            except Exception as e:
                st.error(f"Download/parse failed: {e}")

if df is None:
    st.info(
        "Bring your own CSV **or** click **Fetch from FDA** to try a sample.\n\n"
        "Tips:\n"
        "• Your CSV should include a 'narrative' column (string).\n"
        "• Optional columns like 'safetyreportid', 'drug_name', 'sex', 'age' are preserved if present.\n"
        "• VAERS CSVs require a CAPTCHA on the official site; MAUDE text files are directly downloadable."
    )
    st.stop()

# Ensure we have a 'narrative' column
if "narrative" not in df.columns:
    st.warning("No 'narrative' column detected. Pick which column contains the text:")
    guess = st.selectbox("Narrative column", df.columns.tolist())
    df = df.rename(columns={guess: "narrative"})

# Optional preview
with st.expander("Preview input"):
    st.dataframe(df.head(25), use_container_width=True)

# ------------------------
# Analysis
# ------------------------
cfg_cols = st.columns(4)
top_k = cfg_cols[0].number_input("Top-K terms", 10, 200, 25, step=5)
nlow  = cfg_cols[1].selectbox("Min n-gram", [1,2], index=0)
nhigh = cfg_cols[2].selectbox("Max n-gram", [1,2,3], index=1)
mindf = cfg_cols[3].number_input("min_df", 1, 20, 2, step=1)

cfg = NLPConfig(top_k_terms=int(top_k), ngram_low=int(nlow), ngram_high=int(nhigh), min_df=int(mindf))

results = analyze_narratives(df, "narrative", cfg)
annotated = results["annotated"]
top_terms = results["top_terms"]

st.subheader("Top terms (TF‑IDF across narratives)")
st.dataframe(top_terms, use_container_width=True)

st.subheader("Annotated reports")
st.dataframe(annotated.head(300), use_container_width=True)

csv1 = annotated.to_csv(index=False).encode("utf-8")
csv2 = top_terms.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download annotated.csv", csv1, file_name="annotated_narratives.csv", mime="text/csv")
st.download_button("⬇️ Download top_terms.csv", csv2, file_name="top_terms.csv", mime="text/csv")

st.caption("This page uses a lightweight, transparent NLP pipeline (lexicon sentiment + TF‑IDF keywords).")
