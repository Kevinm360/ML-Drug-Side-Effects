# app/streamlit_app.py
import streamlit as st

# ⬇️ compatibility shim: if cache_data doesn't exist, alias it to st.cache
if not hasattr(st, "cache_data"):
    st.cache_data = st.cache  # <-- important

import pandas as pd
from datetime import date, timedelta
import io, json, zipfile, requests
from nlp_narratives import analyze_narratives, NLPConfig, contrastive_log_odds_terms

# Hi-DPI crisp charts
import altair as alt
import numpy as np
import plotly.graph_objects as go
import math

# PyVis network (no-CDN inline)
from network_graph import (
    build_reaction_drug_pyvis_html,
    popular_drug_names,  # imported but optional
)

# Web component for custom HTML
from streamlit.components.v1 import html as st_html

# Make sure Python can import modules that live next to this file
# Ensure we import from the local app/ folder
import sys, pathlib, importlib, traceback
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

try:
    ml_signal = importlib.import_module("ml_signal")
    score_reactions = getattr(ml_signal, "score_reactions")
    burst_zscores   = getattr(ml_signal, "burst_zscores")
    severity_metrics = getattr(ml_signal, "severity_metrics")
except Exception as e:
    st.error("Import of ml_signal failed. See details below:")
    st.code("".join(traceback.format_exception(e)))
    raise

from faers_client import (
    timeseries as faers_timeseries,
    top_reactions as faers_top,
    sample_reports as faers_samples,
    popular_drugs as faers_popular,
    reaction_counts_all,  # aggregate PT totals across all drugs
    build_cohort_query,   # ← NEW: use cohort filters everywhere
)

# Small constant for logs/clips used locally here
EPS = 1e-12

st.set_page_config(page_title="Side-Effects Signal Monitor — FAERS", layout="wide")

st.title("💊 Side-Effects Signal Monitor — FAERS (openFDA data only)")
st.caption(
    "Live adverse-event reports via openFDA drug/event API. "
    "Add an `OPENFDA_API_KEY` in Secrets or env to raise limits."
)

# ---------- helpers ----------
def _trigger_fetch():
    """Mark that we need to refetch from the API on the next rerun."""
    st.session_state["fetch"] = True

def _params_tuple():
    # ← UPDATED: include cohort filters so they trigger fetches
    return (
        st.session_state.get("drug_final"),
        st.session_state.get("start"),
        st.session_state.get("end"),
        # Cohort filters (NEW)
        st.session_state.get("age_group"),
        tuple(sorted(st.session_state.get("sexes", []))),
        tuple(sorted(st.session_state.get("reporters", []))),
        # Compare mode (existing)
        st.session_state.get("compare_on"),
        st.session_state.get("drug_b") if st.session_state.get("compare_on") else None,
    )

def _render_svg(chart, use_container_width=True):
    """Render Altair as SVG and inline data so Streamlit always sees it."""
    spec = chart.to_dict()

    # Inline datasets -> data.values (some Streamlit builds ignore spec["datasets"])
    try:
        if isinstance(spec.get("data"), dict) and "name" in spec["data"] and "datasets" in spec:
            name = spec["data"]["name"]
            values = spec["datasets"].get(name)
            if values is not None:
                spec["data"] = {"values": values}
                spec.pop("datasets", None)
    except Exception:
        pass

    # Force SVG renderer
    meta = spec.get("usermeta", {})
    embed = meta.get("embedOptions", {})
    embed["renderer"] = "svg"
    meta["embedOptions"] = embed
    spec["usermeta"] = meta

    st.vega_lite_chart(spec, use_container_width=use_container_width)

def _render_force_graph(sig_df: pd.DataFrame, drug_name: str, height: int = 640):
    """
    Render a Neo4j-like 3D force graph where the center node is the drug,
    and each reaction is a node connected by an edge weighted by co-occurrences (a).
    """
    if sig_df is None or sig_df.empty:
        st.caption("No signals to visualize.")
        return

    # Build graph data
    nodes = [{"id": drug_name, "type": "drug"}]
    links = []

    for _, r in sig_df.iterrows():
        rx = str(r["reaction"])
        a = float(r.get("a", 0) or 0)
        prr = float(r.get("prr", 0) or 0)
        chi2 = float(r.get("chi2", 0) or 0)
        signal = bool(r.get("signal", False))

        nodes.append({
            "id": rx,
            "type": "rx",
            "a": a,
            "prr": prr,
            "chi2": chi2,
            "signal": signal
        })
        links.append({
            "source": drug_name,
            "target": rx,
            "value": a,
            "prr": prr,
            "chi2": chi2,
            "signal": signal
        })

    data = {"nodes": nodes, "links": links}

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    :root {{
      --bg: #0b0f14;
      --text: #e6eef7;
      --drug: #00d1d1;
      --rx:   #7a869a;
      --sig:  #2ca02c;
      --link: rgba(255,255,255,0.28);
    }}
    html, body, #fg {{ margin:0; padding:0; width:100%; height:100%; background:var(--bg); color:var(--text); }}
    .tip {{ font: 12px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    #legend {{
      position: absolute; right: 18px; top: 18px; z-index: 10;
      background: rgba(15,19,26,0.86); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px; padding: 10px 12px; backdrop-filter: blur(4px);
      font: 12px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }}
    #legend h4 {{ margin: 0 0 6px 0; font-size: 12px; letter-spacing: .02em; color: #c7d3e0; }}
    .leg-row {{ display:flex; align-items:center; gap:8px; margin: 4px 0; }}
    .swatch {{ width:14px; height:14px; border-radius:50%; border:1px solid rgba(255,255,255,.25); flex:none; }}
    .sw-drug {{ background: var(--drug); box-shadow: 0 0 8px rgba(0,209,209,.55); }}
    .sw-rx   {{ background: var(--rx); }}
    .sw-sig  {{ background: var(--sig); }}
    .leg-note {{ margin-top: 6px; color:#9fb0c1; }}
    .kbd {{ font: 11px monospace; background:#11151b; padding:2px 6px; border-radius:6px; border:1px solid rgba(255,255,255,.08); }}
  </style>

  <!-- Uses 3d-force-graph + three.js from CDN; if blocked, use the PyVis mode below -->
  <script src="https://unpkg.com/three@0.161.0/build/three.min.js"></script>
  <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
  <div id="fg"></div>

  <div id="legend">
    <h4>Legend</h4>
    <div class="leg-row"><span class="swatch sw-drug"></span><div>Drug node</div></div>
    <div class="leg-row"><span class="swatch sw-sig"></span><div>Flagged reaction (PRR≥2, χ²≥4, a≥3)</div></div>
    <div class="leg-row"><span class="swatch sw-rx"></span><div>Reaction (not flagged)</div></div>
    <div class="leg-note">
      Size ≈ log₁₀(a+1). Drag nodes. Scroll to zoom. Press <span class="kbd">R</span> to re-center.
    </div>
  </div>

  <script>
    const DATA = {json.dumps(data)};
    const elem = document.getElementById('fg');

    function getCSS(v) {{
      return getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    }}

    const Graph = ForceGraph3D({{
      rendererConfig: {{ antialias: true, powerPreference: 'high-performance' }}
    }})(elem)
      .graphData(DATA)
      .backgroundColor(getCSS('--bg'))
      .nodeLabel(n => {{
        const rows = [];
        rows.push(`<b>${{n.id}}</b>`);
        if (n.type === 'rx') {{
          rows.push(`a = ${{(n.a||0).toLocaleString()}}`);
          rows.push(`PRR = ${{(n.prr||0).toFixed(2)}}`);
          rows.push(`χ² = ${{(n.chi2||0).toFixed(0)}}`);
          rows.push(`Flagged = ${{n.signal ? 'Yes' : 'No'}}`);
        }} else {{
          rows.push('Drug node');
        }}
        return `<div class="tip">${{rows.join('<br>')}}</div>`;
      }})
      .nodeColor(n =>
        n.type === 'drug'
          ? getCSS('--drug')
          : (n.signal ? getCSS('--sig') : getCSS('--rx'))
      )
      .linkWidth(l => Math.max(0.6, Math.log10((l.value||0) + 1)))
      .linkColor(() => getCSS('--link'))
      .linkCurvature(0.2)
      .linkDirectionalParticles(l => (l.prr||0) > 2 ? 2 : 1)
      .linkDirectionalParticleColor(l =>
        l.signal ? getCSS('--sig') : getCSS('--rx')
      )
      .linkDirectionalParticleSpeed(l => 0.001 + Math.min(0.01, (l.prr||0) / 600))
      .linkDirectionalParticleWidth(1.2)
      .d3VelocityDecay(0.18);

    const THREE = window.THREE;
    const colDrug = new THREE.Color(getCSS('--drug'));
    const colRx   = new THREE.Color(getCSS('--rx'));
    const colSig  = new THREE.Color(getCSS('--sig'));

    Graph.nodeThreeObject(node => {{
      const r = node.type === 'drug'
        ? 8.2
        : Math.max(2.2, Math.log10((node.a || 0) + 1) * 4.8);

      const geo = new THREE.SphereGeometry(r, 32, 24);
      const color = node.type === 'drug' ? colDrug : (node.signal ? colSig : colRx);
      const mat = new THREE.MeshPhongMaterial({{
        color, shininess: 65, specular: 0x333333,
        emissive: node.type === 'drug' ? 0x003333 : 0x000000
      }});
      const mesh = new THREE.Mesh(geo, mat);

      if (node.type === 'drug') {{
        const halo = new THREE.Mesh(
          new THREE.TorusGeometry(r * 1.25, 0.45, 12, 48),
          new THREE.MeshBasicMaterial({{ color: colDrug, transparent:true, opacity:0.3 }})
        );
        halo.rotation.x = Math.PI / 2;
        mesh.add(halo);
      }}
      return mesh;
    }}).nodeThreeObjectExtend(true);

    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    Graph.scene().add(ambient);
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dir1.position.set(1, 1, 1);
    Graph.scene().add(dir1);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.6);
    dir2.position.set(-1, -0.5, 0.8);
    Graph.scene().add(dir2);

    Graph.d3Force('charge').strength(-75);
    Graph.d3Force('link').distance(l => {{
      const prr = Math.max(0, (l.prr||0));
      return 160 - Math.min(110, Math.log2(1 + prr) * 28);
    }});

    try {{ Graph.renderer().setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); }} catch(e) {{}}
    Graph.cameraPosition({{ z: 650 }});

    window.addEventListener('keydown', e => {{
      if ((e.key || '').toLowerCase() === 'r') {{
        Graph.cameraPosition({{ x:0, y:0, z:650 }});
      }}
    }});
  </script>
</body>
</html>
    """
    st_html(html, height=height)

def _do_fetch():
    """Fetch all data needed for the current selections (and compare drug if set)."""
    params = _params_tuple()
    if st.session_state.get("last_params") == params:
        return

    drug  = st.session_state["drug_final"]
    start = st.session_state["start"]
    end   = st.session_state["end"]

    # ← NEW: Build cohort query once for this fetch
    search_extra = build_cohort_query(
        age_group=(st.session_state.get("age_group") if st.session_state.get("age_group") != "All" else None),
        sexes=st.session_state.get("sexes", []),
        reporters=st.session_state.get("reporters", []),
    )

    # Timeseries
    try:
        ts = faers_timeseries(drug, start, end, search_extra=search_extra)  # ← pass cohort
    except Exception as e:
        st.error(f"Timeseries query failed: {e}")
        ts = None
    st.session_state["ts"] = ts

    # Top reactions
    try:
        tr = faers_top(drug, start, end, 50, search_extra=search_extra)      # ← pass cohort
    except Exception as e:
        st.error(f"Top reactions query failed: {e}")
        tr = None
    st.session_state["tr"] = tr

    # Sample reports
    try:
        sr = faers_samples(drug, 20, search_extra=search_extra)              # ← pass cohort
    except Exception as e:
        st.error(f"Sample reports query failed: {e}")
        sr = None
    st.session_state["sr"] = sr

    # ML: burst z-scores on timeseries
    st.session_state["ts_bursts"] = burst_zscores(ts)

    # ML: disproportionality on top reactions
    if isinstance(tr, pd.DataFrame) and not tr.empty:
        a_map = dict(zip(tr["reaction"].astype(str), tr["reports"].astype(int)))

        # aggregate global PT totals (single call → fewer API hits)
        try:
            rx_totals = reaction_counts_all(start, end, limit=1000, search_extra=search_extra)  # ← pass cohort
        except Exception:
            rx_totals = {}

        st.session_state["signals"] = score_reactions(
            drug,
            tr["reaction"],
            start,
            end,
            a_counts=a_map,
            rx_totals=rx_totals,  # avoids per-PT API calls
            top_n=100,
            search_extra=search_extra,  # ← pass cohort
        )
    else:
        st.session_state["signals"] = pd.DataFrame()

    # Optional compare drug
    st.session_state["ts_b"] = None
    if st.session_state.get("compare_on") and st.session_state.get("drug_b"):
        try:
            st.session_state["ts_b"] = faers_timeseries(
                st.session_state["drug_b"], start, end, search_extra=search_extra  # ← pass cohort
            )
        except Exception as e:
            st.error(f"Compare mode failed: {e}")

    st.session_state["last_params"] = params

    # Bookmark current state in the URL
    try:
        st.query_params.clear()
        st.query_params.update(
            start=st.session_state.get("start",""),
            end=st.session_state.get("end",""),
            drug=st.session_state.get("drug_final",""),
            compare=st.session_state.get("drug_b","") if st.session_state.get("compare_on") else "",
        )
    except Exception:
        pass

# ---------- Sidebar ----------
st.sidebar.header("Filters")

today = date.today()
start_default = (today - timedelta(days=180)).strftime("%Y%m%d")
end_default   = today.strftime("%Y%m%d")

# Auto-fetch toggle (kept for UI, but fetch is now smart and won't hammer the API)
st.sidebar.toggle("Auto-fetch on change", value=True, key="auto_fetch")

# Dates
st.sidebar.text_input(
    "Start (YYYYMMDD)", start_default, key="start",
    on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
)
st.sidebar.text_input(
    "End (YYYYMMDD)", end_default, key="end",
    on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
)

# --- Cohort Explorer (NEW) ---
with st.sidebar.expander("Cohort filters", expanded=True):
    # Age bucket (keeps lucene concise)
    st.selectbox(
        "Age group",
        options=["All", "0–11", "12–17", "18–44", "45–64", "65+"],
        index=0, key="age_group",
        help="Uses patientonsetage in YEARS (openFDA unit=801).",
        on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
    )
    st.multiselect(
        "Sex",
        options=["Male", "Female", "Unknown"],
        default=["Male","Female","Unknown"],
        key="sexes",
        on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
    )
    st.multiselect(
        "Reporter type",
        options=["Consumer", "Physician", "Pharmacist", "Other HCP", "Lawyer"],
        default=["Consumer","Physician","Pharmacist","Other HCP","Lawyer"],
        key="reporters",
        on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
    )

# Drug suggestions (reflect current cohort)
_sx = build_cohort_query(
    age_group=(st.session_state.get("age_group") if st.session_state.get("age_group") != "All" else None),
    sexes=st.session_state.get("sexes"),
    reporters=st.session_state.get("reporters"),
)

st.sidebar.caption("Pick from recent FAERS drugs or type your own")
try:
    suggestions = faers_popular(st.session_state["start"], st.session_state["end"], 200, search_extra=_sx)  # ← pass cohort
except Exception:
    suggestions = ["METFORMIN"]

st.sidebar.selectbox(
    "Drug (medicinal product)",
    options=["METFORMIN"] + [s for s in suggestions if s != "METFORMIN"],
    index=0, key="drug_select",
    on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
)
st.sidebar.text_input(
    "…or type a drug name", key="drug_manual",
    on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
)

# Final drug to use
st.session_state["drug_final"] = (st.session_state.get("drug_manual") or "").strip() or st.session_state["drug_select"]

# Compare mode
with st.sidebar.expander("Compare two drugs"):
    st.checkbox(
        "Enable compare mode", key="compare_on",
        on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
    )
    if st.session_state.get("compare_on"):
        st.text_input(
            "Second drug (optional)", "IBUPROFEN", key="drug_b",
            on_change=_trigger_fetch if st.session_state.get("auto_fetch", True) else None
        )

# ---------- Smart fetch logic ----------
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["fetch"] = True

# Only fetch when filters actually change or when the explicit fetch flag is set
should_fetch = st.session_state.get("fetch", False) or (_params_tuple() != st.session_state.get("last_params"))
if should_fetch:
    with st.spinner("Loading from openFDA…"):
        _do_fetch()
    st.session_state["fetch"] = False

# ---------- Data pulled ----------
ts  = st.session_state.get("ts")
tr  = st.session_state.get("tr")
sr  = st.session_state.get("sr")
tsb = st.session_state.get("ts_b")
sig = st.session_state.get("signals")
tsz = st.session_state.get("ts_bursts")

# ---------- Persistent top navigation (no more tab resets) ----------
TAB_LABELS = ["Overview", "Reactions", "Reports", "Signals", "Severity", "Graph", "Narratives NLP"]

active_tab = st.radio(
    "Sections",
    TAB_LABELS,
    key="active_tab",
    horizontal=True,
    label_visibility="collapsed",
)

# ====== Overview ======
if active_tab == "Overview":
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        k1, k2, k3 = st.columns(3)
        total = int(ts["count"].sum())
        last  = int(ts["count"].iloc[-1])
        delta = int(last - ts["count"].iloc[-2]) if len(ts) > 1 else 0
        k1.metric("Total reports", f"{total:,}")
        k2.metric("Last week", f"{last:,}", delta=delta)
        k3.metric("Weeks", len(ts))

        # --- Timeseries chart inline ---
        drug = st.session_state["drug_final"]
        st.subheader(f"Weekly FAERS reports — {drug}")
        line = (
            alt.Chart(ts)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title="Week"),
                y=alt.Y("count:Q", title="Weekly reports"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("count:Q", format=",")]
            )
            .properties(height=320, width="container")
            .configure_axis(labelFontSize=12, titleFontSize=12)
            .configure_view(strokeWidth=0)
        )
        _render_svg(line)
        st.dataframe(ts.rename(columns={"count": "weekly_reports"}), use_container_width=True, height=220)

        if isinstance(tsz, pd.DataFrame) and not tsz.empty:
            bursts = tsz[tsz["z"] > 3].copy()
            if not bursts.empty:
                st.warning(f"Detected {len(bursts)} burst(s) (z > 3). Showing latest:")
                st.dataframe(bursts[["date", "count", "z"]].tail(8), use_container_width=True)

        # --- Compare overlay inline ---
        if tsb is not None and not tsb.empty:
            drug_b = st.session_state.get("drug_b")
            a = ts.assign(series=drug).rename(columns={"count": "value"})[["date","series","value"]]
            b = tsb.assign(series=drug_b).rename(columns={"count": "value"})[["date","series","value"]]
            tidy = pd.concat([a, b], ignore_index=True)

            st.subheader(f"Weekly reports: {drug} vs {drug_b}")
            comp = (
                alt.Chart(tidy)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title="Week"),
                    y=alt.Y("value:Q", title="Weekly reports"),
                    color=alt.Color("series:N", title="Drug"),
                    tooltip=[alt.Tooltip("date:T"), "series:N", alt.Tooltip("value:Q", format=",")]
                )
                .properties(height=320, width="container")
                .configure_axis(labelFontSize=12, titleFontSize=12)
                .configure_view(strokeWidth=0)
            )
            _render_svg(comp)
    else:
        st.info("Pick a drug and date window to load real FAERS data.")

# ====== Reactions ======
elif active_tab == "Reactions":
    st.subheader("Top reactions")
    if isinstance(tr, pd.DataFrame) and not tr.empty:
        top_n = st.slider(
            "How many reactions to show (Top N)",
            min_value=5,
            max_value=min(50, len(tr)),
            value=min(20, len(tr)),
            step=1,
            help="Adjust to see more/less reactions.",
        )
        df_rx = tr.sort_values("reports", ascending=False).head(top_n).copy()

        import plotly.express as px
        fig = px.treemap(
            df_rx,
            path=["reaction"],
            values="reports",
            color="reports",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=420,
            coloraxis_showscale=False,
        )
        if fig.data:
            fig.data[0].hovertemplate = "<b>%{label}</b><br>Reports: %{value:,}<extra></extra>"

        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.dataframe(df_rx.reset_index(drop=True), use_container_width=True, height=320)
    else:
        st.caption("No reaction data for this window. Try expanding the date range.")

# ====== Reports ======
elif active_tab == "Reports":
    st.subheader("Sample raw reports")
    if isinstance(sr, pd.DataFrame) and not sr.empty:
        st.dataframe(sr, use_container_width=True, height=420)
        st.download_button("Download samples (CSV)", sr.to_csv(index=False), "faers_samples.csv", "text/csv")
    else:
        st.caption("No sample reports returned.")

    # one-click “case bundle” ZIP
    try:
        params = {
            "drug": st.session_state.get("drug_final"),
            "compare_drug": st.session_state.get("drug_b") if st.session_state.get("compare_on") else "",
            "start": st.session_state.get("start"),
            "end": st.session_state.get("end"),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        }
        weekly_df    = st.session_state.get("ts")
        reactions_df = st.session_state.get("tr")
        signals_df   = st.session_state.get("signals")
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("params.json", json.dumps(params, indent=2))
            if isinstance(weekly_df, pd.DataFrame) and not weekly_df.empty:
                z.writestr("weekly_timeseries.csv", weekly_df.to_csv(index=False))
            if isinstance(reactions_df, pd.DataFrame) and not reactions_df.empty:
                z.writestr("top_reactions.csv", reactions_df.to_csv(index=False))
            if isinstance(signals_df, pd.DataFrame) and not signals_df.empty:
                z.writestr("signals_scored.csv", signals_df.to_csv(index=False))
        st.download_button(
            "📦 Download case bundle (ZIP)",
            data=mem.getvalue(),
            file_name=f"{st.session_state.get('drug_final','drug')}_{st.session_state.get('start','')}_{st.session_state.get('end','')}.zip",
            mime="application/zip"
        )
    except Exception as e:
        st.caption(f"Bundle export unavailable: {e}")

# ====== Signals ======
elif active_tab == "Signals":
    st.subheader("Disproportionality signals (PRR / ROR / χ²)")
    st.caption("Flags use a simple MHRA-like rule: PRR≥2, χ²≥4, and a≥3 co-occurrences.")
    if isinstance(sig, pd.DataFrame) and not sig.empty:
        flagged = sig[sig["signal"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged signals", f"{len(flagged)}")
        if not flagged.empty:
            best_prr = flagged.loc[flagged["prr"].idxmax()]
            best_chi = flagged.loc[flagged["chi2"].idxmax()]
            c2.metric("Highest PRR", f'{best_prr["prr"]:.2f}', help=best_prr["reaction"])
            c3.metric("Highest χ²", f'{best_chi["chi2"]:.0f}', help=best_chi["reaction"])

        # Explainer
        with st.expander("How to read this (a/b/c/d, PRR, χ², rule)"):
            st.markdown(
                """
**2×2 table** for a drug–reaction in the selected date window:

|                 | Reaction present | Reaction absent |
|-----------------|------------------|-----------------|
| **Drug present**| **a**            | **b**           |
| **Drug absent** | **c**            | **d**           |

- **PRR** = (a/(a+b)) / (c/(c+d))  
- **ROR** = (a·d)/(b·c)  
- **χ²** from the 2×2 table  
- Flag rule: **PRR ≥ 2**, **χ² ≥ 4**, **a ≥ 3**.  
*Signals are not proof of causality.*
                """
            )

        dfp = sig.copy()
        dfp["log2_prr"] = (dfp["prr"].clip(lower=1e-9)).apply(lambda x: math.log(x, 2))
        # approximate p from chi-square (df=1): p = erfc(sqrt(chi2/2))
        dfp["neglog10_p"] = dfp["chi2"].apply(lambda x: -math.log10(max(1e-12, math.erfc(math.sqrt(max(x, 0)/2**0.0)))))

        top_n = st.slider("Top N flagged to show in bar chart", 5, 30, 12, 1)
        flagged_sorted = dfp[dfp["signal"]].sort_values("prr", ascending=False).head(top_n)
        if not flagged_sorted.empty:
            bar = (
                alt.Chart(flagged_sorted)
                .mark_bar()
                .encode(
                    x=alt.X("prr:Q", title="PRR"),
                    y=alt.Y("reaction:N", sort="-x", title="Flagged reactions"),
                    color=alt.value("#4c78a8"),
                    tooltip=["reaction:N","a:Q","b:Q","c:Q","d:Q","prr:Q","ror:Q","chi2:Q"],
                )
                .properties(height=28 * len(flagged_sorted), width="container")
                .configure_axis(labelFontSize=12, titleFontSize=12)
                .configure_view(strokeWidth=0)
            )
            _render_svg(bar)

        st.markdown("### Scatter: log₂(PRR) vs −log₁₀(p-value)")
        sc1 = (
            alt.Chart(dfp)
            .mark_circle(size=80, opacity=0.85)
            .encode(
                x=alt.X("log2_prr:Q", title="log₂(PRR)"),
                y=alt.Y("neglog10_p:Q", title="−log₁₀(p) from χ²"),
                color=alt.Color("signal:N", scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#999999"]),
                                title="Flagged"),
                tooltip=["reaction:N","a:Q","b:Q","c:Q","prr:Q","chi2:Q"],
            )
            .properties(height=360, width="container")
            .configure_axis(labelFontSize=12, titleFontSize=12)
            .configure_view(strokeWidth=0)
        )
        _render_svg(sc1)

        st.markdown("### Scatter: co-occurrence count (a) vs PRR")
        sc2 = (
            alt.Chart(dfp)
            .mark_circle(size=80, opacity=0.85)
            .encode(
                x=alt.X("a:Q", title="Co-occurrences (a)"),
                y=alt.Y("prr:Q", title="PRR"),
                color=alt.Color("signal:N", scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#999999"]),
                                title="Flagged"),
                tooltip=["reaction:N","a:Q","prr:Q","chi2:Q"],
            )
            .properties(height=320, width="container")
            .configure_axis(labelFontSize=12, titleFontSize=12)
            .configure_view(strokeWidth=0)
        )
        _render_svg(sc2)

        st.markdown("### 3D: signals cube (log₂(PRR), −log₁₀(p), log₁₀(a+1))")
        df3 = dfp.copy()
        x = df3["log2_prr"].astype(float)
        y = df3["neglog10_p"].astype(float)
        z = np.log10(df3["a"].astype(float) + 1.0)
        sizes_raw = df3["a"].astype(float).values
        sizeref = 2.0 * sizes_raw.max() / (40.0 ** 2) if sizes_raw.size and sizes_raw.max() > 0 else 1.0

        fig3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="markers",
                    marker=dict(
                        size=sizes_raw, sizemode="area", sizeref=sizeref, sizemin=3,
                        color=df3["log2_prr"], colorscale="Viridis", showscale=True,
                        opacity=0.95,
                        line=dict(width=0.8, color="rgba(0,0,0,0.6)")
                    ),
                    text=df3["reaction"],
                    customdata=np.stack(
                        [df3["a"].values, df3["prr"].values, df3["chi2"].values, df3["signal"].map({True:"Yes", False:"No"}).values], axis=1
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "log₂(PRR)=%{x:.2f}<br>"
                        "−log₁₀(p)=%{y:.2f}<br>"
                        "log₁₀(a+1)=%{z:.2f}<br>"
                        "a=%{customdata[0]:,} | PRR=%{customdata[1]:.2f} | χ²=%{customdata[2]:.0f}<br>"
                        "Flagged=%{customdata[3]}<extra></extra>"
                    ),
                )
            ]
        )
        fig3d.update_layout(
            height=560,
            font=dict(size=13),
            scene=dict(
                xaxis_title="log₂(PRR) → disproportionality",
                yaxis_title="−log₁₀(p) → significance",
                zaxis_title="log₁₀(a+1) → frequency",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig3d, use_container_width=True, theme=None)

        st.dataframe(
            sig.assign(signal=sig["signal"].map({True: "✅", False: ""}))[
                ["reaction","a","b","c","d","prr","ror","chi2","signal"]
            ].style.format({"prr": "{:.2f}", "ror": "{:.2f}", "chi2": "{:.2f}"}),
            use_container_width=True, height=420
        )
        only = st.checkbox("Show only flagged signals")
        if only:
            st.dataframe(sig[sig["signal"]], use_container_width=True, height=320)
    else:
        st.caption("No signals computed (no top reactions returned).")

# ====== Severity ======
elif active_tab == "Severity":
    drug_name = st.session_state.get("drug_final")
    st.subheader(f"Severity — Serious outcomes & risk (drug: {drug_name})")

    if not drug_name or not st.session_state.get("start") or not st.session_state.get("end"):
        st.info("Pick a drug and date window to compute severity metrics.")
    else:
        c1, c2 = st.columns([1,1])
        top_n = c1.slider("Analyze top N reactions (by pair count)", 5, 50, 20, 1)
        show_ci = c2.toggle("Show 95% CI in table", value=True)

        # ← pass cohort to severity
        sev = severity_metrics(
            drug_name,
            st.session_state["start"],
            st.session_state["end"],
            top_n=top_n,
            search_extra=build_cohort_query(
                age_group=(st.session_state.get("age_group") if st.session_state.get("age_group") != "All" else None),
                sexes=st.session_state.get("sexes"),
                reporters=st.session_state.get("reporters"),
            ),
        )

        if sev is None or sev.empty:
            st.caption("No data available for severity scoring in this window.")
        else:
            # ensure neglog10_p exists if not provided
            if "neglog10_p" not in sev.columns and "chi2" in sev.columns:
                sev["neglog10_p"] = sev["chi2"].apply(
                    lambda x: -math.log10(max(EPS, math.erfc(math.sqrt(max(x, 0)/2**0.0))))
                )

            k1, k2, k3 = st.columns(3)
            k1.metric("Top severity score", f"{sev['severity_score'].max():.1f}")
            k2.metric("Top RR", f"{sev['RR'].max():.2f}")
            k3.metric("Max χ²", f"{sev['chi2'].max():.1f}")

            top_bar = (
                alt.Chart(sev.head(15))
                .mark_bar()
                .encode(
                    x=alt.X("severity_score:Q", title="Severity score (0–100)"),
                    y=alt.Y("reaction:N", sort="-x", title="Reaction"),
                    tooltip=[
                        alt.Tooltip("reaction:N"),
                        alt.Tooltip("reports:Q", title="Pair reports", format=",.0f"),
                        alt.Tooltip("A_serious:Q", title="Serious count", format=",.0f"),
                        alt.Tooltip("RR:Q", title="RR", format=".2f"),
                        alt.Tooltip("chi2:Q", title="χ²", format=",.1f"),
                    ],
                )
            )
            st.altair_chart(top_bar, use_container_width=True)

            sev_v = sev.copy()
            sev_v["log2_rr"] = sev_v["RR"].clip(lower=EPS).apply(lambda x: math.log(x, 2))
            volcano = (
                alt.Chart(sev_v)
                .mark_circle(size=80, opacity=0.85)
                .encode(
                    x=alt.X("log2_rr:Q", title="log2(RR)"),
                    y=alt.Y("neglog10_p:Q", title="-log10(p)"),
                    color=alt.condition(
                        alt.datum.RR >= 2, alt.value("#e15759"), alt.value("#4e79a7")
                    ),
                    tooltip=[
                        alt.Tooltip("reaction:N"),
                        alt.Tooltip("RR:Q", format=".2f"),
                        alt.Tooltip("RR_low:Q", title="RR low", format=".2f"),
                        alt.Tooltip("RR_high:Q", title="RR high", format=".2f"),
                        alt.Tooltip("A_serious:Q", title="Serious (A)", format=",.0f"),
                        alt.Tooltip("B_nonser:Q", title="Not serious (B)", format=",.0f"),
                        alt.Tooltip("C_serious_bg:Q", title="BG serious (C)", format=",.0f"),
                        alt.Tooltip("D_nonser_bg:Q", title="BG not serious (D)", format=",.0f"),
                        alt.Tooltip("chi2:Q", title="χ²", format=",.1f"),
                    ],
                )
            )
            st.altair_chart(volcano, use_container_width=True)

            df_show = sev.copy()
            if not show_ci:
                df_show = df_show.drop(columns=["RR_low","RR_high"], errors="ignore")
            st.dataframe(df_show, use_container_width=True, height=420)

            with st.expander("How this is computed"):
                st.markdown("""
We build a 2×2 table for **serious outcomes (serious:1)** within the selected drug–reaction cohort:

|                 | Serious | Not serious |
|-----------------|---------|-------------|
| **Drug present**| **A**   | **B**       |
| **Other drugs** | **C**   | **D**       |

- **RR** = (A/(A+B)) / (C/(C+D)) with a **95% CI** (Wald; Haldane correction if zeros).
- **χ²** is Pearson’s chi-square on the 2×2 table (df=1); we plot **−log10(p)**.
- **Severity Score (0–100)** = weighted serious subtypes:  
  death×3 + life-threatening×2.5 + hospitalization×2 + disabling×2 + congenital×2 + other×1.  
*Signals are for review, not proof of causality.*
                """)

# ====== Graph ======
elif active_tab == "Graph":
    st.subheader("Interactive Graph: drug ↔ reactions")
    drug_name = st.session_state.get("drug_final", "DRUG")

    # Choose which renderer to use
    view = st.radio("Graph mode", ["3D Force (circles)", "2D PyVis (no-CDN)"], index=0, horizontal=True)

    if view.startswith("3D"):
        st.caption("Drag nodes; higher PRR pulls reactions closer. Node size tracks co-occurrences (a).")
        if isinstance(sig, pd.DataFrame) and not sig.empty:
            c1, c2 = st.columns(2)
            min_a = c1.slider(
                "Minimum co-occurrences (a)", 0, int(sig["a"].max()) if not sig.empty else 0, 0
            )
            min_prr = c2.slider(
                "Minimum PRR", 0.0, float(max(2.0, sig["prr"].max() if not sig.empty else 2.0)), 0.0, 0.1
            )
            df_show = sig[(sig["a"] >= min_a) & (sig["prr"] >= min_prr)].copy()
            _render_force_graph(df_show, drug_name, height=640)
            st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=360)
        else:
            st.caption("No signals to visualize yet.")
    else:
        st.caption("PyVis/vis-network with inline JS (works even if CDNs are blocked).")

        cols = st.columns([2,2,2,2,2])
        top_n = int(cols[0].slider("Top reactions (N)", 10, 100, 30, 5))
        min_reports = int(cols[1].slider("Min reports per edge", 1, 50, 5, 1))
        weight_metric = cols[2].selectbox("Edge thickness by", ["z", "reports"], index=0)
        physics_solver = cols[3].selectbox("Physics solver", ["barnesHut", "forceAtlas2Based", "repulsion"], index=0)
        charge = cols[4].slider("Repulsion (gravitationalConstant)", -8000, -200, -2500, 100)

        sx = build_cohort_query(
            age_group=(st.session_state.get("age_group") if st.session_state.get("age_group") != "All" else None),
            sexes=st.session_state.get("sexes"),
            reporters=st.session_state.get("reporters"),
        )

        html_str = build_reaction_drug_pyvis_html(
            drug_name=drug_name,
            start=st.session_state["start"],
            end=st.session_state["end"],
            search_extra=sx,
            top_n=top_n,
            min_reports=min_reports,
            weight_metric=weight_metric,
            physics_solver=physics_solver,
            charge=charge,
            dark_bg=True,
        )
        st_html(html_str, height=860, scrolling=True)
        st.caption("Tip: drag nodes, use the wheel to zoom, and open the Physics panel to tweak layout.")

# ====== Narratives NLP ======
elif active_tab == "Narratives NLP":
    st.subheader("Narratives NLP — keyword & clinical flagging")
    st.caption(
        "Option A: Upload a CSV with a 'narrative' column.  "
        "Option B: Fetch sample narratives from openFDA bulk device/event (includes event description text)."
    )

    import collections

    USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

    # ---------- Fetch & parse helpers ----------
    def _read_openfda_json_zip_bytes(b: bytes):
        """Yield JSON objects from an openFDA bulk *.json.zip file."""
        with zipfile.ZipFile(io.BytesIO(b), "r") as z:
            names = [n for n in z.namelist() if n.lower().endswith(".json")]
            if not names:
                return
            with z.open(names[0]) as fh:
                txt = fh.read().decode("utf-8", errors="ignore").lstrip()

        # Try full parse
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, list):
                for obj in parsed:
                    if isinstance(obj, dict):
                        yield obj
            elif isinstance(parsed, dict):
                if "results" in parsed and isinstance(parsed["results"], list):
                    for obj in parsed["results"]:
                        if isinstance(obj, dict):
                            yield obj
                else:
                    yield parsed
            return
        except Exception:
            pass

        # Fallback: NDJSON
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj

    @st.cache_data(show_spinner=False)
    def _fetch_openfda_zip(url: str, timeout: int = 120) -> bytes:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.content

    def _build_bulk_url(quarter: str, idx: int, total: int) -> str:
        # Example: https://download.open.fda.gov/device/event/2024q1/device-event-0001-of-0006.json.zip
        return (f"https://download.open.fda.gov/device/event/{quarter}/"
                f"device-event-{idx:04d}-of-{total:04d}.json.zip")

    def _candidate_quarters():
        return ["2025q1", "2024q4", "2024q3", "2024q2", "2024q1", "2023q4", "2023q3"]

    def _candidate_shard_totals():
        return [12, 10, 8, 7, 6, 5, 4, 3, 2, 1]

    def _extract_narrative(obj: dict) -> str:
        # prefer “Describe Event or Problem” text
        desc = (obj.get("event_description_text")
                or obj.get("description_of_event"))
        if desc:
            return str(desc)

        mt = obj.get("mdr_text")
        texts = []
        if isinstance(mt, list):
            # prefer text_type_code D (Describe event/problem)
            for t in mt:
                if isinstance(t, dict) and t.get("text_type_code") == "D" and t.get("text"):
                    texts.append(str(t["text"]))
            if not texts:
                for t in mt:
                    if isinstance(t, dict) and t.get("text"):
                        texts.append(str(t["text"]))
        return " ".join(texts)

    def _extract_id(obj: dict) -> str:
        return str(
            obj.get("report_number")
            or obj.get("mdr_report_key")
            or obj.get("unique_event_id")
            or obj.get("event_key")
            or obj.get("id")
            or ""
        )

    def _extract_codes(obj: dict, key: str) -> str:
        """Collect codes from root and device[] nodes."""
        val = obj.get(key)
        codes = set()
        if isinstance(val, list):
            for v in val:
                if isinstance(v, (str, int)):
                    codes.add(str(v))
        elif isinstance(val, (str, int)):
            codes.add(str(val))

        devs = obj.get("device")
        if isinstance(devs, list):
            for d in devs:
                if isinstance(d, dict):
                    v = d.get(key)
                    if isinstance(v, list):
                        for x in v:
                            if isinstance(x, (str, int)):
                                codes.add(str(x))
                    elif isinstance(v, (str, int)):
                        codes.add(str(v))
        return ";".join(sorted(codes))

    def _fetch_openfda_device_event_narratives(limit: int = 10000,
                                               quarter: str = "auto",
                                               max_files: int = 2) -> pd.DataFrame:
        """
        Pull narrative-like fields + useful metadata from openFDA device/event bulk files.
        Returns DataFrame with: safetyreportid, narrative, manufacturer, device_problem_codes, patient_problem_codes
        """
        rows = []
        qtrs = _candidate_quarters() if quarter == "auto" else [quarter]

        for q in qtrs:
            found_any = False
            for total in _candidate_shard_totals():
                for idx in range(1, max_files + 1):
                    url = _build_bulk_url(q, idx, total)
                    try:
                        content = _fetch_openfda_zip(url)
                    except Exception:
                        continue

                    for obj in _read_openfda_json_zip_bytes(content):
                        if not isinstance(obj, dict):
                            continue
                        desc = _extract_narrative(obj)
                        if not desc:
                            continue
                        rid = _extract_id(obj)
                        manufacturer = str(obj.get("manufacturer_d_name") or "")
                        d_codes = _extract_codes(obj, "device_problem_code")
                        p_codes = _extract_codes(obj, "patient_problem_code")

                        rows.append({
                            "safetyreportid": rid,
                            "narrative": desc,
                            "manufacturer": manufacturer,
                            "device_problem_codes": d_codes,
                            "patient_problem_codes": p_codes,
                        })
                        if len(rows) >= limit:
                            return pd.DataFrame(rows)

                    found_any = True
                if found_any:
                    break
            if rows:
                break

        return pd.DataFrame(rows)

    # ---- UI: choose source ----
    mode = st.radio("Data source", ["Upload CSV", "Fetch openFDA (device/event)"], horizontal=True)

    df = None
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload narratives CSV (must contain a 'narrative' column)", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
    else:
        c = st.columns(3)
        q_choice = c[0].selectbox("Quarter", ["auto", "2025q1", "2024q4", "2024q3", "2024q2", "2024q1", "2023q4", "2023q3"], index=0)
        limit = int(c[1].number_input("Max rows", 1000, 200000, 10000, step=1000))
        files = int(c[2].number_input("Max shards to read", 1, 5, 2, step=1,
                                       help="Read first N shard files of that quarter (faster than scanning all)."))

        if st.button("⬇️ Fetch from openFDA"):
            with st.spinner("Downloading & parsing openFDA device/event…"):
                try:
                    df = _fetch_openfda_device_event_narratives(limit=limit, quarter=q_choice, max_files=files)
                    if df is None or df.empty:
                        st.warning("No narratives found in the probed files. Try a different quarter or increase 'Max shards'.")
                    else:
                        st.success(f"Fetched {len(df):,} narratives from openFDA bulk device/event.")
                        st.caption("Source: download.open.fda.gov/device/event/*/device-event-*.json.zip")
                except Exception as e:
                    st.error(f"Download/parse failed: {e}")

    if df is None:
        st.info("Bring your own CSV **or** click **Fetch from openFDA** for a sample.")
        st.stop()

    # Ensure we have a 'narrative' column
    if "narrative" not in df.columns:
        st.warning("No 'narrative' column detected. Pick which column contains the text:")
        guess = st.selectbox("Narrative column", df.columns.tolist())
        df = df.rename(columns={guess: "narrative"})

    # Optional preview
    with st.expander("Preview input"):
        st.dataframe(df.head(25), use_container_width=True)

    # ---- Controls for NLP ----
    cfg_cols = st.columns([1,1,1,2])
    top_k = cfg_cols[0].number_input("Top-K terms", 10, 200, 25, step=5)
    nlow  = cfg_cols[1].selectbox("Min n-gram", [1,2], index=1)  # default 2
    nhigh = cfg_cols[2].selectbox("Max n-gram", [2,3], index=1)  # default 3
    mindf = cfg_cols[3].number_input("min_df", 1, 50, 10, step=1,
                                     help="Minimum docs a term must appear in (raise to suppress boilerplate).")

    custom_sw = st.text_input("Custom stop-words (comma-separated, optional)",
                              help="Add brand names, boilerplate words, etc., to hide them from TF-IDF.")

    cfg = NLPConfig(
        top_k_terms=int(top_k),
        ngram_low=int(nlow),
        ngram_high=int(nhigh),
        min_df=int(mindf),
        custom_stopwords=[w.strip() for w in custom_sw.split(",")] if custom_sw else None
    )

    # ---- Run analysis ----
    results = analyze_narratives(df, "narrative", cfg)
    annotated = results["annotated"]; top_terms = results["top_terms"]

    # =========================
    # KPIs
    # =========================
    total = len(annotated)
    flagged_mask = annotated["clinical_flags"].str.len() > 0
    n_flagged = int(flagged_mask.sum())
    pos_pct = (n_flagged / total * 100.0) if total else 0.0

    n_manuf_unique = annotated.get("manufacturer").nunique(dropna=True) if "manufacturer" in annotated.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Narratives", f"{total:,}")
    k2.metric("With clinical flags", f"{n_flagged:,}", f"{pos_pct:.1f}%")
    k3.metric("Manufacturers (unique)", f"{n_manuf_unique:,}")
    k4.metric("TF-IDF terms", f"{len(top_terms):,}")

    # =========================
    # Visuals
    # =========================

    # 1) Top TF-IDF phrases (bar)
    st.markdown("### Top phrases (TF-IDF)")
    if not top_terms.empty:
        bar_terms = (
            alt.Chart(top_terms)
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title="TF-IDF (corpus-sum)"),
                y=alt.Y("term:N", sort="-x", title="Term"),
                tooltip=[alt.Tooltip("term:N"), alt.Tooltip("score:Q", format=".1f")],
            )
            .properties(height=min(28*len(top_terms), 560), width="container")
        )
        st.altair_chart(bar_terms, use_container_width=True)
    else:
        st.caption("No terms found — try lowering min_df or including 1-grams.")

    # 2) Clinical flags histogram (top 12)
    st.markdown("### Top clinical flags")
    if n_flagged > 0:
        def _explode_flags(s: pd.Series) -> pd.Series:
            vals = []
            for x in s.fillna(""):
                if not x:
                    continue
                vals.extend([v.strip() for v in x.split(",") if v.strip()])
            return pd.Series(vals)

        flag_counts = _explode_flags(annotated["clinical_flags"]).value_counts().head(12).reset_index()
        flag_counts.columns = ["flag", "count"]
        chart_flags = (
            alt.Chart(flag_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("flag:N", sort="-x", title="Clinical flag")
            )
            .properties(height=min(28*len(flag_counts), 420), width="container")
        )
        st.altair_chart(chart_flags, use_container_width=True)
    else:
        st.caption("No rows matched the clinical flag lexicon yet.")

    # 3) Sentiment distribution
    st.markdown("### Sentiment distribution")
    cats = np.select(
        [annotated["sentiment_net"] > 0, annotated["sentiment_net"] < 0],
        ["positive", "negative"], default="neutral"
    )
    sent_df = pd.DataFrame({"bucket": cats}).value_counts().reset_index(name="count")
    sent_chart = (
        alt.Chart(sent_df)
        .mark_bar()
        .encode(x=alt.X("bucket:N", title="Sentiment"), y=alt.Y("count:Q", title="Narratives"))
        .properties(height=220, width="container")
    )
    st.altair_chart(sent_chart, use_container_width=True)

    # 4) Manufacturers (top 10)
    if "manufacturer" in annotated.columns and annotated["manufacturer"].notna().any():
        st.markdown("### Top manufacturers (by narrative count)")
        man = (annotated["manufacturer"].fillna("").replace("", np.nan).dropna())
        man_counts = man.value_counts().head(10).reset_index()
        man_counts.columns = ["manufacturer", "count"]
        man_chart = (
            alt.Chart(man_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Narratives"),
                y=alt.Y("manufacturer:N", sort="-x", title="Manufacturer")
            )
            .properties(height=min(28*len(man_counts), 360), width="container")
        )
        st.altair_chart(man_chart, use_container_width=True)

    # 5) Contrastive phrases (weighted log-odds)
    st.markdown("### Contrastive phrases (flagged vs others) — weighted log-odds (z-score)")
    try:
        pos_texts = annotated.loc[flagged_mask, "narrative"].astype(str).tolist()
        neg_texts = annotated.loc[~flagged_mask, "narrative"].astype(str).tolist()
        if len(pos_texts) >= 20 and len(neg_texts) >= 20:
            logodds_df = contrastive_log_odds_terms(
                pos_texts, neg_texts,
                ngram_range=(int(nlow), int(nhigh)),
                min_df=int(mindf),
                custom_stopwords=([w.strip() for w in custom_sw.split(",")] if custom_sw else None),
                top_k=12
            )
            logodds_chart = (
                alt.Chart(logodds_df)
                .mark_bar()
                .encode(
                    x=alt.X("z:Q", title="z (weighted log-odds)"),
                    y=alt.Y("term:N", sort="-x", title="Term"),
                    color=alt.Color("group:N", legend=alt.Legend(title="Favored in")),
                    tooltip=["term:N","z:Q","c_pos:Q","c_neg:Q","group:N"]
                )
                .properties(height=min(26*len(logodds_df), 520), width="container")
            )
            st.altair_chart(logodds_chart, use_container_width=True)
        else:
            st.caption("Need at least ~20 flagged and 20 other narratives to compute contrastive terms.")
    except Exception as e:
        st.caption(f"Log-odds computation skipped: {e}")

    # Tables + export
    st.write("**Top terms (TF-IDF across narratives)**")
    st.dataframe(top_terms, use_container_width=True)

    st.write("**Annotated reports**")
    st.dataframe(annotated.head(300), use_container_width=True)

    csv1 = annotated.to_csv(index=False).encode("utf-8")
    csv2 = top_terms.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download annotated.csv", csv1, file_name="annotated_narratives.csv", mime="text/csv")
    st.download_button("⬇️ Download top_terms.csv", csv2, file_name="top_terms.csv", mime="text/csv")
