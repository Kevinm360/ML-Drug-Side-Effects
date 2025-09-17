# app/streamlit_ui_overhaul.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import altair as alt
from dataclasses import dataclass
from typing import Optional
import numpy as np

# -------------------------------
# 1) THEME INJECTION (CSS + tweaks)
# -------------------------------
def inject_theme():
    st.set_page_config(page_title="Side-Effects Signal Monitor — FAERS", layout="wide")

    # Global Altair theme
    alt.themes.register("faers_theme", lambda: {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#b8c0d9",
                "titleColor": "#dfe7ff",
                "gridColor": "#1c2230",
                "domainColor": "#2a3142"
            },
            "legend": {"labelColor": "#c8cfe3", "titleColor": "#e6ebff"},
            "range": {
                "category": [
                    "#74d4ff", "#a78bfa", "#ff6ec7", "#5be7a9", "#ffd166",
                    "#4cc9f0", "#e76f51", "#90f7ec", "#f4a261", "#b8f2e6"
                ]
            },
            "font": "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
        }
    })
    alt.themes.enable("faers_theme")

    # Streamlit CSS skin (Fluent / Liquid Glass)
    st.markdown(
        """
        <style>
        :root{
          --bg:#0b0e14;
          --panel:#101521;
          --panel-2:#0f1420;
          --text:#e8ecf7;
          --muted:#a3acc2;
          --line:#1d2536;
          --accent1:#74d4ff;
          --accent2:#a78bfa;
          --accent3:#ff6ec7;
        }

        /* Full gradient background + soft orbs */
        html, body, [data-testid="stAppViewContainer"]{
          background:
            radial-gradient(circle at 10% 20%, rgba(116,212,255,0.15), transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(167,139,250,0.15), transparent 40%),
            linear-gradient(135deg, #0b0e14 0%, #1a1035 50%, #3b0f4c 100%);
        }

        /* Hide Streamlit chrome */
        header[data-testid="stHeader"] {background: transparent;}
        .stDeployButton {filter: drop-shadow(0 2px 6px rgba(0,0,0,.4));}

        /* Sidebar glassy inputs */
        section[data-testid="stSidebar"] > div {background: transparent;}
        section[data-testid="stSidebar"] .block-container{padding: 1.2rem 1rem 2rem;}
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{color:var(--text);}
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stDateInput{
          background: rgba(16,21,33,.55);
          backdrop-filter: blur(10px) saturate(125%);
          -webkit-backdrop-filter: blur(10px) saturate(125%);
          border:1px solid rgba(255,255,255,.10);
          border-radius:14px;
        }

        /* Title */
        .faers-h1{font-size:1.4rem; font-weight:700; color:var(--text); margin: 8px 0 18px;}
        .faers-h1 span{
          background:linear-gradient(90deg, var(--accent1), var(--accent2));
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }

        /* KPI cards */
        .faers-kpi-wrap {display:flex; gap:14px;}
        .faers-kpi{
          background:
            linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)),
            rgba(16,21,33,.55);
          border:1px solid rgba(255,255,255,.12);
          box-shadow: 0 8px 24px rgba(0,0,0,0.25);
          backdrop-filter: blur(14px) saturate(130%);
          -webkit-backdrop-filter: blur(14px) saturate(130%);
          border-radius: 18px; padding:14px 16px;
          display:flex; gap:12px; align-items:center;
        }
        .faers-kpi .n{font-size:1.8rem; font-weight:750;
          background:linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3));
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
        .faers-kpi .l{color:var(--muted); font-size:.9rem;}

        /* Liquid glass cards */
        .faers-card{
          position: relative;
          margin: 10px 0 14px;
          padding: 14px 14px 12px;
          border-radius: 18px;
          background:
            linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)),
            rgba(16,21,33,.55);
          border: 1px solid rgba(255,255,255,.12);
          box-shadow: 0 10px 28px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
          backdrop-filter: blur(14px) saturate(130%);
          -webkit-backdrop-filter: blur(14px) saturate(130%);
        }
        .faers-card::before{
          content:""; position:absolute; inset:0; border-radius:18px; pointer-events:none;
          border: 1px solid transparent;
          background: linear-gradient(140deg,
                    rgba(116,212,255,.45),
                    rgba(167,139,250,.35),
                    rgba(255,110,199,.25)) border-box;
          -webkit-mask: linear-gradient(#000 0 0) padding-box, linear-gradient(#000 0 0);
          -webkit-mask-composite: xor; mask-composite: exclude;
          opacity:.7;
        }
        .faers-card h3{margin:0 0 .4rem 0; font-weight:700; color:var(--text);}

        /* Compact variant */
        .faers-card.faers-card--compact{ padding:10px 12px 10px; border-radius:16px; }
        .faers-card.faers-card--compact h3{ font-size:1.0rem; margin-bottom:.25rem; }

        /* Tabs glass */
        .stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:1px solid var(--line);}
        .stTabs [data-baseweb="tab"]{
          background: rgba(16,21,33,.45);
          border:1px solid rgba(255,255,255,.10);
          border-bottom:none; padding:.45rem .8rem;
          border-top-left-radius:12px; border-top-right-radius:12px;
          backdrop-filter: blur(12px) saturate(130%); -webkit-backdrop-filter: blur(12px) saturate(130%);
          color:var(--muted);
        }
        .stTabs [aria-selected="true"]{
          color:var(--text);
          background: linear-gradient(180deg, rgba(116,212,255,.18), rgba(167,139,250,.12) 60%, rgba(255,110,199,.10));
          border-color: rgba(255,255,255,.18);
        }

        /* Buttons */
        .stButton>button{
          border-radius:12px; border:1px solid var(--line);
          background:linear-gradient(90deg, var(--accent1), var(--accent2));
          color:#0b0e14; font-weight:700;
        }
        .stButton>button:hover{filter:brightness(1.05);}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# 2) HEADER + KPI STRIP
# -------------------------------
def header(title: str, kpis: Optional[dict]=None):
    st.markdown(f"<div class='faers-h1'>💊 <span>{title}</span></div>", unsafe_allow_html=True)
    if kpis:
        cols = st.columns(len(kpis))
        for (label, value), c in zip(kpis.items(), cols):
            with c:
                st.markdown(
                    f"<div class='faers-kpi'><div class='n'>{value:,}</div><div class='l'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

# -------------------------------
# 3) CARD + SECTION HELPERS
# -------------------------------
def card(title: str, body_fn=None, compact: bool=False):
    cls = "faers-card" + (" faers-card--compact" if compact else "")
    with st.container(border=False):
        st.markdown(f"<div class='{cls}'><h3>{title}</h3>", unsafe_allow_html=True)
        if body_fn:
            body_fn()
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 4) READY-MADE CHART RENDERERS
# -------------------------------
def reactions_treemap(df: pd.DataFrame, label_col: str="reaction", size_col: str="reports"):
    if df is None or df.empty:
        st.info("No reaction data available."); return
    d = df[[label_col, size_col]].copy()
    d[size_col] = d[size_col].astype(float)
    chart = (
        alt.Chart(d)
        .mark_circle(opacity=0.9)
        .encode(
            x=alt.X("random():Q", title=None, axis=None),
            y=alt.Y("random():Q", title=None, axis=None),
            size=alt.Size(f"{size_col}:Q", legend=None),
            color=alt.Color(f"{size_col}:Q", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=[alt.Tooltip(label_col, title="Reaction"), alt.Tooltip(size_col, title="Reports", format=",d")],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

def prr_barchart(df: pd.DataFrame, label_col: str="reaction", value_col: str="prr"):
    if df is None or df.empty:
        st.info("No PRR data available."); return
    d = df[[label_col, value_col]].copy()
    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y(f"{label_col}:N", sort="-x", title="Reaction"),
            x=alt.X(f"{value_col}:Q", title="PRR"),
            color=alt.Color(f"{value_col}:Q", scale=alt.Scale(scheme="teals"), legend=None),
            tooltip=[label_col, alt.Tooltip(value_col, format=",.2f")],
        )
        .properties(height=max(240, 16*len(d)))
    )
    st.altair_chart(bars.interactive(), use_container_width=True)

# -------------------------------
# 5) DEMO (optional)
# -------------------------------
@dataclass
class ExampleData:
    total_reports:int=6260
    flagged:int=27
    highest_prr:float=98.21
    weeks:int=104

def demo_page(tr: Optional[pd.DataFrame]=None, prr_df: Optional[pd.DataFrame]=None):
    header(
        "Side-Effects Signal Monitor — FAERS (openFDA only)",
        kpis={"Total reports": ExampleData.total_reports, "Flagged signals": ExampleData.flagged,
              "Highest PRR": ExampleData.highest_prr, "Weeks": ExampleData.weeks}
    )
    tabs = st.tabs(["Overview", "Reactions", "Reports", "Signals"])
    with tabs[1]:
        card("Reactions", lambda: reactions_treemap(tr if tr is not None else _fake_tr()), compact=True)
    with tabs[3]:
        card("Signals", lambda: prr_barchart(prr_df if prr_df is not None else _fake_prr()), compact=True)

# Fake data for preview
def _fake_tr(n: int=24) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({"reaction": [f"REACTION {i+1}" for i in range(n)], "reports": (rng.random(n)*1000+100).astype(int)})
def _fake_prr(n: int=18) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({"reaction": [f"EVENT {i+1}" for i in range(n)], "prr": np.sort(rng.random(n)*100)[::-1]})

# public namespace
class ui:
    inject_theme = staticmethod(inject_theme)
    header = staticmethod(header)
    card = staticmethod(card)
    reactions_treemap = staticmethod(reactions_treemap)
    prr_barchart = staticmethod(prr_barchart)
    demo_page = staticmethod(demo_page)
