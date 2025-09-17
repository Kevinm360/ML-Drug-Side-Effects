# app/network_graph.py
from __future__ import annotations

from typing import Literal, Optional, Tuple
import json
import numpy as np
import pandas as pd

# FAERS helpers from your project
# - top_reactions(drug, start, end, limit=50, search_extra=None) -> DataFrame["reaction","reports"]
# - reaction_counts_all(start, end, limit=1000, search_extra=None) -> DataFrame["reaction","reports"] or dict
# - popular_drugs(start, end, limit=50, search_extra=None) -> DataFrame["drug"] or list[str]
from faers_client import top_reactions as faers_top, reaction_counts_all, popular_drugs


# ---------- data prep ----------
def _assoc_edges_for_drug(
    drug_name: str,
    start: str,
    end: str,
    *,
    search_extra: Optional[str] = None,
    top_n: int = 30,
    min_reports: int = 5,
) -> Tuple[pd.DataFrame, int]:
    """
    Compute simple observed-vs-expected and z for reactions linked to a drug.
    Returns (edges_df, total_observed).
    edges_df columns: reaction, observed, expected, z, weight_reports
    """
    lim = max(int(top_n), 50)
    try:
        df_top = faers_top(drug_name, start, end, lim, search_extra=search_extra)
    except TypeError:
        df_top = faers_top(drug_name)

    if df_top is None or len(df_top) == 0:
        return pd.DataFrame(columns=["reaction", "observed", "expected", "z", "weight_reports"]), 0

    df_top = (
        df_top.rename(columns={"reports": "observed"})
              .sort_values("observed", ascending=False)
    )
    df_top = df_top[df_top["observed"] >= int(min_reports)].head(int(top_n)).copy()

    # Global PT totals across same window -> expected counts baseline
    try:
        df_all = reaction_counts_all(start, end, limit=1000, search_extra=search_extra)
    except TypeError:
        df_all = reaction_counts_all()

    if df_all is None or (isinstance(df_all, pd.DataFrame) and df_all.empty) or (isinstance(df_all, dict) and not df_all):
        df_top["expected"] = 0.0
        df_top["z"] = 0.0
    else:
        if not isinstance(df_all, pd.DataFrame):
            df_all = pd.DataFrame(list(df_all.items()), columns=["reaction", "rx_total"])
        else:
            df_all = df_all.rename(columns={"reports": "rx_total"})
        total_rx_all = float(df_all["rx_total"].sum()) if not df_all.empty else 0.0
        total_rx_for_drug = float(df_top["observed"].sum())

        base = df_all.set_index("reaction")["rx_total"] if not df_all.empty else pd.Series(dtype=float)
        expected = []
        for rx, _row in df_top.set_index("reaction").iterrows():
            rx_total = float(base.get(rx, 0.0))
            e = (rx_total / total_rx_all) * total_rx_for_drug if total_rx_all > 0 else 0.0
            expected.append(e)

        df_top["expected"] = expected
        eps = 1e-9
        df_top["z"] = (df_top["observed"] - df_top["expected"]) / np.sqrt(df_top["expected"] + eps)

    df_top["weight_reports"] = df_top["observed"].astype(float)
    df_top = df_top.reset_index(drop=False)[["reaction", "observed", "expected", "z", "weight_reports"]]
    return df_top, int(df_top["observed"].sum())


# ---------- native vis-network renderer (no PyVis) ----------
def build_reaction_drug_pyvis_html(
    *,
    drug_name: str,
    start: str,
    end: str,
    search_extra: Optional[str] = None,
    top_n: int = 30,
    min_reports: int = 5,
    weight_metric: Literal["reports", "z"] = "z",
    physics_solver: Literal["barnesHut", "forceAtlas2Based", "repulsion"] = "barnesHut",
    charge: int = -2500,
    dark_bg: bool = True,
) -> str:
    """
    Return self-contained HTML for a vis-network graph (inline JS).
    - Color ramp: cyan (low z) → RED (high z)
    - Legend overlay (top-right)
    - High-contrast labels with stroke for readability
    """
    edges_df, total_obs = _assoc_edges_for_drug(
        drug_name, start, end, search_extra=search_extra, top_n=top_n, min_reports=min_reports
    )

    BG   = "#0f172a" if dark_bg else "#ffffff"   # slate-900
    GRID = BG
    TEXT = "#e5e7eb"                             # light gray for labels
    STROKE = BG                                  # outline same as background
    EDGE_BASE = "rgba(255,255,255,0.85)"         # bright edges on dark bg

    if edges_df.empty:
        from html import escape
        return f"<div style='padding:12px;color:{TEXT};background:{BG}'>No edges for <b>{escape(drug_name)}</b> in this window.</div>"

    # ---- sizes & widths ----
    vals = (edges_df["weight_reports"].values.astype(float)
            if weight_metric == "reports"
            else edges_df["z"].clip(lower=0).values.astype(float))
    vmax = float(np.nanmax(vals)) if vals.size else 0.0
    widths = (1.0 + 6.0 * (vals / vmax)) if vmax > 0 else np.ones_like(vals)

    s = np.sqrt(edges_df["observed"].values.astype(float))
    smax = float(s.max()) if s.size else 1.0
    node_sizes = 12.0 + 34.0 * (s / (smax if smax > 0 else 1.0))

    # ---- colors: cyan -> RED for z ----
    def _hex_to_rgb(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _mix_rgba(c1: str, c2: str, t: float, alpha: float = 0.95) -> str:
        r1,g1,b1 = _hex_to_rgb(c1); r2,g2,b2 = _hex_to_rgb(c2)
        r = int(r1 + (r2 - r1) * t); g = int(g1 + (g2 - g1) * t); b = int(b1 + (b2 - b1) * t)
        return f"rgba({r},{g},{b},{alpha})"

    LOW  = "#22d3ee"   # cyan-400 (low z)
    HIGH = "#ef4444"   # red-500  (high z)  <-- changed from orange to RED

    z = edges_df["z"].values.astype(float)
    zpos = np.clip(z, 0, None)
    zmax = float(zpos.max()) if zpos.size else 0.0
    z01 = (zpos / zmax) if zmax > 0 else np.zeros_like(zpos)
    rx_colors = [_mix_rgba(LOW, HIGH, float(t)) for t in z01]

    # ---- nodes & edges ----
    nodes = [{
        "id": f"drug::{drug_name}",
        "label": drug_name,
        "title": f"<b>Drug</b>: {drug_name}<br><i>Total observed links:</i> {total_obs:,}",
        "color": "#f59e0b",  # keep hub amber so it's distinct from red severity
        "size": 40,
        "font": {"color": TEXT, "strokeWidth": 6, "strokeColor": STROKE}
    }]
    for i, row in edges_df.reset_index(drop=True).iterrows():
        rx = str(row["reaction"])
        nodes.append({
            "id": f"rx::{rx}",
            "label": rx,
            "title": (
                f"<b>Reaction</b>: {rx}"
                f"<br><b>Observed</b>: {int(row['observed']):,}"
                f"<br><b>Expected</b>: {float(row['expected']):,.2f}"
                f"<br><b>z-score</b>: {float(row['z']):,.2f}"
            ),
            "color": rx_colors[i],
            "size": float(node_sizes[i]),
            "font": {"color": TEXT, "strokeWidth": 6, "strokeColor": STROKE},
            "shadow": True
        })

    edges = []
    for i, row in edges_df.reset_index(drop=True).iterrows():
        rx = str(row["reaction"])
        edges.append({
            "from": f"drug::{drug_name}",
            "to": f"rx::{rx}",
            "width": float(widths[i]),
            "color": {"color": EDGE_BASE, "highlight": "#ffffff"},
            "title": (
                f"{drug_name} ↔ {rx}"
                f"<br>O={int(row['observed']):,}  E={float(row['expected']):,.2f}  z={float(row['z']):,.2f}"
            )
        })

    opts = {
        "interaction": {"hover": True, "dragNodes": True, "zoomView": True},
        "nodes": {
            "shape": "dot",
            "borderWidth": 1,
            "scaling": {"min": 10, "max": 50},
            "font": {"color": TEXT, "strokeWidth": 6, "strokeColor": STROKE, "size": 16}
        },
        "edges": {"smooth": {"type": "dynamic"}, "scaling": {"min": 1, "max": 12}},
        "physics": {
            "enabled": True,
            "solver": physics_solver,
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "fit": True, "iterations": 250},
            "barnesHut": {
                "gravitationalConstant": int(charge),
                "centralGravity": 0.30, "springLength": 120, "springConstant": 0.03,
                "damping": 0.25, "avoidOverlap": 0.15
            },
            "forceAtlas2Based": {
                "gravitationalConstant": int(charge),
                "centralGravity": 0.01, "springLength": 110, "springConstant": 0.05,
                "damping": 0.40, "avoidOverlap": 0.20
            },
            "repulsion": {
                "nodeDistance": 160,
                "centralGravity": 0.30, "springLength": 140, "springConstant": 0.03,
                "damping": 0.25
            }
        }
    }

    # --- HTML (legend + vis-network) ---
    legend_html = f"""
    <div id="legend" style="
      position:absolute; right:18px; top:18px; z-index:10;
      background:rgba(15,23,42,0.9); color:{TEXT};
      border:1px solid rgba(255,255,255,0.08); border-radius:10px;
      padding:10px 12px; font:12px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      box-shadow:0 4px 14px rgba(0,0,0,0.35);">
      <div style="font-weight:600; margin-bottom:6px;">Legend</div>
      <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#f59e0b;"></span>
        <div>Drug node (center)</div>
      </div>
      <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:{LOW};"></span>
        <div>Reaction — low disproportionality (low z)</div>
      </div>
      <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:{HIGH};"></span>
        <div>Reaction — high disproportionality (high z)</div>
      </div>
      <div style="margin-top:6px; color:#cbd5e1;">
        Node size ≈ √(observed co-occurrences).<br/>
        Edge thickness ≈ {('z-score' if weight_metric == 'z' else 'reports')} (selected in UI).<br/>
        Drag nodes • Wheel to zoom.
      </div>
    </div>"""

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Drug–Reaction Network</title>
  <style>
    html, body {{ margin:0; padding:0; background:{BG}; height:100%; }}
    #net {{ width:100%; height:100vh; background:{GRID}; }}
  </style>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <div id="net"></div>
  {legend_html}
  <script>
    const data = {{
      nodes: new vis.DataSet({json.dumps(nodes)}),
      edges: new vis.DataSet({json.dumps(edges)})
    }};
    const options = {json.dumps(opts)};
    const container = document.getElementById('net');
    const network = new vis.Network(container, data, options);
    network.once('stabilizationIterationsDone', function() {{
      try {{ network.fit({{ animation: true }}); }} catch(e) {{}}
    }});
    setTimeout(() => {{ try {{ network.fit(); }} catch(e) {{}} }}, 300);
  </script>
</body>
</html>"""
    return html


def popular_drug_names(
    start: str,
    end: str,
    limit: int = 50,
    search_extra: Optional[str] = None,
) -> list[str]:
    """Return a list[str] of popular drugs (convenience wrapper)."""
    try:
        df = popular_drugs(start, end, limit=limit, search_extra=search_extra)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return list(df["drug"].dropna().astype(str).unique())
        if isinstance(df, (list, tuple)):
            return [str(x) for x in df]
    except Exception:
        pass
    return []
