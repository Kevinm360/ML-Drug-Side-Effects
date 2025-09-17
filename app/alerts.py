# app/alerts.py
from __future__ import annotations
import os, math, json, smtplib, ssl, time
from email.mime.text import MIMEText
from typing import Iterable, Optional, Dict, List
import pandas as pd

# Reuse your existing modules
from app.faers_client import reaction_counts_for_drug, reaction_counts_all, timeseries, _get_api_key  # type: ignore
from app.ml_signal import score_reactions, burst_zscores

DEFAULTS = {
    "prr_min": 2.0,
    "chi2_min": 4.0,
    "min_reports": 5,
    "z_min": 2.0,             # optional, only if you compute z-bursts
    "top_n_per_drug": 15,
    "use_severity": False,    # flip to True if you want to join severity_scores
    "search_extra": "",       # cohort filter e.g. ' AND patient.patientsex:1'
    "serious_only": False,    # set True to restrict to serious flags
}

def _week_window(now_yyyymmdd: Optional[str] = None) -> tuple[str,str]:
    # last full 7 days window: [start, end]
    import datetime as dt
    today = dt.date.today() if not now_yyyymmdd else dt.datetime.strptime(now_yyyymmdd, "%Y-%m-%d").date()
    end = today.strftime("%Y%m%d")
    start = (today - dt.timedelta(days=7)).strftime("%Y%m%d")
    return start, end

def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return ""

def compute_alerts_for_drug(
    drug: str,
    start: str,
    end: str,
    *,
    prr_min: float,
    chi2_min: float,
    min_reports: int,
    top_n_per_drug: int,
    search_extra: str = "",
    serious_only: bool = False,
    z_min: Optional[float] = None
) -> pd.DataFrame:
    """
    For a given drug and week window [start,end], compute PRR/χ² signals and (optionally) burst z.
    """
    flag = '(serious:1)' if serious_only else None
    # get drug-week reactions observed
    a_map = reaction_counts_for_drug(drug, start, end, flag=flag, limit=1000, search_extra=search_extra) or {}
    rx_list = list(a_map.keys())
    if not rx_list:
        return pd.DataFrame(columns=["drug","reaction","a","b","c","d","prr","chi2","signal","z"])

    # compute disproportionality with your existing helper (fills a,b,c,d,prr,chi2,signal)
    scored = score_reactions(
        drug_name=drug,
        reactions=rx_list,
        start=start, end=end,
        top_n=len(rx_list),
        search_extra=search_extra
    )
    if scored.empty:
        return pd.DataFrame(columns=["drug","reaction","a","b","c","d","prr","chi2","signal","z"])

    # optional: time-burst z on overall drug counts (fast, 1 call)
    z_val = None
    if z_min is not None:
        ts = timeseries(drug, (pd.to_datetime(start)-pd.Timedelta(days=120)).strftime("%Y%m%d"), end, search_extra=search_extra)
        z_df = burst_zscores(ts, window=8, min_periods=5)
        z_val = float(z_df["z"].iloc[-1]) if not z_df.empty else 0.0
        scored["z"] = z_val
    else:
        scored["z"] = float("nan")

    # filter to alert-worthy
    out = (scored
           .query("a >= @min_reports and prr >= @prr_min and chi2 >= @chi2_min")
           .sort_values(["prr","chi2","a"], ascending=[False, False, False])
           .head(top_n_per_drug)
           .copy())
    out.insert(0, "drug", drug)
    # useful human columns
    out["a_bold"] = out["a"].map(int)
    out["prr_txt"] = out["prr"].map(lambda x: f"{x:.2f}")
    out["chi2_txt"] = out["chi2"].map(lambda x: f"{x:.0f}")
    if z_val is not None:
        out["z_txt"] = out["z"].map(lambda x: f"{x:.2f}")
    return out.reset_index(drop=True)

def compute_weekly_alerts(
    drugs: Iterable[str],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    cfg = {**DEFAULTS, **(config or {})}
    if not start or not end:
        start, end = _week_window()

    frames = []
    for d in drugs:
        try:
            frames.append(compute_alerts_for_drug(
                d, start, end,
                prr_min=cfg["prr_min"],
                chi2_min=cfg["chi2_min"],
                min_reports=cfg["min_reports"],
                top_n_per_drug=cfg["top_n_per_drug"],
                search_extra=cfg["search_extra"],
                serious_only=cfg["serious_only"],
                z_min=cfg["z_min"],
            ))
            time.sleep(0.15)  # gentle throttle for openFDA
        except Exception as ex:
            print(f"[warn] {d}: {ex}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    # final rank
    sort_cols = ["z","prr","chi2","a"] if cfg.get("z_min") is not None else ["prr","chi2","a"]
    df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
    return df

# ---------- Notifiers ----------

def send_slack(df: pd.DataFrame, *, webhook_url: str, start: str, end: str) -> None:
    import requests
    if df.empty:
        text = f"✅ No FAERS alerts for {start} → {end}."
        requests.post(webhook_url, json={"text": text}, timeout=20)
        return

    def row_block(r):
        zbit = f" • z={r['z_txt']}" if 'z_txt' in r and r['z_txt'] == r['z_txt'] else ""
        return f"*{r['drug']}* — *{r['reaction']}*  • a={int(r['a'])} • PRR={r['prr_txt']} • χ²={r['chi2_txt']}{zbit}"

    chunks = [row_block(r) for _, r in df.head(20).iterrows()]  # cap reasonable length
    payload = {
        "text": f"🚨 *FAERS Alerts* ({start} → {end})",
        "blocks": [
            {"type":"section","text":{"type":"mrkdwn","text":f"*FAERS Alerts*  `{start} → {end}`"}},
            {"type":"divider"},
            {"type":"section","text":{"type":"mrkdwn","text":"\n".join(chunks) if chunks else "_No hits_"}}
        ]
    }
    requests.post(webhook_url, json=payload, timeout=20)

def send_email(df: pd.DataFrame, *, smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str, to_addr: str, start: str, end: str):
    title = f"FAERS Alerts — {start} → {end}"
    if df.empty:
        body = "No alerts this period."
    else:
        lines = [f"{r.drug} — {r.reaction} | a={int(r.a)} | PRR={r.prr:.2f} | χ²={r.chi2:.0f}" +
                 (f" | z={r.z:.2f}" if 'z' in df.columns and pd.notna(r.z) else "")
                 for r in df.itertuples()]
        body = "\n".join(lines)
    msg = MIMEText(body, "plain")
    msg["Subject"] = title
    msg["From"] = smtp_user
    msg["To"] = to_addr

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx, timeout=30) as s:
        s.login(smtp_user, smtp_pass)
        s.sendmail(smtp_user, [to_addr], msg.as_string())
