# app/ml_signal.py
import math
from typing import Optional
try:
    from typing import Iterable
except Exception:
    from collections.abc import Iterable

import pandas as pd
import streamlit as st

from faers_client import (
    top_reactions, total_count,
    reaction_counts_all, reaction_counts_all_serious, reaction_counts_for_drug
)

__all__ = ["score_reactions", "burst_zscores", "severity_metrics"]

if not hasattr(st, "cache_data"):
    st.cache_data = st.cache

_EPS = 1e-12

@st.cache_data(ttl=3600)
def _total_all_reports(start: str, end: str, search_extra: str = "") -> int:
    return total_count(f'receivedate:[{start} TO {end}]' + (search_extra or ""))

@st.cache_data(ttl=3600)
def _total_drug_reports(drug: str, start: str, end: str, search_extra: str = "") -> int:
    q = f'patient.drug.medicinalproduct:"{drug.upper()}" AND receivedate:[{start} TO {end}]'
    if search_extra:
        q += search_extra
    return total_count(q)

def _chi2_2x2(a:int,b:int,c:int,d:int) -> float:
    n = a+b+c+d
    if n <= 0: return 0.0
    rt1, rt2 = a+b, c+d
    ct1, ct2 = a+c, b+d
    def s(x): return max(_EPS, x)
    e11 = s(rt1*ct1/n); e12 = s(rt1*ct2/n); e21 = s(rt2*ct1/n); e22 = s(rt2*ct2/n)
    return (a-e11)**2/e11 + (b-e12)**2/e12 + (c-e21)**2/e21 + (d-e22)**2/e22

_EPS = 1e-12

def _prr(a, b, c, d):
    # proportions
    p1 = a / max(_EPS, (a + b))
    p2 = c / max(_EPS, (c + d))
    # if background event proportion is 0, apply Haldane-Anscombe smoothing
    if p2 == 0.0:
        a, b, c, d = (a + 0.5), (b + 0.5), (c + 0.5), (d + 0.5)
        p1 = a / (a + b)
        p2 = c / (c + d)
    return p1 / max(_EPS, p2)

def _ror(a:int,b:int,c:int,d:int) -> float:
    if min(a,b,c,d) == 0:
        a+=0.5; b+=0.5; c+=0.5; d+=0.5
    return (a*d)/(b*c)

@st.cache_data(ttl=1800)
def score_reactions(
    drug_name: str,
    reactions: Iterable[str],
    start: str,
    end: str,
    a_counts: Optional[dict] = None,
    rx_totals: Optional[dict] = None,
    top_n: int = 100,
    search_extra: str = "",
) -> pd.DataFrame:
    """
    Computes PRR / ROR / χ² for each reaction, respecting the cohort filter (search_extra).
    """
    rx_list = [str(r) for r in reactions]
    if not rx_list:
        return pd.DataFrame(columns=["reaction","a","b","c","d","prr","ror","chi2","signal"])

    if a_counts is None:
        tr = top_reactions(drug_name, start, end, n=top_n, search_extra=search_extra)
        a_map = dict(zip(tr["reaction"].astype(str), tr["reports"].astype(int))) if tr is not None and not tr.empty else {}
    else:
        a_map = {str(k): int(v) for k, v in (a_counts or {}).items()}

    total_all  = _total_all_reports(start, end, search_extra)
    total_drug = _total_drug_reports(drug_name, start, end, search_extra)
    rx_totals_map = rx_totals or reaction_counts_all(start, end, limit=1000, search_extra=search_extra)
    rx_totals_map = {str(k).upper(): int(v) for k, v in rx_totals_map.items()}

    rows = []
    for rx in rx_list:
        a = int(a_map.get(rx, 0))
        rxU = rx.upper()
        rx_total = int(rx_totals_map.get(rxU, 0))

        b = max(0, total_drug - a)
        c = max(0, rx_total - a)
        d = max(0, total_all - a - b - c)

        prr = _prr(a,b,c,d)
        ror = _ror(a,b,c,d)
        chi2 = _chi2_2x2(a,b,c,d)
        signal = (a >= 3) and (prr >= 2.0) and (chi2 >= 4.0)

        rows.append({"reaction": rx, "a": a, "b": b, "c": c, "d": d,
                     "prr": prr, "ror": ror, "chi2": chi2, "signal": signal})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["signal","prr","chi2","a"], ascending=[False, False, False, False]).reset_index(drop=True)

@st.cache_data(ttl=900)
def burst_zscores(ts_df: Optional[pd.DataFrame], window: int = 8, min_periods: int = 5) -> pd.DataFrame:
    if ts_df is None or ts_df.empty or "count" not in ts_df:
        return pd.DataFrame(columns=["date","count","z"])
    df = ts_df.copy().sort_values("date").reset_index(drop=True)
    mu = df["count"].rolling(window=window, min_periods=min_periods).mean()
    sd = df["count"].rolling(window=window, min_periods=min_periods).std().replace(0, float("nan"))
    df["z"] = (df["count"] - mu) / sd
    df["z"] = df["z"].fillna(0.0)
    return df[["date","count","z"]]

def _wald_rr_ci(A: int, B: int, C: int, D: int, z: float = 1.96):
    if min(A, B, C, D) == 0:
        A += 0.5; B += 0.5; C += 0.5; D += 0.5
    p1 = A / max(_EPS, (A + B))
    p2 = C / max(_EPS, (C + D))
    rr = p1 / max(_EPS, p2)
    se = math.sqrt(max(_EPS, (1/A) - (1/(A+B)) + (1/C) - (1/(C+D))))
    lo = math.exp(math.log(rr) - z * se)
    hi = math.exp(math.log(rr) + z * se)
    return rr, lo, hi

@st.cache_data(ttl=1200)
def severity_metrics(
    drug: str, start: str, end: str, top_n: int = 30, search_extra: str = ""
) -> pd.DataFrame:
    """
    For the top N reactions on this drug, compute rate of serious outcomes vs background,
    respecting the cohort filter (search_extra).
    """
    tr = top_reactions(drug, start, end, n=top_n, search_extra=search_extra)
    if tr is None or tr.empty:
        return pd.DataFrame(columns=[
            "reaction","reports","A_serious","B_nonser","C_serious_bg","D_nonser_bg",
            "rate_drug","rate_bg","RR","RR_low","RR_high","chi2","neglog10_p",
            "death","life_threat","hosp","disable","congen","other","severity_score"
        ])
    tr = tr.copy()
    tr["REACTION_U"] = tr["reaction"].astype(str).str.upper()

    rx_total_map   = reaction_counts_all(start, end, limit=1000, search_extra=search_extra)
    rx_serious_map = reaction_counts_all_serious(start, end, limit=1000, search_extra=search_extra)
    rx_total_map   = {k.upper(): int(v) for k, v in rx_total_map.items()}
    rx_serious_map = {k.upper(): int(v) for k, v in rx_serious_map.items()}

    flags = [
        "serious:1",
        "seriousnessdeath:1",
        "seriousnesslifethreatening:1",
        "seriousnesshospitalization:1",
        "seriousnessdisabling:1",
        "seriousnesscongenitalanomali:1",
        "seriousnessother:1",
    ]
    drug_flag_maps = {
        flg: reaction_counts_for_drug(drug, start, end, flag=flg, limit=1000, search_extra=search_extra)
        for flg in flags
    }
    for flg in drug_flag_maps:
        drug_flag_maps[flg] = {k.upper(): int(v) for k, v in drug_flag_maps[flg].items()}

    rows = []
    for _, row in tr.iterrows():
        rx = str(row["reaction"])
        rxU = row["REACTION_U"]
        a_tot = int(row["reports"])
        if a_tot <= 0:
            continue

        a_serious = int(drug_flag_maps["serious:1"].get(rxU, 0))
        death  = int(drug_flag_maps["seriousnessdeath:1"].get(rxU, 0))
        life   = int(drug_flag_maps["seriousnesslifethreatening:1"].get(rxU, 0))
        hosp   = int(drug_flag_maps["seriousnesshospitalization:1"].get(rxU, 0))
        disab  = int(drug_flag_maps["seriousnessdisabling:1"].get(rxU, 0))
        congen = int(drug_flag_maps["seriousnesscongenitalanomali:1"].get(rxU, 0))
        other  = int(drug_flag_maps["seriousnessother:1"].get(rxU, 0))

        rx_total   = int(rx_total_map.get(rxU, 0))
        rx_serious = int(rx_serious_map.get(rxU, 0))

        A = max(0, a_serious)
        B = max(0, a_tot - a_serious)
        C = max(0, rx_serious - a_serious)
        D = max(0, (rx_total - a_tot) - C)

        rr, lo, hi = _wald_rr_ci(A, B, C, D)
        chi2 = _chi2_2x2(A, B, C, D)
        try:
            from math import erfc
            p = max(_EPS, erfc(math.sqrt(chi2/2)))
            neglog10_p = -math.log10(p)
        except Exception:
            neglog10_p = 0.0

        rate_drug = A / max(_EPS, (A+B))
        rate_bg   = C / max(_EPS, (C+D))

        weighted = (3*death + 2.5*life + 2*hosp + 2*disab + 2*congen + 1*other)
        severity_score = 100.0 * (weighted / max(_EPS, a_tot))

        rows.append({
            "reaction": rx,
            "reports": int(a_tot),
            "A_serious": int(A),
            "B_nonser": int(B),
            "C_serious_bg": int(C),
            "D_nonser_bg": int(D),
            "rate_drug": rate_drug,
            "rate_bg": rate_bg,
            "RR": rr, "RR_low": lo, "RR_high": hi,
            "chi2": chi2, "neglog10_p": neglog10_p,
            "death": death, "life_threat": life, "hosp": hosp,
            "disable": disab, "congen": congen, "other": other,
            "severity_score": severity_score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["severity_score", "RR", "chi2"], ascending=[False, False, False]).reset_index(drop=True)
