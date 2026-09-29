import os
import time
import requests
import streamlit as st
from typing import Dict, Any, Optional, Iterable

# Compatibility shim for older Streamlit
if not hasattr(st, "cache_data"):
    st.cache_data = st.cache  # type: ignore[attr-defined]

OPENFDA_BASE = "https://api.fda.gov/drug/event.json"

def _get_api_key() -> Optional[str]:
    try:
        k = st.secrets.get("OPENFDA_API_KEY")  # type: ignore[attr-defined]
        if k:
            return str(k)
    except Exception:
        pass
    return os.getenv("OPENFDA_API_KEY")

# Single session and gentle throttle
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SideEffectsMonitor/1.0 (+streamlit)"})


def _request(params: Dict[str, Any]) -> Dict[str, Any]:
    """HTTP GET with small throttle + retry; treat 404 'no results' as empty results."""
    key = _get_api_key()
    if key:
        params = {**params, "api_key": key}

    time.sleep(0.18)  # gentle throttle

    last = None
    for i in range(5):
        r = _SESSION.get(OPENFDA_BASE, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()

        # openFDA uses 404 for “no matches found” — return an empty result
        if r.status_code == 404:
            try:
                js = r.json()
                err_txt = str(js.get("error", "")) if isinstance(js, dict) else r.text
            except Exception:
                err_txt = r.text
            if "no matches" in err_txt.lower() or "not found" in err_txt.lower():
                return {"results": [], "meta": {"results": {"total": 0}}}

        if r.status_code in (429, 500, 502, 503, 504):
            last = r
            ra = r.headers.get("Retry-After")
            try:
                sleep_for = float(ra) if ra is not None else 0.8 * (i + 1)
            except Exception:
                sleep_for = 0.8 * (i + 1)
            time.sleep(sleep_for)
            continue

        r.raise_for_status()

    if last is not None:
        last.raise_for_status()
    raise RuntimeError("openFDA request failed")

# ----------------------- cohort → query builder -------------------
# Reporter (drug/event): primarysource.qualification (1..5)
REPORTER_CODE = {
    "Physician": "1",
    "Pharmacist": "2",
    "Other HCP": "3",
    "Lawyer": "4",
    "Consumer": "5",
}
# Sex codes: 1=Male, 2=Female, 0=Unknown
SEX_CODE = {"Male": "1", "Female": "2", "Unknown": "0"}

# Age groups in *years* (hi=None means open-ended)
AGE_BUCKETS = {
    "0–11": (0, 11),
    "12–17": (12, 17),
    "18–44": (18, 44),
    "45–64": (45, 64),
    "65+": (65, None),
}

def _or_block(field: str, values: Iterable[str]) -> Optional[str]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return f"{field}:{vals[0]}"
    return "(" + " OR ".join(f"{field}:{v}" for v in vals) + ")"

def _age_block_years(lo: int, hi: Optional[int]) -> str:
    """
    Build an age filter that matches records recorded in YEARS, MONTHS, WEEKS, or DAYS.
    ICSR unit codes: 801=Year, 802=Month, 803=Week, 804=Day.
    """
    # inclusive upper bounds for “hi in years”
    def rng_years():
        return f"(patient.patientonsetageunit:801 AND patient.patientonsetage:[{lo} TO {hi}])"

    def rng_months():
        lo_m = lo * 12
        hi_m = (hi + 1) * 12 - 1 if hi is not None else 2000
        return f"(patient.patientonsetageunit:802 AND patient.patientonsetage:[{lo_m} TO {hi_m}])"

    def rng_weeks():
        lo_w = int(lo * 52)
        hi_w = int((hi + 1) * 52 - 1) if hi is not None else 8000
        return f"(patient.patientonsetageunit:803 AND patient.patientonsetage:[{lo_w} TO {hi_w}])"

    def rng_days():
        lo_d = int(lo * 365.25)
        hi_d = int((hi + 1) * 365.25) - 1 if hi is not None else 200000
        return f"(patient.patientonsetageunit:804 AND patient.patientonsetage:[{lo_d} TO {hi_d}])"

    parts = [rng_years(), rng_months(), rng_weeks(), rng_days()] if hi is not None else [
        f"(patient.patientonsetageunit:801 AND patient.patientonsetage:[{lo} TO 150])",
        f"(patient.patientonsetageunit:802 AND patient.patientonsetage:[{lo*12} TO 2000])",
        f"(patient.patientonsetageunit:803 AND patient.patientonsetage:[{int(lo*52)} TO 8000])",
        f"(patient.patientonsetageunit:804 AND patient.patientonsetage:[{int(lo*365.25)} TO 200000])",
    ]
    return "(" + " OR ".join(parts) + ")"

def build_cohort_query(
    age_group: Optional[str] = None,
    sexes: Optional[Iterable[str]] = None,
    reporters: Optional[Iterable[str]] = None,
) -> str:
    """
    Returns a lucene string like ' AND ...' to append to 'search'.

    - Age: match across YEARS/MONTHS/WEEKS/DAYS (not just years).
    - Sex: patient.patientsex (1=Male, 2=Female, 0=Unknown).
    - Reporter: primarysource.qualification (1..5).
    """
    blocks = []

    # Age
    if age_group and age_group in AGE_BUCKETS:
        lo, hi = AGE_BUCKETS[age_group]
        blocks.append(_age_block_years(int(lo), None if hi is None else int(hi)))

    # Sex
    if sexes:
        sex_codes = [SEX_CODE.get(s) for s in sexes if s in SEX_CODE]
        sblk = _or_block("patient.patientsex", sex_codes)
        if sblk:
            blocks.append(sblk)

    # Reporter
    if reporters:
        r_codes = [REPORTER_CODE.get(r) for r in reporters if r in REPORTER_CODE]
        rblk = _or_block("primarysource.qualification", r_codes)  # fixed field
        if rblk:
            blocks.append(rblk)

    if not blocks:
        return ""  # no filters
    return " AND " + " AND ".join(blocks)

# --------------------------- cached helpers ----------------------
@st.cache_data(ttl=3600)
def total_count(search: str) -> int:
    """Return meta.results.total for a given search (cached)."""
    js = _request({"search": search, "limit": 1})
    return int(js.get("meta", {}).get("results", {}).get("total", 0))

# --------------------------- queries (cohort-aware) --------------
@st.cache_data(ttl=3600)
def top_reactions(drug_name: str, start: Optional[str] = None, end: Optional[str] = None,
                  n: int = 50, search_extra: str = ""):
    if start and end:
        q = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND receivedate:[{start} TO {end}]'
    else:
        q = f'patient.drug.medicinalproduct:"{drug_name.upper()}"'
    if search_extra:
        q += search_extra
    js = _request({"search": q, "count": "patient.reaction.reactionmeddrapt.exact"})
    import pandas as pd
    df = pd.DataFrame(js.get("results", []) or [])
    if df.empty:
        return pd.DataFrame(columns=["reaction", "reports"])
    return df.rename(columns={"term": "reaction", "count": "reports"}).head(n)

@st.cache_data(ttl=3600)
def timeseries(drug_name: str, start: str, end: str, search_extra: str = ""):
    q = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND receivedate:[{start} TO {end}]'
    if search_extra:
        q += search_extra
    js = _request({"search": q, "count": "receivedate"})
    import pandas as pd
    df = pd.DataFrame(js.get("results", []) or [])
    if df.empty:
        return df.assign(date=pd.to_datetime([]), count=[])
    df["date"] = pd.to_datetime(df["time"])
    return df[["date", "count"]].sort_values("date")

@st.cache_data(ttl=3600)
def sample_reports(drug_name: str, n: int = 10, search_extra: str = ""):
    q = f'patient.drug.medicinalproduct:"{drug_name.upper()}"'
    if search_extra:
        q += search_extra
    js = _request({"search": q, "limit": max(1, min(int(n), 1000))})
    rows = []
    for r in js.get("results", []) or []:
        p = r.get("patient", {}) or {}
        rx = p.get("reaction", []) or []
        dr = p.get("drug", []) or []
        ps = r.get("primarysource") or {}
        reporter_qual = ps.get("qualification") if isinstance(ps, dict) else None
        rows.append({
            "safetyreportid": r.get("safetyreportid"),
            "receivedate": r.get("receivedate"),
            "drug_name": (dr[0] or {}).get("medicinalproduct") if dr else None,
            "reactions": ", ".join(x.get("reactionmeddrapt", "") for x in rx if x.get("reactionmeddrapt")),
            "sex": p.get("patientsex"),
            "age": p.get("patientonsetage"),
            "reporteroccupation": reporter_qual,  # legacy column name, correct source
        })
    import pandas as pd
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def popular_drugs(start: str, end: str, n: int = 200, search_extra: str = ""):
    q = f"receivedate:[{start} TO {end}]"
    if search_extra:
        q += search_extra
    js = _request({"search": q, "count": "patient.drug.medicinalproduct.exact"})
    import pandas as pd
    df = pd.DataFrame(js.get("results", []) or [])
    if df.empty:
        return ["METFORMIN"]
    return [t for t in df["term"].head(n).tolist() if isinstance(t, str)]

@st.cache_data(ttl=3600)
def reaction_counts_all(start: str, end: str, limit: int = 1000, search_extra: str = "") -> Dict[str, int]:
    q = f"receivedate:[{start} TO {end}]"
    if search_extra:
        q += search_extra
    js = _request({
        "search": q,
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": max(1, min(int(limit), 1000)),
    })
    results = js.get("results", []) or []
    return {row.get("term"): int(row.get("count", 0)) for row in results if row.get("term")}

@st.cache_data(ttl=3600)
def reaction_counts_all_serious(start: str, end: str, limit: int = 1000, search_extra: str = "") -> Dict[str, int]:
    q = f"serious:1 AND receivedate:[{start} TO {end}]"
    if search_extra:
        q += search_extra
    js = _request({
        "search": q,
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": max(1, min(int(limit), 1000)),
    })
    results = js.get("results", []) or []
    return {(row.get("term") or "").upper(): int(row.get("count", 0))
            for row in results if row.get("term")}

@st.cache_data(ttl=3600)
def reaction_counts_for_drug(drug_name: str, start: str, end: str,
                             flag: Optional[str] = None, limit: int = 1000,
                             search_extra: str = "") -> Dict[str, int]:
    base = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND receivedate:[{start} TO {end}]'
    if flag:
        base = f"{base} AND {flag}"
    if search_extra:
        base += search_extra
    js = _request({
        "search": base,
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": max(1, min(int(limit), 1000)),
    })
    results = js.get("results", []) or []
    return {(row.get("term") or "").upper(): int(row.get("count", 0))
            for row in results if row.get("term")}


# ---------------- report-level seriousness classification data ----------------
REPORT_LEVEL_COLUMNS = [
    "safetyreportid",
    "receivedate",
    "serious",
    "patient_age_years",
    "patient_weight_kg",
    "patient_sex",
    "reporter_qualification",
    "reporter_country",
    "primary_source_country",
    "occurrence_country",
    "report_type",
    "drug_count",
    "reaction_count",
    "primary_suspect_drug_count",
    "concomitant_drug_count",
    "interacting_drug_count",
    "subject_drug_role",
    "subject_drug_route",
]


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_in_years(value: Any, unit: Any) -> Optional[float]:
    """Convert ICH age-unit codes to years without using any outcome field."""
    age = _as_float(value)
    if age is None or age < 0:
        return None
    factors = {
        "800": 10.0,       # decade
        "801": 1.0,        # year
        "802": 1.0 / 12.0, # month
        "803": 1.0 / 52.0, # week
        "804": 1.0 / 365.25, # day
        "805": 1.0 / (365.25 * 24.0), # hour
    }
    factor = factors.get(str(unit))
    return age * factor if factor is not None else None


def _subject_drug(drugs: list, drug_name: str) -> Dict[str, Any]:
    """Return the queried drug entry when possible, otherwise the first drug entry."""
    wanted = str(drug_name or "").strip().casefold()
    for drug in drugs:
        if not isinstance(drug, dict):
            continue
        product = str(drug.get("medicinalproduct") or "").casefold()
        if wanted and wanted in product:
            return drug
    return next((d for d in drugs if isinstance(d, dict)), {})


def _report_level_row(report: Dict[str, Any], drug_name: str) -> Optional[Dict[str, Any]]:
    """Flatten one real FAERS report into a leakage-controlled modeling row."""
    serious_code = str(report.get("serious") or "")
    if serious_code not in {"1", "2"}:
        return None

    patient = report.get("patient") if isinstance(report.get("patient"), dict) else {}
    primary_source = (
        report.get("primarysource") if isinstance(report.get("primarysource"), dict) else {}
    )
    drugs = patient.get("drug") if isinstance(patient.get("drug"), list) else []
    reactions = patient.get("reaction") if isinstance(patient.get("reaction"), list) else []
    subject_drug = _subject_drug(drugs, drug_name)
    roles = [str(d.get("drugcharacterization") or "") for d in drugs if isinstance(d, dict)]

    # Only the binary target is retained from the outcome fields. In particular,
    # seriousness subtypes, reaction outcomes, patient death, and the existing
    # severity score are never copied into this report-level table.
    return {
        "safetyreportid": report.get("safetyreportid"),
        "receivedate": report.get("receivedate"),
        "serious": 1 if serious_code == "1" else 0,
        "patient_age_years": _age_in_years(
            patient.get("patientonsetage"), patient.get("patientonsetageunit")
        ),
        "patient_weight_kg": _as_float(patient.get("patientweight")),
        "patient_sex": patient.get("patientsex"),
        "reporter_qualification": primary_source.get("qualification"),
        "reporter_country": primary_source.get("reportercountry"),
        "primary_source_country": report.get("primarysourcecountry"),
        "occurrence_country": report.get("occurcountry"),
        "report_type": report.get("reporttype"),
        "drug_count": len(drugs),
        "reaction_count": len(reactions),
        "primary_suspect_drug_count": roles.count("1"),
        "concomitant_drug_count": roles.count("2"),
        "interacting_drug_count": roles.count("3"),
        "subject_drug_role": subject_drug.get("drugcharacterization"),
        "subject_drug_route": subject_drug.get("drugadministrationroute"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def report_level_reports(
    drug_name: str,
    start: str,
    end: str,
    max_records: int = 3000,
    search_extra: str = "",
):
    """
    Fetch real report-level FAERS records for seriousness classification.

    The capped sample is distributed across the requested date window instead
    of taking only the earliest API page, then ordered by receipt date.
    The returned table contains the target plus a deliberately small set of
    non-outcome candidate predictors; no seriousness subtype is returned.
    """
    import pandas as pd

    requested = max(1, min(int(max_records), 5000))
    rows = []
    start_date = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
    end_date = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        raise ValueError("start and end must be valid YYYYMMDD dates with start <= end")

    span_days = int((end_date - start_date).days) + 1
    # Up to ten temporal buckets prevents a capped result from representing only
    # the beginning of a long date range. Each bucket stays below the API's
    # 1,000-record page maximum.
    bucket_count = min(10, span_days, requested)
    base_allocation, extra = divmod(requested, bucket_count)

    for bucket in range(bucket_count):
        bucket_start = start_date + pd.Timedelta(days=(span_days * bucket) // bucket_count)
        bucket_end = start_date + pd.Timedelta(days=(span_days * (bucket + 1)) // bucket_count - 1)
        allocation = base_allocation + (1 if bucket < extra else 0)
        query = (
            f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND '
            f'receivedate:[{bucket_start:%Y%m%d} TO {bucket_end:%Y%m%d}]'
        )
        if search_extra:
            query += search_extra
        for bucket_skip in range(0, allocation, 1000):
            page_size = min(1000, allocation - bucket_skip)
            payload = _request({
                "search": query,
                "sort": "receivedate:asc",
                "limit": page_size,
                "skip": bucket_skip,
            })
            reports = payload.get("results", []) or []
            for report in reports:
                if isinstance(report, dict):
                    row = _report_level_row(report, drug_name)
                    if row is not None:
                        rows.append(row)
            if len(reports) < page_size:
                break

    if not rows:
        return pd.DataFrame(columns=REPORT_LEVEL_COLUMNS)

    frame = pd.DataFrame(rows, columns=REPORT_LEVEL_COLUMNS)
    frame["receivedate"] = pd.to_datetime(
        frame["receivedate"], format="%Y%m%d", errors="coerce"
    )
    frame = frame.drop_duplicates(subset=["safetyreportid"], keep="last")
    return frame.sort_values(["receivedate", "safetyreportid"], na_position="last").reset_index(drop=True)
