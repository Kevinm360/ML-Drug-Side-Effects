import re
import pandas as pd

DRUG_NORMALIZATION = {
    "ibuprofen": "Ibuprofen",
    "advil": "Ibuprofen",
    "tylenol": "Acetaminophen",
    "acetaminophen": "Acetaminophen",
    "metformin": "Metformin",
    "glucophage": "Metformin",
}

SIDE_EFFECT_TAXONOMY = {
    "gastrointestinal": ["nausea","vomit","vomiting","diarrhea","constipation","stomach","cramp","abdominal","bloating"],
    "neurological": ["headache","dizzy","dizziness","insomnia","sleep","fatigue","tired","brain fog","tremor"],
    "dermatological": ["rash","itch","itching","hives","redness","swelling","acne"],
    "cardiovascular": ["palpitation","palpitations","tachycardia","bradycardia","chest pain","pressure","bp","blood pressure"],
    "psychiatric": ["anxiety","depress","depression","mood","irritable","panic"]
}

def normalize_drug(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    n = name.strip().lower()
    return DRUG_NORMALIZATION.get(n, name.strip().title())

def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    return t

def match_taxonomy(text: str) -> dict:
    t = text.lower()
    hits = {k:0 for k in SIDE_EFFECT_TAXONOMY}
    for cat, kws in SIDE_EFFECT_TAXONOMY.items():
        for kw in kws:
            if kw in t:
                hits[cat] = 1
                break
    return hits

def ensure_datetime_week(df: pd.DataFrame, col: str = "created_at"):
    df[col] = pd.to_datetime(df[col])
    df["week_start"] = df[col] - pd.to_timedelta(df[col].dt.dayofweek, unit="D")
    df["week_start"] = df["week_start"].dt.normalize()
    return df
