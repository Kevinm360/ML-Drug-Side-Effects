import random, datetime as dt
import pandas as pd
from .utils import simple_clean, normalize_drug

DRUGS = [
    "Ibuprofen","Advil","Acetaminophen","Tylenol","Metformin","Glucophage"
]

GI_WORDS = ["nausea","vomiting","diarrhea","constipation","stomach cramps","bloating"]
NEURO_WORDS = ["headache","dizzy","insomnia","fatigue","brain fog","tremor"]
DERM_WORDS = ["rash","itching","hives","redness"]
CARD_WORDS = ["palpitations","chest pain","blood pressure"]
PSY_WORDS  = ["anxiety","depression","irritable","panic"]

POOL = GI_WORDS + NEURO_WORDS + DERM_WORDS + CARD_WORDS + PSY_WORDS

def generate_synthetic_reviews(n=2000, start="2024-01-01", end=None):
    if end is None:
        end = dt.date.today().isoformat()
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    dates = pd.date_range(start_dt, end_dt, freq="D").to_list()
    rows = []
    for i in range(n):
        d = random.choice(DRUGS)
        text_k = random.randint(0, 3)
        toks = random.sample(POOL, k=max(1, text_k))
        text = f"I took {d} and experienced {' and '.join(toks)}. It helped but had side effects."
        rating = random.randint(1,5)
        created_at = random.choice(dates) + pd.Timedelta(hours=random.randint(0,23))
        rows.append({"review_id": i+1, "drug_raw": d, "review_text": text, "rating": rating, "created_at": created_at})
    df = pd.DataFrame(rows)
    df["review_text"] = df["review_text"].map(simple_clean)
    df["drug"] = df["drug_raw"].map(normalize_drug)
    return df

def ingest_to_csv(path_csv: str, n=2000):
    df = generate_synthetic_reviews(n=n)
    df.to_csv(path_csv, index=False)
    return path_csv
