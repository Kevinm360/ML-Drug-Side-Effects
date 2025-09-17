# app/nlp_narratives.py
"""
Lightweight NLP on FAERS/MAUDE-like case narratives.

This module expects a DataFrame with a "narrative" column (CSV upload or fetched data).
It extracts keywords, flags clinical phrases, assigns coarse sentiment, and can compute
contrastive weighted log-odds terms between two groups of narratives.

This version:
- Removes FDA redaction markers like "(b)(4)".
- Uses English + device-domain stopwords (as a LIST; scikit-learn expects list/'english'/None).
- Stricter token pattern (>=3 letters) for cleaner terms and better 2–3-gram phrases.
- Extends clinical flags with device/problem language (e.g., "signal loss", "occlusion").
- Adds an optional `custom_stopwords` list in NLPConfig.
- Adds `contrastive_log_odds_terms(...)` for “flagged vs others” phrase discovery.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
)

# -------------------------
# Flags & sentiment lexicon
# -------------------------

CLINICAL_FLAGS = [
    # drug/severity oriented
    "anaphylaxis", "sudden collapse", "cardiac arrest", "arrhythmia", "stroke",
    "seizure", "convulsion", "respiratory failure", "apnea",
    "liver failure", "hepatotoxicity", "renal failure", "kidney failure",
    "gi bleed", "internal bleeding", "hemorrhage",
    "rhabdomyolysis", "neutropenia", "thrombocytopenia", "pancytopenia",
    "stevens-johnson", "toxic epidermal necrolysis", "anuria",
    "syncope", "loss of consciousness", "hypotension", "hypertensive crisis",
    "angioedema", "dystonia",
    # device/problem oriented
    "signal loss", "loss of signal", "loss of connection", "no signal",
    "sensor failure", "sensor error", "inaccurate reading", "false low",
    "erroneous result", "calibration failure", "occlusion",
    "lead fracture", "electrode detachment", "battery failure",
    "overheating", "burn", "migration", "breakage", "fragment",
    "infection", "site infection", "extravasation"
]

# Tiny lexicon for coarse sentiment scoring (transparent & minimal)
POS_WORDS = {
    "improved", "resolved", "better", "stable", "recovering", "asymptomatic", "tolerated"
}
NEG_WORDS = {
    "worse", "severe", "severely", "bad", "badly", "serious", "critical",
    "collapsed", "died", "fatal", "hospitalized", "admitted", "icu", "unconscious",
    "bleeding", "vomiting", "rash", "swollen", "pain", "ache", "paralysis",
    "weakness", "dyspnea", "fainted", "syncope", "chest pain", "shock",
    # device-ish negatives
    "failure", "error", "inaccurate", "occlusion", "fracture", "overheating", "burn"
}

# Extra domain stop-words to suppress MAUDE boilerplate
DOMAIN_STOP = {
    "patient", "device", "reported", "report", "allegation", "complaint", "confirmed",
    "evaluation", "investigation", "customer", "product", "data", "reviewed", "phone",
    "email", "call", "lot", "serial", "model", "return", "analysis", "conclusion",
    "informed", "notified", "representative", "facility", "clinic",
    # very generic in narratives
    "information", "event"
}

# Token regex (lowercase words, allow hyphen/apostrophe, length >=3)
_WORD_RE = re.compile(r"\b[a-z][a-z\-']{2,}\b")

# -------------------------
# Normalization & helpers
# -------------------------

def _normalize(s: str) -> str:
    """Lowercase; remove FDA redaction markers; trim punctuation noise; squeeze spaces."""
    s = s or ""
    s = s.lower()
    # strip FDA redaction markers like "(b)(4)" / "(b)(6)"
    s = re.sub(r"\(b\)\(\d+\)", " ", s)
    # remove most punctuation except word chars, space, hyphen, apostrophe
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clinical_flags(narrative: str) -> List[str]:
    text = _normalize(narrative)
    hits = []
    for phrase in CLINICAL_FLAGS:
        if phrase in text:
            hits.append(phrase)
    return hits

def coarse_sentiment(narrative: str) -> Tuple[int, int, int]:
    """
    Returns (pos_count, neg_count, net_score) with a very small transparent lexicon.
    net_score = pos_count - neg_count.
    """
    text = _normalize(narrative)
    tokens = _WORD_RE.findall(text)
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    return pos, neg, pos - neg

# -------------------------
# Keyword extraction (TF-IDF)
# -------------------------

def _build_stopwords(custom: Optional[List[str]] = None) -> List[str]:
    base = set(ENGLISH_STOP_WORDS).union(DOMAIN_STOP)
    if custom:
        base.update([w.strip().lower() for w in custom if w and w.strip()])
    # scikit-learn expects list or 'english' or None
    return list(sorted(base))

def extract_keywords_tfidf(
    texts: List[str],
    top_k: int = 20,
    ngram_range=(1, 2),
    min_df: int = 2,
    custom_stopwords: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fit a TF-IDF vectorizer and return top-k terms across the corpus.
    Uses English + domain stop-words (plus any custom additions).
    """
    cleaned = [_normalize(x) for x in texts]
    if not any(cleaned):
        return pd.DataFrame(columns=["term", "score"])

    stop = _build_stopwords(custom_stopwords)

    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.9,
        stop_words=stop,
        token_pattern=r"(?u)\b[a-z][a-z\-']{2,}\b",  # words length ≥3
        lowercase=True,
    )
    X = vec.fit_transform(cleaned)
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_k]
    data = [{"term": terms[i], "score": float(scores[i])} for i in order]
    return pd.DataFrame(data)

# -------------------------
# Contrastive (weighted log-odds)
# -------------------------

def contrastive_log_odds_terms(
    pos_texts: List[str],
    neg_texts: List[str],
    ngram_range=(1, 3),
    min_df: int = 2,
    custom_stopwords: Optional[List[str]] = None,
    top_k: int = 15
) -> pd.DataFrame:
    """
    Compute weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008)
    between two corpora. Positive z means term is overrepresented in POS (e.g., flagged)
    vs NEG (e.g., non-flagged).

    Returns DataFrame: term, z, log_odds, c_pos, c_neg
    """
    stop = _build_stopwords(custom_stopwords)

    vec = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stop,
        token_pattern=r"(?u)\b[a-z][a-z\-']{2,}\b",
        lowercase=True,
    )
    texts = [*pos_texts, *neg_texts]
    X = vec.fit_transform([_normalize(t) for t in texts])
    terms = vec.get_feature_names_out()

    n_pos = len(pos_texts)
    X_pos = X[:n_pos, :].sum(axis=0).A1
    X_neg = X[n_pos:, :].sum(axis=0).A1

    # Dirichlet prior from background counts (empirical Bayes)
    alpha = X_pos + X_neg
    alpha0 = alpha.sum()

    def log_odds(counts, n_total):
        num = counts + alpha
        den = (n_total - counts) + (alpha0 - alpha)
        return np.log((num + 1e-12) / (den + 1e-12))

    n1 = X_pos.sum()
    n2 = X_neg.sum()
    lo1 = log_odds(X_pos, n1)
    lo2 = log_odds(X_neg, n2)
    delta = lo1 - lo2

    # Variance approx
    var = (1.0 / (X_pos + alpha)) + (1.0 / ((n1 - X_pos) + (alpha0 - alpha))) \
        + (1.0 / (X_neg + alpha)) + (1.0 / ((n2 - X_neg) + (alpha0 - alpha)))
    z = delta / np.sqrt(var + 1e-12)

    df = pd.DataFrame({
        "term": terms,
        "log_odds": delta,
        "z": z,
        "c_pos": X_pos.astype(int),
        "c_neg": X_neg.astype(int),
    }).sort_values("z", ascending=False)

    return pd.concat([
        df.head(top_k).assign(group="flagged"),
        df.tail(top_k).assign(group="other").iloc[::-1]
    ], axis=0).reset_index(drop=True)

# -------------------------
# Public API
# -------------------------

@dataclass
class NLPConfig:
    top_k_terms: int = 25
    ngram_low: int = 2      # default tuned for phrases
    ngram_high: int = 3     # default tuned for phrases
    min_df: int = 10        # default for ~10k rows
    custom_stopwords: Optional[List[str]] = None

def analyze_narratives(df: pd.DataFrame, narrative_col: str, config: NLPConfig = NLPConfig()) -> Dict[str, pd.DataFrame]:
    """
    Adds columns:
      - clinical_flags: comma-joined string of matched phrases
      - pos_count, neg_count, sentiment_net: ints
    Returns dict with:
      - "annotated": original DF + annotations
      - "top_terms": corpus-level TF-IDF terms
    """
    if narrative_col not in df.columns:
        raise ValueError(f"DataFrame missing required column '{narrative_col}'.")

    # Build annotations
    flags_list = []
    pos_list, neg_list, net_list = [], [], []
    for text in df[narrative_col].astype(str).tolist():
        fl = clinical_flags(text)
        p, n, net = coarse_sentiment(text)
        flags_list.append(", ".join(fl) if fl else "")
        pos_list.append(p); neg_list.append(n); net_list.append(net)

    annotated = df.copy()
    annotated["clinical_flags"] = flags_list
    annotated["pos_count"] = pos_list
    annotated["neg_count"] = neg_list
    annotated["sentiment_net"] = net_list

    # Corpus keywords
    top_terms = extract_keywords_tfidf(
        texts=df[narrative_col].astype(str).tolist(),
        top_k=config.top_k_terms,
        ngram_range=(config.ngram_low, config.ngram_high),
        min_df=config.min_df,
        custom_stopwords=config.custom_stopwords
    )

    return {"annotated": annotated, "top_terms": top_terms}
