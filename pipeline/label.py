import pandas as pd
from .utils import match_taxonomy

def distant_supervision(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["review_text"].apply(match_taxonomy).apply(pd.Series)
    out = pd.concat([df, labels], axis=1)
    return out
