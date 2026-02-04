# %%
import pandas as pd
from .constants import *
import re

# %%
## Functions for loading data
def load(csv_path):
    return pd.read_csv(
        csv_path,
        encoding="latin1",
        na_values=["NA", "null"],
        encoding_errors="ignore"
    )

def normalize(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()

def serialize_record(record, fields):
    parts = []
    for field in fields:
        value = normalize(record.get(field, ""))
        parts.append(f"{field}: {value}")
    return "\n".join(parts)
# %%