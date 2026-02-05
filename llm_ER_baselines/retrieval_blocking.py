# %%
import time
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

from src.loader import load, serialize_record
from src.constants import *
from src.blocker import vectorizer   # only using vectorizer, NOT blocking

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
load_dotenv()
random.seed(42)
np.random.seed(42)

TOP_K = 50
start = time.perf_counter()

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
amazon_df = load(AMAZON_PATH)
amazon_df["serialized"] = amazon_df.apply(
    lambda r: serialize_record(r, AMAZON_FIELDS), axis=1
)

google_df = load(GOOGLE_PATH)
google_df["serialized"] = google_df.apply(
    lambda r: serialize_record(r, GOOGLE_FIELDS), axis=1
)

gt_df = pd.read_csv(GT_PATH)

# -------------------------------------------------------------------
# TF-IDF (fit ONCE on Google)
# -------------------------------------------------------------------
google_tfidf = vectorizer.fit_transform(google_df["serialized"])

# -------------------------------------------------------------------
# Candidate generation (SIMILARITY ONLY)
# -------------------------------------------------------------------
rows = []

for _, a_row in amazon_df.iterrows():
    amazon_id = a_row["id"]

    amazon_vec = vectorizer.transform([a_row["serialized"]])

    # cosine similarity against ALL google
    scores = cosine_similarity(amazon_vec, google_tfidf)[0]

    # top-k indices
    top_idx = np.argsort(scores)[::-1][:TOP_K]

    for rank, g_idx in enumerate(top_idx, start=1):
        rows.append({
            AMAZON_ID_COL: amazon_id,
            GOOGLE_ID_COL: google_df.iloc[g_idx]["id"],
            "rank": rank,
            "tfidf_score": float(scores[g_idx]),
            "amazon_serialized": a_row["serialized"],
            "google_serialized": google_df.iloc[g_idx]["serialized"],
        })

candidates_df = pd.DataFrame(rows)

print("Total candidates:", len(candidates_df))

# -------------------------------------------------------------------
# Candidate generation recall@K (BLOCKING QUALITY)
# -------------------------------------------------------------------
def recall_at_k_blocking(candidates_df, gt_df, k):
    hits = 0
    grouped = candidates_df.groupby(AMAZON_ID_COL)

    for _, row in gt_df.iterrows():
        aid = row[AMAZON_ID_COL]
        gid = row[GOOGLE_ID_COL]

        if aid not in grouped.groups:
            continue

        topk = grouped.get_group(aid).sort_values("rank").head(k)
        if gid in topk[GOOGLE_ID_COL].values:
            hits += 1

    return hits / len(gt_df)


print("\nCandidate generation recall (before LLM)")
for k in [5, 10, 20, 50]:
    print(f"Recall@{k}: {recall_at_k_blocking(candidates_df, gt_df, k):.4f}")

# -------------------------------------------------------------------
# Save for LLM verification
# -------------------------------------------------------------------
candidates_df.to_csv("candidates.csv", index=False)