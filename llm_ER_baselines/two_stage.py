# %%
import os
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

from src.loader import load, serialize_record
from src.constants import *
from src.blocker import make_block_keys, get_block_candidates, vectorizer

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
load_dotenv()
random.seed(42)
np.random.seed(42)

top_k_num = 50
GLOBAL_TFIDF_K = 200   # cheap, safe
BLOCK_K = 50
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
# Build Google block index
# -------------------------------------------------------------------
google_blocks = defaultdict(set)

for idx, row in google_df.iterrows():
    strong_keys, weak_keys = make_block_keys(row["serialized"])
    for key in strong_keys + weak_keys:
        google_blocks[key].add(idx)

# -------------------------------------------------------------------
# TF-IDF (fit ONCE)
# -------------------------------------------------------------------
google_tfidf = vectorizer.fit_transform(google_df["serialized"])

# -------------------------------------------------------------------
# Candidate generation + ranking
# -------------------------------------------------------------------
rows = []

for _, a_row in amazon_df.iterrows():
    amazon_id = a_row["id"]

    amazon_vec = vectorizer.transform([a_row["serialized"]])

    # -------- 1. GLOBAL TF-IDF retrieval (NO BLOCKING) --------
    global_scores = cosine_similarity(amazon_vec, google_tfidf)[0]
    global_top_idx = np.argsort(global_scores)[::-1][:GLOBAL_TFIDF_K]

    # -------- 2. BLOCKING retrieval --------
    block_indices = get_block_candidates(a_row, google_blocks)

    # -------- 3. UNION --------
    candidate_indices = set(global_top_idx) | set(block_indices)

    # Safety fallback (almost never triggered)
    if not candidate_indices:
        candidate_indices = set(range(len(google_df)))

    candidate_indices = list(candidate_indices)

    # -------- 4. FINAL RANKING --------
    blocked_vecs = google_tfidf[candidate_indices]
    scores = cosine_similarity(amazon_vec, blocked_vecs)[0]

    ranked = np.argsort(scores)[::-1][:top_k_num]

    for rank, local_idx in enumerate(ranked, start=1):
        g_idx = candidate_indices[local_idx]

        rows.append({
            AMAZON_ID_COL: amazon_id,
            GOOGLE_ID_COL: google_df.iloc[g_idx]["id"],
            "rank": rank,
            "tfidf_score": float(scores[local_idx]),
            "amazon_serialized": a_row["serialized"],
            "google_serialized": google_df.iloc[g_idx]["serialized"]
        })
        
candidates_df = pd.DataFrame(rows)

print("Total candidates:", len(candidates_df))
print(candidates_df.head())

# -------------------------------------------------------------------
# Proper Recall@k′ (ranking-aware)
# -------------------------------------------------------------------
def recall_at_k(candidates_df, gt_df, k):
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


for k in [5, 10, 20, 50]:
    r = recall_at_k(candidates_df, gt_df, k)
    print(f"Recall@{k}: {r:.4f}")

print("Elapsed:", time.perf_counter() - start)