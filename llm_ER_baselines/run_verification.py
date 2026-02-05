import pandas as pd
from llm_verify import call_llm
from src.constants import AMAZON_ID_COL, GOOGLE_ID_COL
from sklearn.metrics import precision_recall_fscore_support
from src.constants import GT_PATH
import time
# -----------------------
# Config
# -----------------------
HIGH_CONF = 0.90
LOW_CONF = 0.30

llm_calls = 0
total_tokens = 0
start_time = time.perf_counter()

# -----------------------
# Load data
# -----------------------
candidates_df = pd.read_csv("candidates.csv")
gt_df = pd.read_csv(GT_PATH)  # adjust path if needed

gold_pairs = set(zip(gt_df[AMAZON_ID_COL], gt_df[GOOGLE_ID_COL]))

results = []
llm_calls = 0

# -----------------------
# LLM verification + gating
# -----------------------
for _, row in candidates_df.iterrows():
    score = row["tfidf_score"]

    if score >= HIGH_CONF:
        label = "match"
        confidence = 1.0

    elif score <= LOW_CONF:
        label = "no_match"
        confidence = 1.0

    else:
        llm_calls += 1
        out = call_llm(
            row["amazon_serialized"],
            row["google_serialized"]
        )
        label = out["label"]
        confidence = out["confidence"]
        total_tokens += out.get("tokens", 0)

    results.append({
        AMAZON_ID_COL: row[AMAZON_ID_COL],
        GOOGLE_ID_COL: row[GOOGLE_ID_COL],
        "label": label,
        "confidence": confidence,
        "rank": row["rank"]
    })

end_time = time.perf_counter()
total_time = end_time - start_time
total_pairs = len(candidates_df)

avg_latency = total_time / total_pairs
throughput = total_pairs / total_time

print("\nRuntime metrics")
print(f"Total time (sec): {total_time:.2f}")
print(f"Avg latency per pair (sec): {avg_latency:.4f}")
print(f"Throughput (pairs/sec): {throughput:.2f}")

final_df = pd.DataFrame(results)

# -----------------------
# Recall@k (after verification)
# -----------------------
def recall_at_k_verified(df, gold_pairs, k):
    hits = 0
    grouped = df[df["label"] == "match"].groupby(AMAZON_ID_COL)

    for aid, gid in gold_pairs:
        if aid not in grouped.groups:
            continue
        topk = grouped.get_group(aid).sort_values("rank").head(k)
        if gid in topk[GOOGLE_ID_COL].values:
            hits += 1

    return hits / len(gold_pairs)


print("\nRecall AFTER LLM verification")
for k in [5, 10, 20, 50]:
    print(f"Recall@{k}: {recall_at_k_verified(final_df, gold_pairs, k):.4f}")

# -----------------------
# Final Precision / Recall / F1
# -----------------------
y_true = []
y_pred = []

for _, row in final_df.iterrows():
    pair = (row[AMAZON_ID_COL], row[GOOGLE_ID_COL])
    y_true.append(1 if pair in gold_pairs else 0)
    y_pred.append(1 if row["label"] == "match" else 0)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary"
)
  
print("\nFinal metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")

# -----------------------
# LLM savings
# -----------------------
total_pairs = len(candidates_df)
saved_calls = total_pairs - llm_calls

print("\nLLM usage")
print("Total candidate pairs:", total_pairs)
print("LLM calls made:", llm_calls)
print("LLM calls saved:", total_pairs - llm_calls)
print(f"Saved %: {(total_pairs - llm_calls) / total_pairs:.2%}")
print("Total LLM tokens:", total_tokens)

# -----------------------
# Save results
# -----------------------
final_df.to_csv("final_matches.csv", index=False)
