# %%
import os
from dotenv import load_dotenv
load_dotenv()
import random
random.seed(42) ### Reproducible

from src.matcher import llm_match_cached
from src.labels import generate_gold_df
from src.loader import load, serialize_record
import pandas as pd
import time
from collections import defaultdict
from src.constants import *
from src.blocker import calculate_similiarity
from openai import OpenAI

# Load the variables from .env

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) 

start = time.perf_counter()
# %%
#### Loading data
amazon_df = load(AMAZON_PATH)
amazon_df["serialized"] = amazon_df.apply(lambda r: serialize_record(r, AMAZON_FIELDS), axis= 1)

google_df = load(GOOGLE_PATH)
google_df["serialized"] = google_df.apply(lambda r: serialize_record(r, GOOGLE_FIELDS), axis= 1)

gt_df = pd.read_csv(GT_PATH)

### Generate gold dataframe
gold_df = generate_gold_df(gt_df, all_google_ids=google_df["id"].tolist())
### Report class balance
print("------------- Class balance -------------")
class_balance = gold_df["label"].value_counts()

print(class_balance)
print(class_balance / len(gold_df))


print("------------- - -------------")

### Clone new pairs_df from ground truth for filtering
pairs_df = gold_df.drop(columns=["label"])

pairs_df = pairs_df.merge(
    amazon_df[["id", "serialized"]],
    left_on=AMAZON_ID_COL,
    right_on="id",
    how="left"
).drop(columns=["id"]).merge(
    google_df[["id", "serialized"]],
    left_on=GOOGLE_ID_COL,
    right_on="id",
    suffixes=("_amazon", "_google"),
    how="left"
).drop(columns=["id"])

### TF/IDF Cosine Similiarity 
pairs_df['serialized_amazon'] = pairs_df['serialized_amazon'].fillna('')
pairs_df['serialized_google'] = pairs_df['serialized_google'].fillna('')
# %%
pairs_df = calculate_similiarity(pairs_df, "serialized_amazon", "serialized_google")

end = time.perf_counter()
total_time = end - start
num_pairs = len(pairs_df)

avg_latency = total_time / num_pairs
throughput = num_pairs / total_time
# %%
### Filter to choose candidates for LLM matcher
BLOCK_THRESHOLD = 0.3
candidates = pairs_df[pairs_df["similarity"] >= BLOCK_THRESHOLD]
print(len(candidates))

results = []

for _, row in candidates.iterrows():
    output = llm_match_cached(
        row[AMAZON_ID_COL],
        row[GOOGLE_ID_COL],
        row["serialized_amazon"],
        row["serialized_google"]
    )

    results.append({
        AMAZON_ID_COL: row[AMAZON_ID_COL],
        GOOGLE_ID_COL: row[GOOGLE_ID_COL],
        "pred_label": 1 if output["label"] == "match" else 0,
        "confidence": output["confidence"],
        "latency": output["latency"],
        "tokens": output["tokens"]
    })

llm_df = pd.DataFrame(results)


### Evaluation
gold_pairs = set(
    zip(
        gold_df[gold_df["label"] == 1][AMAZON_ID_COL],
        gold_df[gold_df["label"] == 1][GOOGLE_ID_COL]
    )
)

pred_pairs = set(
    zip(
        llm_df[llm_df["pred_label"] == 1][AMAZON_ID_COL],
        llm_df[llm_df["pred_label"] == 1][GOOGLE_ID_COL]
    )
)

TP = len(gold_pairs & pred_pairs)
FP = len(pred_pairs - gold_pairs)
FN = len(gold_pairs - pred_pairs)

### Performance metrics
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

### Runtime + cost metrics
avg_latency = llm_df["latency"].mean()
throughput = 1 / avg_latency
total_calls = len(llm_df)
total_tokens = llm_df["tokens"].sum()

summary = pd.DataFrame([{
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "avg_latency_sec": avg_latency,
    "throughput_pairs_per_sec": throughput,
    "llm_calls": total_calls,
    "total_tokens": total_tokens
}])

print(summary)