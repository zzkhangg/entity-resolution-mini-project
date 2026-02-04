# %%
from src.labels import generate_gold_df
from src.loader import load, serialize_record
import pandas as pd
import numpy as np
import time
from src.constants import *
from src.eval import compute_metrics
from src.blocker import calculate_similiarity


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
## Evaluation
gold_pairs = set(
        zip(
                gold_df[gold_df["label"] == 1][AMAZON_ID_COL],
                gold_df[gold_df["label"] == 1][GOOGLE_ID_COL]
        )
)
thresholds = np.arange(0.0, 1, 0.05)

rows = []

for t in thresholds:
    p, r, f1 = compute_metrics(pairs_df, gold_pairs, t)
  
    rows.append({
        "threshold": round(t, 2),
        "precision": p,
        "recall": r,
        "f1": f1,
        "avg_latency_sec": avg_latency,
        "throughput_pairs_per_sec": throughput
    })

summary_df = pd.DataFrame(rows)

best_row = summary_df.loc[summary_df["f1"].idxmax()]

print("Best threshold by F1:")
print(best_row)