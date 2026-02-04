# %%
import pandas as pd
import random
random.seed(42)
from .constants import AMAZON_ID_COL, GOOGLE_ID_COL

# %%
def generate_positive_pairs(gt):
    positive_pairs = gt.copy()
    positive_pairs["label"] = 1
    return positive_pairs

def generate_negative_pairs(gt_map, all_google_ids, k=3):
    negatives = []

    for amazon_id, true_google_ids in gt_map.items():
        candidates = list(set(all_google_ids) - true_google_ids)

        sampled = random.sample(
            candidates,
            min(k, len(candidates),)
        )

        for google_id in sampled:
            negatives.append({
                AMAZON_ID_COL: amazon_id,
                GOOGLE_ID_COL: google_id,
                "label": 0
            })

    return pd.DataFrame(negatives)

def generate_gold_df(gt_df, all_google_ids):
    gt_map = (
        gt_df.groupby(AMAZON_ID_COL)[GOOGLE_ID_COL]
        .apply(set)
        .to_dict()
    )
    positive_pairs = generate_positive_pairs(gt_df)
    negative_pairs = generate_negative_pairs(gt_map,all_google_ids)

    gold_df = pd.concat(
        [positive_pairs, negative_pairs],
        ignore_index=True
    )

    return gold_df