from .constants import *

def compute_metrics(pairs_df, gold_pairs, threshold):
    pairs_df["prediction"] = (pairs_df["similarity"] >= threshold).astype(int)
    pred_df = pairs_df[pairs_df["prediction"] == 1]
    pred_pairs = set(zip(pred_df[AMAZON_ID_COL], pred_df[GOOGLE_ID_COL]))

    TP = len(gold_pairs & pred_pairs)
    FP = len(pred_pairs - gold_pairs)
    FN = len(gold_pairs - pred_pairs)

    precision = TP / (TP + FP)  
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1