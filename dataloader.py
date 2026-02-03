# %%
import pandas as pd

# %%
## Functions for loading data
def load_and_normalize(csv_path, text_cols):
    df = pd.read_csv(
        csv_path,
        encoding="latin1",
        na_values=["NA", "null"],
        encoding_errors="ignore"
    )

    # Normalize text columns
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

    return df

def to_records(df):
    return df.to_dict(orient="records")

# %%
TABLE_ONE_PATH = "./dataset/Amazon-GoogleProducts/Amazon.csv"
TABLE_TWO_PATH = "./dataset/Amazon-GoogleProducts/GoogleProducts.csv"
GT_PATH = "dataset/Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv"
TEXT_COLS = ["id", "name", "description", "manufacturer", "price"]

amazon = load_and_normalize(TABLE_ONE_PATH, TEXT_COLS)
google = load_and_normalize(TABLE_TWO_PATH, TEXT_COLS)
gt = pd.read_csv(GT_PATH)