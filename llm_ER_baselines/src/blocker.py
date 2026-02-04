import re
from sklearn.feature_extraction.text import TfidfVectorizer


# Global vectorizer used ONLY for retrieval
vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(2, 3),
    min_df=2
)


def normalize_numbers(text: str) -> str:
    return re.sub(r"(\d)\s+(\d)", r"\1\2", text)


def extract_field(text: str, field: str) -> str:
    suffix_pattern = r"\b(inc|corp|corporation|ltd|limited|llc|co|company)\b|[^\w\s]"
    match = re.search(rf"{field}:\s*([^\n]+)", text, re.IGNORECASE)

    if not match:
        return None

    value = match.group(1).lower().strip()
    value = re.sub(suffix_pattern, "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_name_prefix(text: str, n_tokens: int = 2) -> str:
    fields = ["name", "title"]
    text = normalize_numbers(text)

    for field in fields:
        match = re.search(rf"{field}:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            name = match.group(1).lower()
            name = re.sub(r"[^\w\s]", " ", name)
            name = re.sub(r"\s+", " ", name).strip()
            tokens = name.split()
            return " ".join(tokens[:n_tokens]) if tokens else None

    return None


def make_block_keys(text: str):
    strong = []
    weak = []

    manu = extract_field(text, "manufacturer")
    name = extract_name_prefix(text, n_tokens=2)

    if manu and name:
        strong.append(f"{manu}__{name}")

    if manu:
        weak.append(manu)

    if name:
        weak.append(name.split()[0])

    return strong, weak


def get_block_candidates(amazon_row, google_blocks, min_candidates=20):
    strong_keys, weak_keys = make_block_keys(amazon_row["serialized"])

    candidates = set()

    # 1. strong keys
    for key in strong_keys:
        candidates |= google_blocks.get(key, set())

    # 2. weak expansion
    if len(candidates) < min_candidates:
        for key in weak_keys:
            candidates |= google_blocks.get(key, set())

    return list(candidates)
