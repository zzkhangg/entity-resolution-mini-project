import os
import json
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
from src.constants import PROMPT_TEMPLATE

# ----------------------------------
# Setup
# ----------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"

# ----------------------------------
# Cache setup
# ----------------------------------
CACHE_PATH = "llm_cache.json"

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}


def _cache_key(amazon_record: str, google_record: str) -> str:
    """
    Stable hash key for a pair of records.
    """
    raw = amazon_record + "|||" + google_record
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, indent=2)


# ----------------------------------
# LLM call (with cache)
# ----------------------------------
def call_llm(amazon_record: str, google_record: str) -> dict:
    """
    Calls the LLM to verify whether two product records match.

    Returns:
    {
        "label": "match" | "no_match",
        "confidence": float,
        "evidence": [str, ...]
    }
    """

    key = _cache_key(amazon_record, google_record)

    # -------- 1. CACHE HIT --------
    if key in CACHE:
        return CACHE[key]

    # -------- 2. LLM CALL --------
    prompt = PROMPT_TEMPLATE.format(
        amazon_record=amazon_record,
        google_record=google_record
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # -------- 3. HARD JSON PARSE --------
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned invalid JSON:\n{content}"
        ) from e

    # -------- 4. VALIDATION --------
    if result["label"] not in {"match", "no_match"}:
        raise ValueError(f"Invalid label: {result['label']}")

    result["confidence"] = float(result["confidence"])

    # -------- 5. SAVE TO CACHE --------
    CACHE[key] = result
    _save_cache()

    return result