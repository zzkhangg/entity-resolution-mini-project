import os
import json
import hashlib
import time
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
    """Stable hash key for a pair of records."""
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
        "evidence": [str],
        "tokens": int,
        "latency": float
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

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    latency = time.perf_counter() - start
    content = response.choices[0].message.content.strip()

    # -------- 3. HARD JSON PARSE --------
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned invalid JSON:\n{content}"
        ) from e

    # -------- 4. VALIDATION --------
    label = parsed.get("label")
    confidence = float(parsed.get("confidence", 0.0))
    evidence = parsed.get("evidence", [])

    if label not in {"match", "no_match"}:
        raise ValueError(f"Invalid label: {label}")

    result = {
        "label": label,
        "confidence": confidence,
        "evidence": evidence,
        "tokens": response.usage.total_tokens,
        "latency": latency
    }

    # -------- 5. SAVE TO CACHE --------
    CACHE[key] = result
    _save_cache()

    return result
