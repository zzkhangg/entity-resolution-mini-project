from openai import OpenAI
from .constants import *
import time
import json
from .cache import *

client = OpenAI()
llm_cache = load_cache()

def llm_match(amazon_text, google_text):
    prompt = PROMPT_TEMPLATE.format(
        amazon_record=amazon_text,
        google_record=google_text
    )

    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap + fast 
        messages=[
            {"role": "system", "content": "You are an expert entity resolution system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    latency = time.time() - start

    content = response.choices[0].message.content

    result = json.loads(content)

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "evidence": result["evidence"],
        "latency": latency,
        "tokens": response.usage.total_tokens
    }

def llm_match_cached(amazon_id, google_id, amazon_text, google_text):
    key = f"{amazon_id}||{google_id}"

    # cache hit
    if key in llm_cache:
        return llm_cache[key]

    # cache miss → call LLM
    result = llm_match(amazon_text, google_text)

    llm_cache[key] = result
    save_cache(llm_cache)

    return result