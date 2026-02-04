AMAZON_ID_COL = "idAmazon"
GOOGLE_ID_COL = "idGoogleBase"
THRESHOLD = 0.4

AMAZON_PATH = "./dataset/Amazon-GoogleProducts/Amazon.csv"
AMAZON_FIELDS = ["title", "description", "manufacturer"]

GOOGLE_PATH = "./dataset/Amazon-GoogleProducts/GoogleProducts.csv"
GOOGLE_FIELDS = ["name", "description", "manufacturer"]

GT_PATH = "./dataset/Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv"

PROMPT_TEMPLATE = """
You are an expert in product entity resolution.

Determine whether the following two product records refer to the SAME real-world product.

Amazon product:
{amazon_record}

Google product:
{google_record}

Rules:
- Answer ONLY with valid JSON
- Do NOT include any extra text
- Decide strictly between "match" or "no_match"

Output format:
{{
  "label": "match" or "no_match",
  "confidence": a number between 0 and 1,
  "evidence": [
    "short reason 1",
    "short reason 2"
  ]
}}
"""