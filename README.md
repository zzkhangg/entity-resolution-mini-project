# Entity Resolution with LLMs: LLM-Based ER Baselines (Direct vs. Two-Stage)

This repository implements and evaluates **LLM-based Entity Resolution (ER)** using two strategies:

1. **Direct LLM Pairwise Classifier**
2. **Two-Stage (Blocking/Retrieval) + LLM Verification**

The goal is to understand trade-offs between **accuracy, cost, latency, and scalability**.

---

## Problem Definition

Given two datasets of entities (e.g., product listings), determine whether a pair of records refers to the **same real-world entity**.

Each pair is classified as:
- `MATCH`
- `NO_MATCH`

---

## Approaches

### 1. Direct LLM-Based Matching

**Description**

The LLM is invoked on every evaluated pair in a controlled candidate set.

**Pipeline**
```text
Evaluated pairs
      ↓
     LLM
      ↓
Match / No Match
```
**Pros**
- Very simple implementation
- No blocking or feature engineering
- High semantic understanding
- Suitable for small datasets

**Cons**
- Cost grows as O(n × m)
- High latency
- Not scalable

---

### 2. Two-Stage LLM-Based Matching

**Description**

A cheap blocking or retrieval stage is used to reduce candidate pairs before invoking the LLM for high-precision verification.

**Pipeline**
```text
All pairs (n × m)
↓
Blocking / Candidate Generation
↓
Reduced pairs (k ≪ n × m)
↓
LLM
↓
Match / No Match
```
In our project, we first perform candidate generation using TF-IDF cosine similarity, retrieving top-k candidates per Amazon record. This reduces the n×m comparison space and achieves Recall@50 ≈ 0.87. The resulting candidate pairs are then passed to an LLM for verification (Stage 2).

**Blocking methods**
- String similarity
- Shared attributes (manufacturer)
- Token overlap
- Rule-based filtering

**Pros**
- Much lower cost
- Faster execution
- Scales to large datasets
- Production-ready architecture

**Cons**
- Recall depends on blocker quality
- Blocker errors are irreversible
- More complex pipeline

---

## Evaluation Metrics

- Precision
- Recall
- F1 Score
- Average latency per pair
- Throughput (pairs/sec)
- Number of LLM calls
- Total token usage

---

## Key Results

| Metric | Value |
|------|------|
| Precision | 1.00 |
| Recall | 0.91 |
| F1 Score | 0.95 |
| Avg Latency (sec) | 1.72 |
| Throughput (pairs/sec) | 0.58 |
| LLM Calls | 2,177 |
| Total Tokens | 1,379,956 |

**Observations**
- Perfect precision indicates no false positives.
- In the Two-Stage setting, Recall loss is mainly due to conservative decisions or blocker misses.
- Latency and throughput confirm that scoring all n·m pairs is infeasible.

---

## Direct vs Two-Stage Comparison

### When Direct LLM Beats Two-Stage

- Small datasets
- Prototyping or demos
- No reliable blocking features
- Maximum recall required
- One-off experiments

**Reason:**  
No blocker errors and manageable cost at small scale.

---

### When Direct LLM Fails

- Large datasets
- Production systems
- Real-time constraints
- Limited budget

**Failure modes**
- Quadratic cost explosion
- High latency
- Excessive token usage

---

### When Two-Stage LLM Wins

- Medium to large datasets
- Production ER systems
- Cost-sensitive environments
- Batch or streaming ER

**Reason:**  
Blocking drastically reduces LLM calls while preserving most true matches.

---

## How to Run

### 1. Clone repo and install dependencies
```bash
git clone https://github.com/zzkhangg/entity-resolution-mini-project.git
pip install -r requirements.txt
cd llm_ER_baselines/
```
### 2. Configure LLM
```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Run experiments
For direct LLM Classifier:
```bash
python direct.py
```
For two stage Blocking + LLM Verfication:

```bash
python retrieval_blocking.py
python run_verification.py
```

## Reproducibility

- Fixed random seeds

- Deterministic prompts

- Response caching

## Conclusion

Direct LLM matching provides strong accuracy but does not scale.
Two-stage LLM-based ER achieves a strong accuracy–cost tradeoff and is recommended for real-world systems.