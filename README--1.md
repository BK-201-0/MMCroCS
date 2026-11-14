# MMCroCS

Code retrieval across domains. Queries and code are enriched by an LLM, encoded by three embedding models, and ranked by a fused similarity score. Evaluated on **CoSQA**, **Solidity** and **SQL** in two settings: **Cross-Domain** and **RAPID**.

Four stages, run in order:

| Stage | Name | What it does |
| ----- | ---- | ------------ |
| (a) | LLM Zero-Shot Generation | query → description, code → comment |
| (b) | Hybrid Retrieval Ranking | rank candidates with fused similarity |
| (c) | Query Expansion Scoring | score and emit expanded queries |
| (d) | Reverse Weighted Search | re-rank with the expanded queries |

Stage (a) is optional — generated files are already in `data/`.

## Setup

```bash
pip install -r requirements.txt
```

- `python==3.8.18`, `torch==2.0.1`, CUDA required
- Embedding models: [CoCoSoDa](https://huggingface.co/DeepSoftwareAnalytics/CoCoSoDa), [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [UniXcoder](https://huggingface.co/microsoft/unixcoder-base)
- Generation model: [DeepSeek-Coder-1.3b-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- Scripts pin `CUDA_VISIBLE_DEVICES`; change it to match your machine

---

## (1) Cross-Domain Setting

Data root: `data/cross-domain`

### (a) LLM Zero-Shot Generation

```
Input
  cosqa/code_idx_map.txt, cosqa/cosqa-retrieval-test-500.json
  solidity/batch_0.txt
  sql/batch_0.txt

Run   (set MODEL_PATH to your local DeepSeek-Coder checkpoint first)
  bash scripts/gen_query_code.sh   # query -> description
  bash scripts/gen_code_code.sh    # code  -> comment

Output
  {cosqa,solidity,sql}/*_test_gen_des.jsonl
  {cosqa,solidity,sql}/*_test_comment.jsonl
```

### (b) Hybrid Retrieval Ranking

```
Input
  the corpora from (a)
  {cosqa,solidity,sql}/*_test_gen_des.jsonl
  {cosqa,solidity,sql}/*_test_comment.jsonl

Run
  bash scripts/get_embedding.sh          # -> ./vectors/cross-domain/
  # set --mode "eval_hrr" in scripts/eval.sh
  bash scripts/eval.sh

Output
  MRR / Top-1 / Top-5 / Top-10 per language
```

### (c) Query Expansion Scoring

```
Input
  same as (b)

Run
  bash scripts/get_embedding.sh
  # set --mode "eval_qes" in scripts/eval.sh
  bash scripts/eval.sh

Output
  cosqa/cosqa_test_exquery_1.jsonl
  solidity/solidity_test_exquery_1.jsonl
  sql/sql_test_exquery_1.jsonl
```

### (d) Reverse Weighted Search

```
Input
  {cosqa,solidity,sql}/*_test_comment.jsonl
  {cosqa,solidity,sql}/*_test_exquery_1.jsonl

Run
  bash scripts/get_embedding.sh          # exquery vectors are new
  # set --mode "eval_rws" in scripts/eval.sh
  bash scripts/eval.sh

Output
  ranked results per language
```

---

## (2) RAPID Setting

Data root: `data/rapid`

The four stages are unchanged. Only three things differ:

| | Cross-Domain | RAPID |
| --- | --- | --- |
| Embeddings | `scripts/get_embedding.sh` | `scripts/get_embedding_rapid.sh` |
| Evaluation | `scripts/eval.sh` | `scripts/eval_rapid.sh` |
| Test data | `cosqa-retrieval-test-500.json`, `solidity/batch_0.txt`, `sql/batch_0.txt` | `cosqa-test-901.json`, `solidity/Solidity.txt`, `sql/batch_0.txt` |

### (a) LLM Zero-Shot Generation

```
Run
  bash scripts/gen_query_rapid.sh

Output
  data/rapid/{cosqa,solidity,sql}/*_test_gen_des.jsonl
  data/rapid/{cosqa,solidity,sql}/*_test_comment.jsonl
  data/rapid/{cosqa,solidity,sql}/*_test_gen_code.jsonl
```

### (b) Hybrid Retrieval Ranking

```
Run
  bash scripts/get_embedding_rapid.sh    # -> ./vectors/rapid/
  # set --mode "eval_hrr" in scripts/eval.sh
  bash scripts/eval_rapid.sh
```

### (c) Query Expansion Scoring

```
Run
  bash scripts/get_embedding_rapid.sh
  # set --mode "eval_qes" in scripts/eval.sh
  bash scripts/eval_rapid.sh

Output
  data/rapid/{cosqa,solidity,sql}/*_test_exquery_1_3.jsonl
```

### (d) Reverse Weighted Search

```
Run
  bash scripts/get_embedding_rapid.sh
  # set --mode "eval_rws" in scripts/eval.sh
  bash scripts/eval_rapid.sh
```

---

## Quick Reference

| Script | Purpose |
| ------ | ------- |
| `gen_query_code.sh` | query → description |
| `gen_code_code.sh` | code → comment |
| `gen_exquery.sh` | further query expansion |
| `gen_query_rapid.sh` | RAPID generation |
| `get_embedding.sh` / `get_embedding_rapid.sh` | build embeddings |
| `eval.sh` / `eval_rapid.sh` | run evaluation |

`eval.py --mode` accepts `embedding`, `eval_hrr`, `eval_qes`, `eval_rws`, plus the one-shot `eval1` / `eval2` used by the shipped shell scripts.

## Notes

- `get_embedding*.sh` hold a **single** `lang` variable (`sql` / `cosqa` / `solidity` / `rust`) — change it and re-run per language.
- Re-run `get_embedding*.sh` after stage (c); the expanded queries need their own vectors.
- `MODEL_PATH` in the generation scripts is an absolute local path.
- Metrics come from `metric.py` (`MRR@k`, `Recall@k`); `analyze.py` computes rank statistics over saved results in `results/`.

## Citation

```bibtex
@inproceedings{mmcrocs,
  title     = {MMCroCS: Multi-Model Code Search across Domains},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```
