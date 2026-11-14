# MMCroCS
Code retrieval across domains. Queries and code are enriched by an LLM, encoded by three embedding models, and ranked by a fused similarity score. Evaluated on **CoSQA**, **Solidity** and **SQL** in two settings: **Cross-Domain** and **RAPID**.

### Project Structure
```bash
-data # datasets and generated data
    -cross-domain 
    -rapid
-models # base models
-scripts # scripts for data preprocessing and evaluation
```

### Model
We use the following models:
* [CoCoSoDa](https://huggingface.co/DeepSoftwareAnalytics/CoCoSoDa)
* [UniXcoder](https://huggingface.co/microsoft/unixcoder-base)
* [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

### Environment
* `python==3.8.18`
* `torch==2.0.1`
* You can install the dependencies by:
```bash
pip install -r requirements.txt
```

### Evaluation in Cross-Domain Setting
* We have provided the generated files in `data/`. For instructions on generating the description and comments using [DeepSeek-Coder-1.3b-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct), please refer to `generate_query.py` and `generate_comment.py`.
* To reproduce the result, start by obtaining the embeddings from various models. The embeddings will be stored in the folder `vectors`. You can modify the dataset name in `scripts/get_embedding.sh` to get different embeddings.
```bash
bash scripts/get_embedding.sh
```
* Evaluate the results.
```bash
bash scripts/eval.sh
```

### Evaluation in RAPID Setting
* Get embeddings.
```bash
bash scripts/get_embedding_rapid.sh
```
* Evaluate the results.
```bash
bash scripts/eval_rapid.sh
```
### Detailed Steps of Each Stage

Four stages, run in order:

| Stage | Name | What it does |
| ----- | ---- | ------------ |
| (a) | LLM Zero-Shot Generation | query → description, code → comment |
| (b) | Hybrid Retrieval Ranking | rank candidates with fused similarity |
| (c) | Query Expansion Scoring | score and emit expanded queries |
| (d) | Reverse Weighted Search | re-rank with the expanded queries |

Stage (a) is optional — generated files are already in `data/`.

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
  cosqa/code_idx_map.txt, cosqa/cosqa-retrieval-test-500.json
  solidity/batch_0.txt
  sql/batch_0.txt
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
  cosqa/code_idx_map.txt, cosqa/cosqa-retrieval-test-500.json
  solidity/batch_0.txt
  sql/batch_0.txt
  {cosqa,solidity,sql}/*_test_gen_des.jsonl
  {cosqa,solidity,sql}/*_test_comment.jsonl

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
  cosqa/code_idx_map.txt, cosqa/cosqa-retrieval-test-500.json
  solidity/batch_0.txt
  sql/batch_0.txt
  {cosqa,solidity,sql}/*_test_comment.jsonl
  {cosqa,solidity,sql}/*_test_exquery_1.jsonl

Run
  bash scripts/get_embedding.sh          # exquery vectors are new
  # set --mode "eval_rws" in scripts/eval.sh
  bash scripts/eval.sh

Output
  MRR / Top-1 / Top-5 / Top-10 per language
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

### Data Analysis
* `analyze.py` is used to analyze the results of hybrid retrieval ranking (HRR) and hybrid retrieval ranking + query expansion scoring (QES). We provide the experimental outputs of HRR and HRR+QES in the `results` folder. `sql_res3.jsonl` stores queries and their top-3 matched code candidates after HRR, while `sql_res10_ex.jsonl` contains queries and their top-10 matched code candidates after HRR+QES.
* Please update the data path inside `analyze.py` before running to avoid file-not-found errors.

### Function Introduction
In the file `eval.py`:
- The `get_exinfomation` function implements the query expansion scoring function.
- The `get_exrank` function implements the reverse weighted search function.
- The `get_example` function obtains the final result data and saves it as a file.

Results at different stages can be obtained by calling various functions.

