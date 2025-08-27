# Hybrid-CS
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
* We have provided the generated files in `data/`. For instructions on generating the code and comments using [DeepSeek-Coder-1.3b-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct), please refer to `generate_code.py` and `generate_comment.py`.
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
