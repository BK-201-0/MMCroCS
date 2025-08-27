langs=(sql solidity cosqa)
#langs=sql

MODE="samples"
MODEL_NAME="deepseek"
MODEL_PATH="/data/hugang/JjyCode/llm/deepseek-coder-1.3b-instruct"


for lang in "${langs[@]}"; do

  dir="./data/cross-domain/$lang"

  cosqa_query_path=$dir/cosqa-retrieval-test-500.json
  cosqa_code_path=$dir/code_idx_map.txt

  sql_query_path=$dir/batch_0.txt
  sql_code_path=$dir/batch_0.txt

  solidity_query_path=$dir/batch_0.txt
  solidity_code_path=$dir/batch_0.txt

  rust_query_path=$dir/rust_1000.txt

  query_path="${lang}_query_path"
  query_path=${!query_path}

  code_path="${lang}_code_path"
  code_path=${!code_path}

  CUDA_VISIBLE_DEVICES="0" \
  python generate_code.py \
      --mode "$MODE" \
      --model_name "$MODEL_NAME" \
      --model_path "$MODEL_PATH" \
      --query_data_file $query_path \
      --output_path "$dir" \
      --lang $lang
done