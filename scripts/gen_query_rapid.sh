langs=(sql solidity cosqa)
#langs=sql

MODE="samples"
MODEL_NAME="deepseek"
MODEL_PATH="/data/hugang/JjyCode/llm/deepseek-coder-1.3b-instruct"


dir="./data/rapid"
cosqa_query_path=$dir/cosqa/cosqa-test-901.json
cosqa_code_path=$dir/cosqa/cosqa-test-901.json

sql_query_path=$dir/sql/batch_0.txt
sql_code_path=$dir/sql/batch_0.txt

solidity_query_path=$dir/solidity/Solidity.txt
solidity_code_path=$dir/solidity/Solidity.txt


for lang in "${langs[@]}"; do

  query_path="${lang}_query_path"
  query_path=${!query_path}

  code_path="${lang}_code_path"
  code_path=${!code_path}

  CUDA_VISIBLE_DEVICES="0,2" \
  python generate_code.py \
      --mode "$MODE" \
      --model_name "$MODEL_NAME" \
      --model_path "$MODEL_PATH" \
      --query_data_file $query_path \
      --output_path "$dir/${lang}" \
      --lang $lang
done