#langs=(sql solidity cosqa)
langs=(solidity)

#langs=rust

MODE="samples"
MODEL_NAME="deepseek"
MODEL_PATH="/data/hugang/JjyCode/llm/deepseek-coder-1.3b-instruct"


for lang in "${langs[@]}"; do

  dir="./data/cross-domain/$lang"

  exquery_path="$dir/${lang}_test_exquery.jsonl"

  CUDA_VISIBLE_DEVICES="4" \
  python generate_query.py \
      --mode "$MODE" \
      --model_name "$MODEL_NAME" \
      --model_path "$MODEL_PATH" \
      --query_data_file $exquery_path \
      --output_path "$dir" \
      --lang $lang
done