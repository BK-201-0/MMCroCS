#lang=sql # sql/cosqa/solidity/rust
#lang=cosqa
lang=solidity
#lang=rust

models=(cocosoda bge-large-en-v1.5 unixcoder) #bge-large-en-v1.5/unixcoder/cocosoda
#formats=(recode)
formats=(exquery)


dir="./data/cross-domain/$lang"

gencode_path="$dir/${lang}_test_gen_code_python.jsonl"
recode_path="$dir/${lang}_test_code_python.jsonl"
gendes_path="$dir/${lang}_test_gen_des.jsonl"
exquery_path="$dir/${lang}_test_exquery_1.jsonl" # 1 qwen 2 gpt 3 deepseek



for model in "${models[@]}"; do
    for format in "${formats[@]}"; do
        datafile="${format}_path"
        datafile=${!datafile}
        CUDA_VISIBLE_DEVICES="0,2" \
        python eval.py \
            --mode embedding \
            --model_name_or_path $model\
            --format $format \
            --datafile  $datafile \
            --output_dir ./vectors/cross-domain \
            --eval_batch_size 8 \
            --device cuda \
            --lang $lang
    done
done
