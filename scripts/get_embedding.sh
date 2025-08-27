lang=sql # sql/cosqa/solidity/rust
#lang=cosqa # sql/cosqa/solidity/rust
#lang=solidity # sql/cosqa/solidity/rust
#lang=rust # sql/cosqa/solidity/rust

models=(cocosoda bge-large-en-v1.5 unixcoder) #bge-large-en-v1.5/unixcoder/cocosoda
#models=(codet5p220)
formats=(comment)
#formats=(query code comment gencode)


dir="./data/cross-domain/$lang"
cosqa_query_path=$dir/cosqa-retrieval-test-500.json
cosqa_code_path=$dir/code_idx_map.txt


sql_query_path=$dir/batch_0.txt
sql_code_path=$dir/batch_0.txt

solidity_query_path=$dir/batch_0.txt
solidity_code_path=$dir/batch_0.txt


query_path="${lang}_query_path"
query_path=${!query_path}

code_path="${lang}_code_path"
code_path=${!code_path}


#comment_path="$dir/${lang}_test_comment.jsonl"
comment_path="$dir/${lang}_test_comment_1.jsonl"


gencode_path="$dir/${lang}_test_gen_code.jsonl"


for model in "${models[@]}"; do
    for format in "${formats[@]}"; do
        datafile="${format}_path"
        datafile=${!datafile}
        CUDA_VISIBLE_DEVICES="2, 4" \
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