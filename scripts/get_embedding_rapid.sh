lang=sql  # sql/cosqa/solidty
#lang=solidity
#lang=cosqa

models=(cocosoda bge-large-en-v1.5 unixcoder) #bge-large-en-v1.5/unixcoder/cocosoda
#formats=(query code comment gendes)
formats=(exquery)



dir="./data/rapid"

cosqa_query_path=$dir/cosqa/cosqa-test-901.json
cosqa_code_path=$dir/cosqa/cosqa-test-901.json


sql_query_path=$dir/sql/batch_0.txt
sql_code_path=$dir/sql/batch_0.txt

solidity_query_path=$dir/solidity/Solidity.txt
solidity_code_path=$dir/solidity/Solidity.txt


query_path="${lang}_query_path"
query_path=${!query_path}

code_path="${lang}_code_path"
code_path=${!code_path}


comment_path="$dir/${lang}/${lang}_test_comment.jsonl"

gencode_path="$dir/${lang}/${lang}_test_gen_code.jsonl"

gendes_path="$dir/${lang}/${lang}_test_gen_des.jsonl"

exquery_path="$dir/${lang}/${lang}_test_exquery_1_3.jsonl"



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
            --output_dir ./vectors/rapid \
            --eval_batch_size 8 \
            --device cuda \
            --lang $lang 
    done
done