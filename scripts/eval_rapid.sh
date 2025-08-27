#langs=(sql solidity cosqa)
langs=(sql)

models=(cocosoda bge-large-en-v1.5 unixcoder)

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


    comment_path="$dir/${lang}/${lang}_test_comment.jsonl"

    gencode_path="$dir/${lang}/${lang}_test_gen_code.jsonl"
    paths=()

    for model in "${models[@]}"; do
        path="./vectors/rapid/$model/$lang/"
        paths+=("$path")  
    done 
    python ./eval.py  \
        --mode "eval1"    \
        --model_name_or_path $model \
        --eval_batch_size 8 \
        --device cuda \
        --lang $lang \
        --query2code_cache_path "${paths[0]}$lang-query.npy" \
        --query_target_code_cache_path   "${paths[0]}$lang-code.npy" \
        --query2comment_cache_path  "${paths[1]}$lang-query.npy" \
        --comment_cache_path   "${paths[1]}$lang-comment.npy" \
        --gencode_target_code_cache_path "${paths[2]}$lang-code.npy" \
        --gen_code_cache_path   "${paths[2]}$lang-gencode.npy" \
        --query_cocosoda_path   "${paths[0]}$lang-query.npy" \
        --code_cocosoda_path   "${paths[0]}$lang-code.npy" \
        --query_bge_path   "${paths[1]}$lang-query.npy" \
        --code_bge_path   "${paths[1]}$lang-code.npy" \
        --query_unixcoder_path   "${paths[2]}$lang-query.npy" \
        --code_unixcoder_path   "${paths[2]}$lang-code.npy" \
        --comment_cocosoda_path   "${paths[0]}$lang-comment.npy" \
        --comment_bge_path   "${paths[1]}$lang-comment.npy" \
        --comment_unixcoder_path   "${paths[2]}$lang-comment.npy" \
        --gendes_cocosoda_path   "${paths[0]}$lang-gendes.npy" \
        --gendes_bge_path   "${paths[1]}$lang-gendes.npy" \
        --gendes_unixcoder_path   "${paths[2]}$lang-gendes.npy" \
        --exquery_cocosoda_path   "${paths[0]}$lang-exquery-qwen-3.npy" \
        --exquery_bge_path   "${paths[1]}$lang-exquery-qwen-3.npy" \
        --exquery_unixcoder_path   "${paths[2]}$lang-exquery-qwen-3.npy" \
        --query_data_file $query_path \
        --code_data_file $code_path \
        --comment_data_file $comment_path \
        --gen_code_data_file $gencode_path \
        --output_path "$dir/${lang}" \
        --w1 0.65 \
        --w2 0.25 \
        --w3 0.10
done
