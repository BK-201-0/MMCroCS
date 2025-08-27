langs=(sql solidity cosqa)
#langs=(cosqa)

models=(cocosoda bge-large-en-v1.5 unixcoder)

dir="./data/cross-domain"
cosqa_query_path=$dir/cosqa/cosqa-retrieval-test-500.json
cosqa_code_path=$dir/cosqa/code_idx_map.txt


sql_query_path=$dir/sql/batch_0.txt
sql_code_path=$dir/sql/batch_0.txt

solidity_query_path=$dir/solidity/batch_0.txt
solidity_code_path=$dir/solidity/batch_0.txt


for lang in "${langs[@]}"; do
    dir1="./data/cross-domain/$lang"
    query_path="${lang}_query_path"
    query_path=${!query_path}

    code_path="${lang}_code_path"
    code_path=${!code_path}


    comment_path="$dir/${lang}/${lang}_test_comment.jsonl"

    gencode_path="$dir/${lang}/${lang}_test_gen_code.jsonl"

    recode_path="$dir/${lang}/${lang}_test_code_python.jsonl"

    regencode_path="$dir/${lang}/${lang}_test_gen_code_python.jsonl"

    gendes_path="$dir/${lang}/${lang}_test_gen_des.jsonl"


    paths=()

    for model in "${models[@]}"; do
        path="./vectors/cross-domain/$model/$lang/"
        paths+=("$path")  
    done
    CUDA_VISIBLE_DEVICES="2" \
    python ./eval.py  \
        --mode "eval1"    \
        --model_name_or_path $model \
        --eval_batch_size 8 \
        --device cuda \
        --lang $lang \
        --query2code_cache_path "${paths[1]}$lang-query.npy" \
        --query_target_code_cache_path   "${paths[1]}$lang-code.npy" \
        --query2comment_cache_path  "${paths[1]}$lang-query.npy" \
        --comment_cache_path   "${paths[1]}$lang-comment.npy" \
        --gencode_target_code_cache_path "${paths[1]}$lang-code.npy" \
        --gen_code_cache_path   "${paths[1]}$lang-gencode.npy" \
        --gencode_target_recode_cache_path "${paths[1]}$lang-recode-python.npy" \
        --gen_code_python_cache_path   "${paths[1]}$lang-gencode-python.npy" \
        --query_cocosoda_path   "${paths[0]}$lang-query.npy" \
        --code_cocosoda_path   "${paths[0]}$lang-code.npy" \
        --query_bge_path   "${paths[1]}$lang-query.npy" \
        --code_bge_path   "${paths[1]}$lang-code.npy" \
        --query_unixcoder_path   "${paths[2]}$lang-query.npy" \
        --code_unixcoder_path   "${paths[2]}$lang-code.npy" \
        --gencode_cocosoda_path   "${paths[0]}$lang-gencode.npy" \
        --comment_cocosoda_path   "${paths[0]}$lang-comment.npy" \
        --gencode_bge_path   "${paths[1]}$lang-gencode.npy" \
        --comment_bge_path   "${paths[1]}$lang-comment.npy" \
        --gencode_unixcoder_path   "${paths[2]}$lang-gencode.npy" \
        --comment_unixcoder_path   "${paths[2]}$lang-comment.npy" \
        --gendes_cocosoda_path   "${paths[0]}$lang-gendes.npy" \
        --gendes_bge_path   "${paths[1]}$lang-gendes.npy" \
        --gendes_unixcoder_path   "${paths[2]}$lang-gendes.npy" \
        --gendes1_cocosoda_path   "${paths[0]}$lang-gendes-1.npy" \
        --gendes1_bge_path   "${paths[1]}$lang-gendes-1.npy" \
        --gendes1_unixcoder_path   "${paths[2]}$lang-gendes-1.npy" \
        --exquery_cocosoda_path   "${paths[0]}$lang-exquery-qwen-3.npy" \
        --exquery_bge_path   "${paths[1]}$lang-exquery-qwen-3.npy" \
        --exquery_unixcoder_path   "${paths[2]}$lang-exquery-qwen-3.npy" \
        --query_data_file $query_path \
        --code_data_file $code_path \
        --comment_data_file $comment_path \
        --gen_code_data_file $gencode_path \
        --output_path "$dir1" \
        --w1 0.65 \
        --w2 0.25 \
        --w3 0.10 \
        --w4 0.05
done
