"""
clean version of evaluation
"""
import os
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
import logging
import numpy as np
import argparse
import csv
import random
from text_dataset import TextDataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                        RobertaConfig, RobertaModel, RobertaTokenizer)
from embedding import ModelEmbedding, get_embedding_model
import metric
from prettytable import PrettyTable
from llm_gen import generate_rank, generate_exinfo

logger = logging.getLogger(__name__)


class ResultTable:
    def __init__(self, title):
        self.table = PrettyTable(title)       
    def add_row(self,row):
        self.table.add_row(row)
    def print_table(self):
        print(self.table)
def get_datasets(args, tokenizer,data_file, use_origin=True):
    """
    get datasets
    :param use_origin: return text dataset
    :return tokenized_dataloader:sequential dataloader for encoded input
    :return origin_dataset: origin dataset for text
    """
    tokenized_dataset = TextDataset(args, tokenizer, "tokenize", data_file)
    origin_dataset = None
    if use_origin:
        origin_dataset = TextDataset(args, tokenizer, "text", data_file)
    sampler = SequentialSampler(tokenized_dataset)
    tokenized_dataloader = DataLoader(tokenized_dataset, sampler=sampler, batch_size=args.eval_batch_size,num_workers=4)
    return tokenized_dataloader, origin_dataset


def get_embeddings(device, embedding_model,dataloader, is_nl, save_vector_path=None):
    """
    return matrix of embeddings (num, hidden_size)
    """
    vecs = []
    for batch in tqdm(dataloader):
        # batch[0]: code_input
        # btach[1]: nl_input
        if is_nl:
            inputs = batch[1].to(device)
        else:
            inputs = batch[0].to(device)
        embeds = embedding_model.get_embedding(inputs, is_nl)
        vecs.append(embeds.cpu().numpy())
    vecs = np.concatenate(vecs,0)
    if save_vector_path:
        logger.info(f"saving query vector to {save_vector_path} {vecs.shape}")
        np.save(save_vector_path, vecs)
    return vecs
  

def get_exinfomation(args, scores):
    query_dataset = TextDataset(args, None, "text", args.query_data_file)
    code_dataset = TextDataset(args, None, "text", args.code_data_file)

    nl_urls = []
    for ex in query_dataset:
        nl_urls.append({'url': ex['url'], 'query': ex['nl_input']})
    code_urls = []
    for ex in code_dataset:
        code_urls.append({'url': ex['url'], 'code': ex['code_input']})

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    sort_urls = [[code_urls[idx] for idx in sort_id] for sort_id in sort_ids]

    # k = [1, 5, 7, 9]
    k = [3]
    for x in k:
        final_ans = []
        for i in tqdm(range(len(sort_urls))):
            top_codes = [{'url': code['url'], 'code': code['code']} for code in sort_urls[i][:x]]
            res = {'nl_input': nl_urls[i]['query'], 'code_input': top_codes, 'url': nl_urls[i]['url']}
            exinfo = generate_exinfo(args, res)
            final_ans.append({'nl_input': exinfo, 'url': nl_urls[i]['url']})

        # print(final_ans[0])
        filename = os.path.join(args.output_path, f'{args.lang}_test_exquery_1_{x}.jsonl')
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(final_ans, file, indent=4)
        print(f"Data has been written to {filename}")


def evaluate(args,query2code_scores,query2comment_scores, code2code_scores, gendes2code_scores=None):
    if args.mode == 'eval':
        assert args.w2 != None and args.w3 != None

        query_dataset = TextDataset(args,None,"text",args.query_data_file)
        code_dataset = TextDataset(args, None, "text",args.code_data_file)
        gen_code_dataset = TextDataset(args, None, "text", args.gen_code_data_file)
        
        nl_urls = []
        for ex in query_dataset:
            nl_urls.append(ex['url'])
        code_urls = []
        for ex in code_dataset:
            code_urls.append(ex['url'])
            
        # test different weights
        title = ["MRR","Top-1","Top-5","Top-10"]
        result_table = ResultTable(title)   
        w = [args.w1,args.w2, args.w3]
            
        res = get_results(w[0]*query2code_scores + w[1]*query2comment_scores + w[2]*code2code_scores,nl_urls, code_urls)
    
        result_table.add_row(res)
        result_table.print_table()
    elif args.mode == 'eval1':
        query_dataset = TextDataset(args, None, "text", args.query_data_file)
        code_dataset = TextDataset(args, None, "text", args.code_data_file)

        nl_urls = []
        for ex in query_dataset:
            nl_urls.append(ex['url'])
        code_urls = []
        for ex in code_dataset:
            code_urls.append(ex['url'])

        # test different weights
        title = ["MRR", "Top-1", "Top-5", "Top-10"]
        result_table = ResultTable(title)
        w = [args.w1, args.w2, args.w3, args.w4]
        if gendes2code_scores is not None:
            res = get_results(w[0] * query2code_scores + w[1] * query2comment_scores + w[2] * code2code_scores + w[3] * gendes2code_scores, nl_urls,
                          code_urls)
        elif code2code_scores is not None:
            res = get_results(w[0] * query2code_scores + w[1] * query2comment_scores + w[2] * code2code_scores, nl_urls,
                          code_urls)
        else:
            res = get_results(query2code_scores, nl_urls, code_urls)

        result_table.add_row(res)
        result_table.print_table()

    elif args.mode == 'traverse':
        query_dataset = TextDataset(args, None, "text", args.query_data_file)
        code_dataset = TextDataset(args, None, "text", args.code_data_file)
        nl_urls = []
        for ex in query_dataset:
            nl_urls.append(ex['url'])
        code_urls = []
        for ex in code_dataset:
            code_urls.append(ex['url'])

        # create result table
        title = ["weights","MRR","Top-1","Top-5","Top-10"]
        result_table = ResultTable(title)   
        # traverse with different weights
        step_size = 0.01
        # traverse all the weights, where w1+w2+w3=1, with step_size
        w1_list = np.arange(0,1+step_size,step_size)
        w2_list = np.arange(0,1+step_size,step_size)
        w3_list = np.arange(0,1+step_size,step_size)
        ans_list = []
        control_ans_list = []
        for w1 in tqdm(w1_list):
            for w2 in w2_list:
                if w1 + w2 > 1:
                    continue
                for w3 in w3_list:
                    # check if the sum of w1,w2,w3 is 1
                    if w1 + w2 + w3 != 1:
                        continue
                    res = get_results(w1*query2code_scores + w2*query2comment_scores + w3*code2code_scores,nl_urls, code_urls)
                    r = ["-".join([str(round(w1,2)),str(round(w2,2)),str(round(w3,2))])] + res
                    ans_list.append(r)
        with open('traverse.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(ans_list)


def get_results(scores, nl_urls, code_urls):
    """
    given scores matrix(nl,cl) and labeld urls, return a list containing:
    """
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1] 
        
    sort_urls = [[code_urls[idx] for idx in sort_id] for sort_id in sort_ids]

    mrrs = metric.cal_mrr(sort_urls, nl_urls)
    recalls = metric.cal_recall(sort_urls, nl_urls)
    mrrs = [round(float(r),3) for r in list(mrrs.values())]
    # mrr10 mrr1000, hr1,hr5, hr10
    mrrs = [mrrs[5]]
    recalls = [round(float(r),3) for r in list(recalls.values())][:3]
    return mrrs + recalls
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--code_data_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--comment_data_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).") 
    parser.add_argument("--gen_code_data_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).") 
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    # there're two queries embeddings maybe
    parser.add_argument("--query2code_cache_path", default=None, type=str, 
                        help="tt")
    parser.add_argument("--query2comment_cache_path", default=None, type=str, 
                        help="tt")
    parser.add_argument("--query_target_code_cache_path", default=None, type=str, 
                        help="maching query2code")
    parser.add_argument("--gencode_target_code_cache_path", default=None, type=str,help="matching gencode cache")

    parser.add_argument("--gencode_target_recode_cache_path", default=None, type=str,help="exteng gencode cache")

    parser.add_argument("--gen_code_python_cache_path", default=None, type=str,help="extend gencode cache")

    parser.add_argument("--query_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--code_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--query_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--code_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--query_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--code_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gencode_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gencode_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gencode_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes1_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes1_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes1_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment1_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment1_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment1_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--exquery_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gpt_exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gpt_exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gpt_exquery_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--ds_exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--ds_exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--ds_exquery_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--qwen_exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--qwen_exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--qwen_exquery_unixcoder_path", default=None, type=str,
                        help="tt")


    parser.add_argument("--comment_cache_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gen_code_cache_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--output_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--w1", default=None, type=float, 
                        help="tt")
    parser.add_argument("--w2", default=None, type=float, 
                        help="tt")    
    parser.add_argument("--w3", default=None, type=float, 
                        help="tt")
    parser.add_argument("--w4", default=None, type=float,
                        help="tt")
    
    parser.add_argument("--mode",default=None, type=str,help="eval/traverse", required=True)
    
    parser.add_argument("--model_name_or_path",default=None, type=str,help="embedding/eval")
    parser.add_argument("--format",default=None, type=str,help="query/code/comment/gencode")

    parser.add_argument("--output_dir",default=None, type=str,help="embedding/eval")
    parser.add_argument("--device",default='cuda', type=str,help="embedding/eval")
    parser.add_argument("--datafile",default=None, type=str,help="embedding/eval")
    


    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    print(args)
    n_gpu = torch.cuda.device_count()
    print(f"n_gpu: {n_gpu}")
    
    
    if args.mode == "embedding":
        # get embeddings and store
        embedding_model=get_embedding_model(n_gpu, args.device, args.model_name_or_path)
        dataloader, _ = get_datasets(args,embedding_model.tokenizer, args.datafile, use_origin=False)
        is_nl = "query" in args.format or "comment" in args.format or "gendes" in args.format or "exquery" in args.format
        dir_path = os.path.join(
                           args.output_dir,
                           args.model_name_or_path, 
                           args.lang)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_vector_path=os.path.join(dir_path,f"{args.lang}-{args.format}-qwen-3.npy")
        # save_vector_path=os.path.join(dir_path,f"{args.lang}-{args.format}.npy")
        logger.info(f"getting embedding of {args.lang} {args.format}")
        logger.info(f"save vectors to {save_vector_path}, is_nl = {is_nl}")
        get_embeddings(args.device, 
                       embedding_model,
                       dataloader,
                       is_nl,
                       save_vector_path=save_vector_path
                       )
    elif args.mode == "eval1":
        print("=====evaluating {}=====".format(args.lang))

        query_cocosoda_embedding = np.load(args.query_cocosoda_path)
        code_cocosoda_embedding = np.load(args.code_cocosoda_path)
        query_bge_embedding = np.load(args.query_bge_path)
        code_bge_embedding = np.load(args.code_bge_path)
        query_unixcoder_embedding = np.load(args.query_unixcoder_path)
        code_unixcoder_embedding = np.load(args.code_unixcoder_path)

        comment_cocosoda_embedding = np.load(args.comment_cocosoda_path)
        comment_bge_embedding = np.load(args.comment_bge_path)
        comment_unixcoder_embedding = np.load(args.comment_unixcoder_path)

        gendes_cocosoda_embedding = np.load(args.gendes_cocosoda_path)
        gendes_bge_embedding = np.load(args.gendes_bge_path)
        gendes_unixcoder_embedding = np.load(args.gendes_unixcoder_path)

        exquery_cocosoda_embedding = np.load(args.exquery_cocosoda_path)
        exquery_bge_embedding = np.load(args.exquery_bge_path)
        exquery_unixcoder_embedding = np.load(args.exquery_unixcoder_path)


        global_query_1 = np.concatenate((query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding),
                                        axis=-1)
        global_target_1 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding),
                                         axis=-1)

        global_query_2 = np.concatenate((query_bge_embedding, query_bge_embedding, query_bge_embedding), axis=-1)
        global_target_2 = np.concatenate((comment_bge_embedding, comment_bge_embedding, comment_bge_embedding), axis=-1)

        global_query_4 = np.concatenate((gendes_cocosoda_embedding, gendes_cocosoda_embedding, gendes_cocosoda_embedding), axis=-1)
        global_target_4 = np.concatenate((code_cocosoda_embedding, code_cocosoda_embedding, code_cocosoda_embedding), axis=-1)

        global_query_5 = np.concatenate((exquery_cocosoda_embedding, exquery_bge_embedding, exquery_unixcoder_embedding), axis=-1)
        global_target_5 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)

        scores_1 = global_query_1 @ global_target_1.T
        scores_2 = global_query_2 @ global_target_2.T
        scores_4 = global_query_4 @ global_target_4.T
        scores_5 = global_query_5 @ global_target_5.T

        # scores = scores_1 * 0.4 + scores_2 * 0.25 + scores_4 * 0.00 + scores_5 * 0.35

        # get_exinfomation(args, scores)

        # evaluate(args, scores, None, None)
        evaluate(args, scores_1, scores_2, scores_5)

