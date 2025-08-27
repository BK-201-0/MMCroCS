

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import pickle
import numpy as np
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


class TextDataset(Dataset):
    def __init__(self,args, tokenizer, mode, file_path):
        print("dataset from: "+str(file_path))
        self.tokenizer = tokenizer
        data = []
        self.examples = []
        self.args = args
        self.mode = mode
        self.first = True
        
        with open(file_path) as f:
            # code_tokens, url, docstring_tokens
            if "comment" in file_path:
                # only contain nl_inputs, and url
                print("load dataset from comment....")
                
                for js in json.load(f):
                    temp = {}
                    temp['code_tokens'] = []
                    temp['url'] = js['url']
                    temp['docstring_tokens'] = js['nl_input'].split()
                    data.append(temp)
            elif "exquery" in file_path:
                print("load dataset from exquery....")
                for js in json.load(f):
                    temp = {}
                    t = []
                    if 'code_input' in js:
                        t = js['code_input']
                    temp['code_tokens'] = t
                    temp['url'] = js['url']
                    temp['docstring_tokens'] = js['nl_input'].split()
                    data.append(temp)
            elif "gen_code" in file_path:
                print("load dataset from gen code....")
                for idx,js in enumerate(json.load(f)):
                    temp = {}
                    temp['code_tokens'] = js['code_input']
                    temp['docstring_tokens'] = js['nl_input']
                    temp['url'] = js['url']
                    temp['gt'] = js['gt']
                    data.append(temp)

            elif "gen_des" in file_path:
                print("load dataset from gen des....")
                for js in json.load(f):
                    temp = {}
                    temp['code_tokens'] = []
                    temp['url'] = js['url']
                    temp['docstring_tokens'] = js['nl_input'].split()
                    data.append(temp)

            elif "test_code_python" in file_path:
                print("load dataset from code python....")
                for idx,js in enumerate(json.load(f)):
                    temp = {}
                    temp['code_tokens'] = js['code_input']
                    temp['docstring_tokens'] = []
                    temp['url'] = js['url']
                    temp['gt'] = js['gt']
                    data.append(temp)

            elif "jsonl" in file_path:
                print("load dataset from jsonl....")
                
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append(js)
                             
            # cosqa codebase
            elif "code_idx_map" in file_path:
                js = json.load(f)
                for idx, key in enumerate(js):
                    temp = {}
                    temp["url"] = js[key]
                    temp['code_tokens'] = key.split()
                    temp['docstring_tokens'] = ""
                    data.append(temp)

            elif "json" in file_path:
                # test for cosqa
                for js in json.load(f):
                    temp = {}
                    # else condition is for rapid cosqa
                    temp['code_tokens'] =  js['code_tokens'] # code
                    temp['docstring_tokens'] = js['doc'].split() if 'doc' in js else js['docstring_tokens'] # query
                    temp['url'] = js['retrieval_idx'] if 'retrieval_idx' in js else js['url']
                    data.append(temp) 
             
            elif "txt" in file_path:
                for idx, line in enumerate(f.readlines()):
                    line = line.strip().split('<CODESPLIT>')
                    if len(line) != 5:
                        continue
                    temp = {}
                    temp['docstring_tokens'] = line[3]
                    temp['code_tokens'] = line[4]
                    temp['url'] = idx
                    data.append(temp)
        if self.mode == "text":
            for js in data:
                self.examples.append(self.textlize(js))
        else:
            if "unixcoder" in self.args.model_name_or_path or "cocosoda" in self.args.model_name_or_path:
                print("dataset using unixcoder")
                for js in data:
                    self.examples.append(self.convert_examples_to_features_unixcoder(js,tokenizer,nl_length=128, code_length=256))
            elif "bge" in self.args.model_name_or_path or "codet5p" in self.args.model_name_or_path or "t5" in self.args.model_name_or_path or "codet5p220" in self.args.model_name_or_path:
                # tokenize the 
                for js in tqdm(data):
                    self.examples.append(self.tokenize(js))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        if self.mode == "text":
            return self.examples[i]
        else:
            if "bge" in self.args.model_name_or_path or "codet5p" in self.args.model_name_or_path or "t5" in self.args.model_name_or_path or "codet5p220" in self.args.model_name_or_path:
                return (self.examples[i]['code_input'], self.examples[i]['nl_input'])
            
            elif "unixcoder" in self.args.model_name_or_path or "cocosoda" in self.args.model_name_or_path:
                return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
    def textlize(self, js):
        if type(js['code_tokens']) == list:
            code = ' '.join(js['code_tokens'])
        else:
            code = js['code_tokens']
        if type(js['docstring_tokens']) == list:
            nl = ' '.join(js['docstring_tokens'])
        else:
            nl = js['docstring_tokens']
        if "gt" in js:
            final = {
                'code_input':code,
                'nl_input':nl,
                'url':js['url'],
                'gt':js['gt']
            }
        else:
            final = {
                'code_input':code,
                'nl_input':nl,
                'url':js['url']
            }
        
        return final
        
    def tokenize(self, js):
        
        if type(js['code_tokens']) == list:
            code = ' '.join(js['code_tokens'])
        else:
            code = js['code_tokens']
        if type(js['docstring_tokens']) == list:
            nl = ' '.join(js['docstring_tokens'])
        else:
            nl = js['docstring_tokens']
        
       
        code_input = self.tokenizer(code, padding="max_length", max_length=256,truncation=True, return_tensors = 'pt')
        
        code_input['input_ids'] = code_input['input_ids'].squeeze()
        code_input['attention_mask'] = code_input['attention_mask'].squeeze()
        if 'token_type_ids' in code_input:
            code_input['token_type_ids'] = code_input['token_type_ids'].squeeze()
        instruction = "Represent this sentence for searching relevant passages: "
       
        nl_input = self.tokenizer(nl, padding="max_length", truncation=True,max_length=128,return_tensors = 'pt' )

        nl_input['input_ids'] = nl_input['input_ids'].squeeze()
        nl_input['attention_mask'] = nl_input['attention_mask'].squeeze()
        if 'token_type_ids' in code_input:
            nl_input['token_type_ids'] = nl_input['token_type_ids'].squeeze()
        if self.first:
            print(f"shape of code_input: {code_input['input_ids'].shape}")
            print(f"shape of nl_input:{nl_input['input_ids'].shape}")
            print(f"code:\n {code}")
            print(f"nl:\n{nl}")
            self.first=False
        
        return {
            "code_input":code_input,
            "nl_input":nl_input,
            "url":js['url']
        }
    def convert_examples_to_features_unixcoder(self,js,tokenizer,nl_length=128, code_length=256):
        """convert examples to token ids"""
        code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
        code_tokens = tokenizer.tokenize(code)[:code_length-4]
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
        
        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['docstring_tokens'].split())
        
        nl_tokens = tokenizer.tokenize(nl)[:nl_length-4]
        nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length  
        if self.first:
                print(f"shape of code_input: {len(code_ids)}")
                print(f"shape of nl_input:{len(nl_ids)}")
                print(f"code:\n {code}")
                print(f"nl:\n{nl}")
                self.first=False  
        return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'])
   
