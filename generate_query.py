from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from text_dataset import TextDataset
import re
from tqdm import tqdm
import os
import json


class DeepSeekModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).cuda()

    def generate(self, content, max_new_tokens=128):
        messages = [
            {'role': 'user', 'content': content}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            self.model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def generate_query(model, dataset, args, max_new_tokens=256):
    # check dir
    if not os.path.exists(args.output_path):
        print(f"creating dir: {args.output_path}")
        os.makedirs(args.output_path)

    # check
    prompt = """
After thinking step by step.This is a task of matching queries with code. 
Below, I will provide you with a query and multiple pieces of code in different {language}. 
You need to identify the code that meets the requirements of the query from these options and provide me with information about the code that aligns with the query's needs.
If there is no suitable code available, provide a description of the requirements for the query.
{query}, {code}

"""
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    final_ans = []
    for i, item in tqdm(enumerate(dataset)):
        code = item['code_input']
        query = item['nl_input']
        prompt_text = prompt.format(language=language, query=query, code=code)
        exquery = model.generate(prompt_text, max_new_tokens=max_new_tokens)
        final_ans.append({'nl_input': exquery, 'url': item['url']})

    filename = os.path.join(args.output_path, f'{args.lang}_test_exquery_1.jsonl')
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(final_ans, file, indent=4)

    print(f"Data has been written to {filename}")

def generate_query_t(model, data, args, max_new_tokens=256):
    prompt = """
After thinking step by step.This is a task of matching queries with code. 
Below, I will provide you with a query and multiple pieces of code in different {language}. 
You need to identify the code that meets the requirements of the query from these options and provide me with information about the code that aligns with the query's needs.
If there is no suitable code available, provide a description of the requirements for the query.
{query}, {code}
Do not provide descriptions for each piece of code.
"""
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    code = data['code_input']
    query = data['nl_input']
    prompt_text = prompt.format(language=language, query=query, code=code)
    gencode = model.generate(prompt_text, max_new_tokens=max_new_tokens)
    # json_str = gencode.split('```json')[1].split('```')[0].strip()
    # data = json.loads(json_str)
    print(f"code_input", gencode)
    # print(f"code_in", data[0])

def generate_rank_t(data, args, max_new_tokens=256):
    model = DeepSeekModel(args.model_path)
    prompt = """
After thinking step by step.This is a task of matching queries with code.
I will provide you with a query and multiple code entries, where each code entry contains both a code and a url.
Score each code based on whether its functionality meets the query, with a scoring range of 0-100. The output format should be the URL of each code followed by its score, and do not output anything else.
{query}, {code}
Do not generate any other content or provide code snippets.
    """
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    code = data['code_input']
    query = data['nl_input']
    prompt_text = prompt.format(query=query, code=code)
    gencode = model.generate(prompt_text, max_new_tokens=max_new_tokens)
    print(f"code_input", gencode)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="samples", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--query_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--output_path", default=None, type=str, help="output path")
    parser.add_argument("--is_dev", action='store_true')
    parser.add_argument("--lang", default="python", type=str, help="language")

    args = parser.parse_args()
    # query_dataset = TextDataset(args, None, "text", args.query_data_file)
    if 'deepseek' in args.model_path:
        gen_model = DeepSeekModel(args.model_path)

    # print(f"len of query dataset: {len(query_dataset)}")

    # generate_query(gen_model, query_dataset, args, max_new_tokens=256)

    # print(query_dataset[0])
    # data = {'nl_input': 'Replace the owner with a new owner .', 'code_input': 'contract c29479{ function replaceOwner(address owner, address newOwner) public onlyWallet onlyOwnerExists(owner) onlyOwnerDoesNotExist(newOwner) { for (uint256 i = 0; i < owners.length; i++) { if (owners[i] == owner) { owners[i] = newOwner; break; } } isOwner[owner] = false; isOwner[newOwner] = true; OwnerRemoval(owner); OwnerAddition(newOwner); } } contract c39269{ function changeOwner(address _owner) public onlyOwner returns (bool) { ChangedOwner(owner, _owner); owner = _owner; return true; } } contract c15553{ function changeOwner(address _newOwner) external onlyOwner() { owner = _newOwner; emit ChangedOwner(owner); } }'}
    # data = {'nl_input': 'Replace the owner with a new owner .',
    #         'code_input': 'contract c24941{ function computeRealCap(uint256 _cap, uint256 _key) public pure returns (bytes32) { return keccak256(_cap, _key); } } contract c90{ function HashnodeTestCoin() { balances[msg.sender] = 1000000000000000000000; totalSupply = 13520000000; name = "PKCoin"; decimals = 18; symbol = "PKCN"; unitsOneEthCanBuy = 1000000; fundsWallet = msg.sender; } }'}

    # generate_query_t(gen_model, data, args, max_new_tokens=256)

    data = {'nl_input': 'Replace the owner with a new owner .', 'code_input': [{'url': 0, 'code': 'contract c29479{ function replaceOwner(address owner, address newOwner) public onlyWallet onlyOwnerExists(owner) onlyOwnerDoesNotExist(newOwner) { for (uint256 i = 0; i < owners.length; i++) { if (owners[i] == owner) { owners[i] = newOwner; break; } } isOwner[owner] = false; isOwner[newOwner] = true; OwnerRemoval(owner); OwnerAddition(newOwner); } }'}, {'url': 579, 'code': 'contract c15553{ function changeOwner(address _newOwner) external onlyOwner() { owner = _newOwner; emit ChangedOwner(owner); } }'}, {'url': 150, 'code': 'contract c39269{ function changeOwner(address _owner) public onlyOwner returns (bool) { ChangedOwner(owner, _owner); owner = _owner; return true; } }'}], 'url': 0}
    generate_rank_t(data, args, max_new_tokens=256)