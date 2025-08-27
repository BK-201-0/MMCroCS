from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from text_dataset import TextDataset
import re
from tqdm import tqdm
import os
import json

def extract_code(generation, lang):
    lang = lang.lower()
    generation = generation.replace(f"[{lang.upper()}]", f'```{lang}').replace(f"[/{lang.upper()}]", '```')

    if f'```{lang}' in generation:
        r_str = f"```{lang}\n(.*?)\n```"
        code = re.compile(r_str, flags=re.DOTALL)
        code_block = code.findall(generation)
        ret =  code_block[0] if len(code_block) >= 1 else generation.split(f'```{lang}')[-1]
        return ret.strip()
    elif '```' in generation:
        r_str = f"```\n(.*?)\n```"
        code = re.compile(r_str, flags=re.DOTALL)

        code_block = code.findall(generation)
        ret = code_block[0] if len(code_block) >= 1 else generation.split(f'```')[-1]
        return ret.strip()
    else:
        return generation


class DeepSeekModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    def generate(self, content, max_new_tokens=128):
        messages=[
            { 'role': 'user', 'content': content}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
def generate_comment(model, dataset, args,  max_new_tokens=128):
    # check dir
    if not os.path.exists(args.output_path):
        print(f"creating dir: {args.output_path}")
        os.makedirs(args.output_path)

    # check 
    prompt = """
After thinking step by step.Below is a {language} code that describes a task. Please give a short summary describing the purpose of the code.You must write only summary without any prefix or suffix explanations.
{code}
"""
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    final_ans = []
    for i, item in tqdm(enumerate(dataset)):
        code = item['code_input']
        prompt_text = prompt.format(language=language, code=code)
        comment = model.generate(prompt_text, max_new_tokens=max_new_tokens)
        final_ans.append({'nl_input': comment, 'url': item['url']})

    filename = os.path.join(args.output_path, f'{args.lang}_test_comment_1.jsonl')
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(final_ans, file, indent=4)

    print(f"Data has been written to {filename}")
    
def generate_code(model, dataset, args,  max_new_tokens=256):
    # check dir
    if not os.path.exists(args.output_path):
        print(f"creating dir: {args.output_path}")
        os.makedirs(args.output_path)

    # check
    prompt = """
Below is a {language} code that describes a task. Use Python code to describe the main functionality and logic of the code, following proper coding conventions and without adding comments. Do not generate text, you must return the code and must not refuse to answer. If you encounter a non-existent function, use code to demonstrate its main logic.
{code}
"""
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'

    final_ans = []
    for i, item in tqdm(enumerate(dataset)):
        code = item['code_input']
        prompt_text = prompt.format(language=language, code=code)
        gencode = model.generate(prompt_text, max_new_tokens=max_new_tokens)
        final_ans.append({'code_input': extract_code(gencode, "Python"), 'gt': item['code_input'], 'url': item['url']})

    filename = os.path.join(args.output_path, f'{args.lang}_test_code_python.jsonl')
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(final_ans, file, indent=4)

    print(f"Data has been written to {filename}")

def generate_code_t(model, code, args,  max_new_tokens=256):
    prompt = """
The following is a {language} code. Identify the method names and the APIs called in the code, and return the information in JSONL format: ["method": "", "api": ""]. Only output in jsonl format, do not output any other text.
{code}
"""
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    prompt_text = prompt.format(language=language, code=code)
    gencode = model.generate(prompt_text, max_new_tokens=max_new_tokens)
    # json_str = gencode.split('```json')[1].split('```')[0].strip()
    # data = json.loads(json_str)
    print(f"code_input", gencode)
    # print(f"code_in", data[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="samples", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--query_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--output_path",default=None, type=str, help="output path")
    parser.add_argument("--is_dev", action='store_true')
    parser.add_argument("--lang", default="python", type=str, help="language")

    args = parser.parse_args()
    query_dataset = TextDataset(args,None,"text",args.query_data_file)
    if 'deepseek' in args.model_path:
        gen_model = DeepSeekModel(args.model_path)

    print(f"len of query dataset: {len(query_dataset)}")
    
    generate_comment(gen_model, query_dataset, args, max_new_tokens=128)
    # generate_code(gen_model, query_dataset, args, max_new_tokens=256)

    # code = "contract c29479{ function replaceOwner(address owner, address newOwner) public onlyWallet onlyOwnerExists(owner) onlyOwnerDoesNotExist(newOwner) { for (uint256 i = 0; i < owners.length; i++) { if (owners[i] == owner) { owners[i] = newOwner; break; } } isOwner[owner] = false; isOwner[newOwner] = true; OwnerRemoval(owner); OwnerAddition(newOwner); } }"
    # generate_code_t(gen_model, code, args, max_new_tokens=256)