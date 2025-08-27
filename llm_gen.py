from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import re
from tqdm import tqdm
import os
import json
from openai import OpenAI

def generate_exinfo(args, data):
    client = OpenAI(
        api_key="your key",
        base_url="url",
        timeout=120,
    )

    prompt = """
Next, I will provide you with a query and some code.
The provided code is for reference only.
You need to extract information from this code that meets the requirements of the query and finally give a description of the query's needs.
Do not provide an analysis of each code; the result should only be a description of the query's requirements within 80 words.
{query}, {code}
"""

    code = data['code_input']
    query = data['nl_input']
    language = args.lang
    if args.lang == 'cosqa':
        language = 'Python'
    prompt_text = prompt.format(language=language, query=query, code=code)
    messages = [{"role": "system", "content": "You are an expert in code search tasks, skilled at evaluating the relationship between a given text and multiple pieces of code. Given a query text, you can identify the most suitable code from the provided options and rank and score them according to their match degree with the text."},
                {"role": "user", "content": prompt_text}]

    completion = client.chat.completions.create(
        # model='gpt-4.1-mini',
        # model='deepseek-v3',
        model='qwen-plus',
        messages=messages
    )
    res = completion.choices[0].message.content
    return res



if __name__ == '__main__':

    data = {'nl_input': 'Replace the owner with a new owner .', 'code_input': [{'url': 0, 'code': 'contract c29479{ function replaceOwner(address owner, address newOwner) public onlyWallet onlyOwnerExists(owner) onlyOwnerDoesNotExist(newOwner) { for (uint256 i = 0; i < owners.length; i++) { if (owners[i] == owner) { owners[i] = newOwner; break; } } isOwner[owner] = false; isOwner[newOwner] = true; OwnerRemoval(owner); OwnerAddition(newOwner); } }'}, {'url': 579, 'code': 'contract c15553{ function changeOwner(address _newOwner) external onlyOwner() { owner = _newOwner; emit ChangedOwner(owner); } }'}, {'url': 150, 'code': 'contract c39269{ function changeOwner(address _owner) public onlyOwner returns (bool) { ChangedOwner(owner, _owner); owner = _owner; return true; } }'}], 'url': 0}
    generate_rank_t(data)
