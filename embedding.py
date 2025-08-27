"""
get embeddings of different models
"""

from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import torch
from transformers import (RobertaModel, RobertaTokenizer)
import models.unixcoder.model
import models.cocosoda.model

def get_embedding_model(n_gpu, device, model_name):
    m = None
    if "bge" in model_name:
        print("using BGE embdding model")
        m = BGE_Embedding(n_gpu, device)
    elif "unixcoder" in model_name:
        print("using unixcoder embdding model")
        m = UnixcoderEmbedding(n_gpu, device)
    elif "cocosoda" in model_name:
        print("using cocosoda embdding model")
        m = CocosodaEmbedding(n_gpu, device)
    # elif "codet5p" in model_name:
    #     print("using codet5p embdding model")
    #     m = Codet5pEmbedding(n_gpu, device)
    # elif "t5" in model_name:
    #     print("using t5 embdding model")
    #     m = t5Embedding(n_gpu, device)
    elif "codet5p220" in model_name:
        print("using codet5p220 embdding model")
        m = Codet5p220Embedding(n_gpu, device)
    return m
        
class ModelEmbedding:
    """
    base class
    """
    def __init__(self):
        self.model = None
        
    def get_embedding(self,encoded_input, is_nl):
        with torch.no_grad():
            if is_nl:
                vec = self.model(nl_inputs=encoded_input) 
            else:
                vec = self.model(code_inputs=encoded_input)
            return vec

class Codet5p220Embedding(ModelEmbedding):
    def __init__(self, n_gpu, device, model_path="/data/hugang/JjyCode/llm/codet5p-220m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = T5EncoderModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

    def get_embedding(self, encoded_input, is_nl=None):
        # Compute token embeddings
        with torch.no_grad():
            attention_mask = encoded_input['attention_mask']
            model_output = self.model(**encoded_input)[0]
            #  Perform pooling. In this case, cls pooling.
            sentence_embeddings = (model_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings

class t5Embedding(ModelEmbedding):
    def __init__(self, n_gpu, device, model_path="/data/hugang/JjyCode/llm/t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = T5EncoderModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

    def get_embedding(self, encoded_input, is_nl=None):
        # Compute token embeddings
        with torch.no_grad():
            attention_mask = encoded_input['attention_mask']
            model_output = self.model(**encoded_input)[0]
            #  Perform pooling. In this case, cls pooling.
            sentence_embeddings = (model_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings

class Codet5pEmbedding(ModelEmbedding):
    def __init__(self, n_gpu, device, model_path="/data/hugang/JjyCode/llm/codet5p-110m-embedding"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

    def get_embedding(self, encoded_input, is_nl=None):
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            #  Perform pooling. In this case, cls pooling.

            sentence_embeddings = model_output
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings


class BGE_Embedding(ModelEmbedding):
    def __init__(self, n_gpu, device, model_path="/data/hugang/JjyCode/llm/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)
            
    def get_embedding(self, encoded_input,is_nl=None):
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            #  Perform pooling. In this case, cls pooling.
        
            sentence_embeddings = model_output[0][:, 0]
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings

        
class UnixcoderEmbedding(ModelEmbedding):
    def __init__(self, n_gpu, device, model_path="/data/hugang/JjyCode/llm/unixcoder-base"):
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaModel.from_pretrained(model_path) 
        self.model = models.unixcoder.model.Model(model)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)


class CocosodaEmbedding(ModelEmbedding):
    def __init__(self, n_gpu, device, model_path="/data/hugang/JjyCode/llm/CoCoSoDa"):
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaModel.from_pretrained(model_path) 
        self.model = models.cocosoda.model.Model(model)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)




