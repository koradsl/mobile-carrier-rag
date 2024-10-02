import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.model.eval()
        self.model = self.model.half()

    def get_embedding(self, sentences: list[str]):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            embeddings = self.model(**inputs, return_dict=False)

        return embeddings

    def cal_score(self, a, b):
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]

        return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100
