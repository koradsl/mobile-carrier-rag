import os
import pickle as pkl

from tqdm import tqdm

from src.chat_engine import ChatEngine
from src.embedding_model import EmbeddingModel
from src.text_splitter import create_chunks


class RagSystem:
    def __init__(self, documents_path: str = "", host: str = "") -> None:
        self.idx2document = {}
        self.idx2embedding = {}

        if os.path.exists("idx2document.pkl"):
            with open("idx2document.pkl", "rb") as f:
                self.idx2document = pkl.load(f)
        else:
            self.documents = self.read_documents(path=documents_path)
            self.idx2document = {idx: document for idx, document in enumerate(self.documents)}

            with open("idx2document.pkl", "wb") as f:
                pkl.dump(self.idx2document, f)

        self.embedding_model = EmbeddingModel(model_name="BM-K/KoSimCSE-roberta-multitask")
        self.embedding = {}

        if os.path.exists("idx2embedding.pkl"):
            with open("idx2embedding.pkl", "rb") as f:
                self.idx2embedding = pkl.load(f)
        else:
            for idx, document in tqdm(self.idx2document.items()):
                self.idx2embedding[idx] = self.embedding_model.get_embedding(document.page_content)

            # self.batch = 8
            #
            # for idx in tqdm(range(0, len(self.idx2document), self.batch)):
            #     documents = [self.idx2document[i].page_content for i in range(idx, idx + self.batch)]
            #     embeddings = self.embedding_model.get_embedding(documents)[0]
            #
            #     # embedding size = (batch_size, sequence_length, hidden_size)
            #     for i in range(idx, idx + self.batch):
            #         self.idx2embedding[i] = embeddings[idx - i]

            with open("idx2embedding.pkl", "wb") as f:
                pkl.dump(self.idx2embedding, f)

        self.chat_engine = ChatEngine(host=host)

    def read_documents(self, path: str) -> list[str]:
        directories = os.listdir(path)
        documents = []

        for directory in directories:
            directories2 = os.listdir(os.path.join(path, directory))

            for directory2 in directories2:
                files = os.listdir(os.path.join(path, directory, directory2))

                for file in files:
                    file_path = os.path.join(path, directory, directory2, file)

                    documents += create_chunks(file_path)

        return documents

    def __call__(self, query):
        query_embedding = self.embedding_model.get_embedding(query)

        scores = []

        for idx, embedding in self.idx2embedding.items():
            score = self.embedding_model.cal_score(query_embedding[1], embedding[1])
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        top_k = 5
        chunks = ""

        for idx, _ in scores[:top_k]:
            chunks += self.idx2document[idx].page_content + "\n"

        content = f"""{chunks}
        
위 문서에 대해 질문에 대한 답을 해주세요.
{query}"""

        print(chunks)

        response = self.chat_engine.chat(messages=[{"role": "user", "content": content}])

        # print(content)
        # print(response)
        # for r in response:
        #     print(r["message"]["content"], end="")
        for r in response:
            # Yield each part of the streamed response as it becomes available
            yield r["message"]["content"]


