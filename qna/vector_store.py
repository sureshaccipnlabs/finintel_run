from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreBuilder:

    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name

    def build(self, documents):
        embeddings = OllamaEmbeddings(model=self.model_name)
        return FAISS.from_documents(documents, embeddings)
