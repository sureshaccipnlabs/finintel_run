from langchain_community.llms import Ollama

from documents import DocumentFactory
from knowledge import KnowledgeBuilder
from loader import AggregatedJSONLoader
from models import KnowledgeBase
from router import RuleBasedRouter
from vector_store import VectorStoreBuilder


SYSTEM_PROMPT = """
You are a financial analytics AI assistant.

Rules:
- Answer only from the provided context.
- Do not infer, assume, estimate, or invent facts that are not explicitly present in the context.
- If the answer is not fully supported by the context, say: "I do not have enough information in the provided data."
- Use exact figures, names, and values from the context whenever available.
- If multiple values are relevant, present them clearly and do not drop important qualifiers.
- If a field is missing in the context, say that it is not available in the provided data.
- Do not use outside knowledge.
- Do not make up trends, reasons, summaries, or comparisons unless they are directly supported by the context.
- Be concise and business oriented.
- Prefer short bullet-style answers when comparing entities.
"""


class FinancialAnalyticsAgent:

    def __init__(self, data_folder: str = "data_", model_name: str = "mistral"):
        self.data_folder = data_folder
        self.model_name = model_name
        self.loader = AggregatedJSONLoader(folder_path=data_folder)
        self.knowledge_builder = KnowledgeBuilder()
        self.document_factory = DocumentFactory()
        self.vector_store_builder = VectorStoreBuilder(model_name=model_name)
        self.router = RuleBasedRouter()
        self.llm = Ollama(model=model_name)
        self.knowledge: KnowledgeBase | None = None
        self.vector_db = None

    def bootstrap(self) -> None:
        payloads = self.loader.load()
        self.knowledge = self.knowledge_builder.build(payloads)
        documents = self.document_factory.build(self.knowledge)
        self.vector_db = self.vector_store_builder.build(documents)

    def ask(self, question: str) -> str:
        if self.knowledge is None or self.vector_db is None:
            raise RuntimeError("Agent is not initialized. Call bootstrap() before ask().")

        direct_answer = self.router.answer(question=question, knowledge=self.knowledge)
        if direct_answer:
            return direct_answer

        docs = self.vector_db.similarity_search(question, k=6)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n"
        )
        return self.llm.invoke(prompt)
