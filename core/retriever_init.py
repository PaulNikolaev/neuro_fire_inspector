from pydantic import Field
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from core.model_init import (
    load_structured_documents,
    split_documents,
    init_embeddings,
    get_or_create_faiss,
    init_bm25_and_crossencoder
)

# Инициализация моделей и индекс
print("🚀 Инициализация Retriever и моделей...")

docs = load_structured_documents()
split_docs = split_documents(docs)

embedding = init_embeddings()
faiss_db = get_or_create_faiss(split_docs, embedding)
bm25_model, cross_encoder = init_bm25_and_crossencoder(split_docs)


# Retriever
class RerankingRetrieverBM25(BaseRetriever):
    vectorstore: FAISS = Field(exclude=True)
    cross_encoder: CrossEncoder = Field(exclude=True)
    bm25_model: BM25Okapi = Field(exclude=True)
    split_docs: List[Document] = Field(exclude=True)
    k_bm25: int = 15
    k_final: int = 3

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        tokenized_query = query.split()
        # BM25
        bm25_docs = self.bm25_model.get_top_n(tokenized_query, self.split_docs, n=self.k_bm25)
        # FAISS
        faiss_docs = self.vectorstore.similarity_search(query, k=self.k_bm25)
        # Объединяем
        combined_docs = {id(doc): doc for doc in (bm25_docs + faiss_docs)}.values()
        # CrossEncoder reranking
        pairs = [(query, doc.page_content) for doc in combined_docs]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.k_final]]


# Экземпляр Retriever
retriever = RerankingRetrieverBM25(
    vectorstore=faiss_db,
    cross_encoder=cross_encoder,
    bm25_model=bm25_model,
    split_docs=split_docs,
    k_bm25=15,
    k_final=3
)

print("✅ Retriever готов к использованию.")
