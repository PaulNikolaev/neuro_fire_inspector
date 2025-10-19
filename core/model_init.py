import os
import pickle
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from core.pdf_loader import extract_text_from_pdf, parse_articles
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# ==========================
# Пути
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf_docs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
FAISS_DIR = os.path.join(CACHE_DIR, "faiss_index")
BM25_PATH = os.path.join(CACHE_DIR, "bm25.pkl")
CROSSENCODER_PATH = os.path.join(CACHE_DIR, "crossencoder.pkl")

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================
# Устройство
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Используем устройство: {DEVICE}")

# ==========================
# Загрузка и структурирование PDF
# ==========================
def load_structured_documents() -> List[Document]:
    structured_documents = []
    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, filename)
        text = extract_text_from_pdf(path)
        if not text:
            continue
        articles = parse_articles(text)
        for article, content in articles.items():
            structured_documents.append(
                Document(page_content=content, metadata={"source": filename, "article": article})
            )
    print(f"📘 Загружено {len(structured_documents)} статей")
    return structured_documents

# ==========================
# Разбиение документов
# ==========================
def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    split_docs = splitter.split_documents(docs)
    print(f"📑 Разбито на {len(split_docs)} чанков")
    return split_docs

# ==========================
# Embeddings
# ==========================
class HuggingFaceE5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

def init_embeddings() -> HuggingFaceE5Embeddings:
    return HuggingFaceE5Embeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": DEVICE}
    )

# ==========================
# FAISS
# ==========================
def get_or_create_faiss(split_docs: List[Document], embedding: HuggingFaceE5Embeddings) -> FAISS:
    index_file = os.path.join(FAISS_DIR, "faiss.index")
    if os.path.exists(index_file):
        print("💾 Загрузка существующего FAISS индекса...")
        return FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
    print("🧠 Создание нового FAISS индекса...")
    faiss_db = FAISS.from_documents(split_docs, embedding)
    faiss_db.save_local(FAISS_DIR)
    return faiss_db

# ==========================
# BM25 и CrossEncoder
# ==========================
def init_bm25_and_crossencoder(split_docs: List[Document]) -> Tuple[BM25Okapi, CrossEncoder]:
    # BM25
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            bm25_model = pickle.load(f)
    else:
        corpus = [doc.page_content.split() for doc in split_docs]
        bm25_model = BM25Okapi(corpus)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25_model, f)

    # CrossEncoder
    if os.path.exists(CROSSENCODER_PATH):
        with open(CROSSENCODER_PATH, "rb") as f:
            cross_encoder = pickle.load(f)
    else:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=DEVICE)
        with open(CROSSENCODER_PATH, "wb") as f:
            pickle.dump(cross_encoder, f)

    return bm25_model, cross_encoder