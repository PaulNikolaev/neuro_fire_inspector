import os
import re
from typing import Dict
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document


def extract_text_from_scanned_pdf(path: str, lang: str = "rus") -> str:
    """
    Извлечение текста с отсканированного PDF с помощью OCR.
    """
    text = ""
    try:
        pages = convert_from_path(path)
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page, lang=lang)
            text += f"\n--- Page {i + 1} ---\n{page_text}"
    except Exception as e:
        print(f"⚠️ Ошибка OCR для {os.path.basename(path)}: {e}")
    return text.strip()


def extract_text_from_pdf(path: str, min_length: int = 100) -> str:
    """
    Извлечение текста из PDF. Если текст слишком короткий, применяем OCR.
    """
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ Ошибка чтения PDF {os.path.basename(path)}: {e}")

    # нормализация переносов строк под Windows
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # если текста мало, используем OCR
    if len(text.strip()) < min_length:
        print(f"🔍 Используем OCR для {os.path.basename(path)}")
        text = extract_text_from_scanned_pdf(path)

    # удаляем лишние пробелы
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_articles(text: str) -> Dict[str, str]:
    """
    Разбор текста нормативного акта на статьи.
    Возвращает словарь: {название статьи: текст статьи}
    """
    # нормализация переносов строк
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    articles = {}
    # паттерн ищет "Статья N" или "Статья N.M" и текст до следующей статьи
    pattern = re.compile(r"(Статья\s+\d+[\.\d]*)(.*?)(?=Статья\s+\d+|\Z)", re.DOTALL)
    for article, content in pattern.findall(text):
        articles[article.strip()] = content.strip()
    return articles


def load_documents_from_pdf_dir(pdf_dir: str) -> list[Document]:
    """
    Загружает все PDF из директории, парсит статьи и возвращает список Document.
    """
    structured_documents = []
    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, filename)
        print(f"📘 Обрабатываем: {filename}")
        text = extract_text_from_pdf(path)
        if not text:
            print(f"⚠️ Не удалось извлечь текст из {filename}")
            continue

        articles = parse_articles(text)
        for article, content in articles.items():
            structured_documents.append(
                Document(
                    page_content=content,
                    metadata={"source": filename, "article": article}
                )
            )

    print(f"✅ Загружено {len(structured_documents)} статей из {pdf_dir}")
    return structured_documents
