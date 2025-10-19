import os
import re
from typing import Dict, Any
from core.gigachat_init import chat, guard_chat
from core.retriever_init import retriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

# Пути к промтам
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")


def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Очистка текста от Markdown-маркеров заголовков, цитат и лишних переносов."""
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"---+", "", text)
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
    return text.strip()


# Загрузка промтов
GUARDRAIL_TEMPLATE = load_prompt("guardrail_prompt.txt")
INSPECTOR_TEMPLATE = load_prompt("inspector_prompt.txt")

#  Константы
START_INFO_RESPONSE = (
    "👋 Привет! Я — Нейро-инспектор государственного пожарного надзора. "
    "Моя задача — консультировать вас по вопросам пожарной безопасности, "
    "законодательства РФ и нормативным актам (ФЗ, Постановления, СП, Приказы). "
    "Просто задайте свой вопрос по теме!"
)
# Ответ для вопросов "О СЕБЕ"
ABOUT_ME_RESPONSE = (
    "💡 Я — Нейро-инспектор государственного пожарного надзора. "
    "Моя основная задача — предоставлять точную информацию по пожарной безопасности, "
    "опираясь на нормативные акты РФ. Я могу проконсультировать вас по требованиям "
    "законодательства, служебным инструкциям и помочь найти ответы в предоставленной базе знаний. "
    "Спрашивайте!"
)
# Ответ для вопросов "НЕТ" (не по теме)
OFF_TOPIC_RESPONSE = "💡 Я могу отвечать только на вопросы по пожарной безопасности, нормативам и инструкциям."
# Список командных слов, которые Guardrail должен пропустить
EXCEPTIONS = ["старт", "инфо", "/start", "/info"]

# Guardrail
GUARDRAIL_PROMPT = PromptTemplate(template=GUARDRAIL_TEMPLATE, input_variables=["query"])
guardrail_chain = LLMChain(llm=guard_chat, prompt=GUARDRAIL_PROMPT)

# RAG (Retrieval + QA)
INSPECTOR_PROMPT = PromptTemplate(template=INSPECTOR_TEMPLATE, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": INSPECTOR_PROMPT},
    return_source_documents=True
)


# Функция запроса к инспектору
def ask_inspector(query: str) -> Dict[str, Any]:
    print(f"\n🔹 Вопрос: {query}")

    # Нормализуем запрос для проверки на исключения
    normalized_query = query.strip().lower()

    # Обработка команд-исключений (СТАРТ, ИНФО)
    if normalized_query in EXCEPTIONS:
        print(f"Обработка исключения: {normalized_query}")
        return {
            "result": START_INFO_RESPONSE,
            "source_documents": []
        }

    # Guardrail проверка
    guard_result = guardrail_chain.invoke({"query": query})

    # Извлечение и нормализация ответа Guardrail
    if isinstance(guard_result, dict):
        guard_text = guard_result.get("text", "").strip().upper()
    else:
        guard_text = str(guard_result).strip().upper()

    # Дополнительная чистка (оставляем только русские заглавные буквы)
    guard_text = re.sub(r"[^А-Я\s]", "", guard_text).strip()

    # Обработка результатов Guardrail
    if guard_text == "О СЕБЕ":
        return {
            "result": ABOUT_ME_RESPONSE,
            "source_documents": []
        }

    if guard_text == "НЕТ" or not guard_text:
        return {
            "result": OFF_TOPIC_RESPONSE,
            "source_documents": []
        }

    # RAG: выполнение цепочки для "ДА" (пожарная безопасность)
    if guard_text == "ДА":
        retriever_docs = retriever.get_relevant_documents(query)

        context_text = "\n\n---\n\n".join([clean_text(doc.page_content) for doc in retriever_docs])

        prompt_text = INSPECTOR_PROMPT.format(context=context_text, question=query)
        response_msg = chat.invoke(prompt_text)

        response_text = ""
        if hasattr(response_msg, "content"):
            response_text = response_msg.content
        elif isinstance(response_msg, dict) and "text" in response_msg:
            response_text = response_msg["text"]
        else:
            response_text = str(response_msg)

        response_text = clean_text(response_text)

        return {
            "result": response_text,
            "source_documents": retriever_docs
        }

    return {
        "result": OFF_TOPIC_RESPONSE,
        "source_documents": []
    }
