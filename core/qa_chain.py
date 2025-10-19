import os
import re
from typing import Dict, Any
from core.gigachat_init import chat, guard_chat
from core.retriever_init import retriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

# ==========================
# Пути к промтам
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")


def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Очистка текста от Markdown и лишних переносов."""
    text = re.sub(r"\*+", "", text)  # убрать звёздочки
    text = re.sub(r"---+", "", text)  # убрать ---
    text = re.sub(r"\n{2,}", "\n", text)  # убрать лишние пустые строки
    return text.strip()


# ==========================
# Загрузка промтов
# ==========================
GUARDRAIL_TEMPLATE = load_prompt("guardrail_prompt.txt")
INSPECTOR_TEMPLATE = load_prompt("inspector_prompt.txt")

# ==========================
# Guardrail
# ==========================
GUARDRAIL_PROMPT = PromptTemplate(template=GUARDRAIL_TEMPLATE, input_variables=["query"])
guardrail_chain = LLMChain(llm=guard_chat, prompt=GUARDRAIL_PROMPT)

# ==========================
# RAG (Retrieval + QA)
# ==========================
INSPECTOR_PROMPT = PromptTemplate(template=INSPECTOR_TEMPLATE, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": INSPECTOR_PROMPT},
    return_source_documents=True
)


# ==========================
# Функция запроса к инспектору
# ==========================
def ask_inspector(query: str) -> Dict[str, Any]:
    print(f"\n🔹 Вопрос: {query}")

    # Guardrail проверка
    guard_result = guardrail_chain.invoke({"query": query})
    if isinstance(guard_result, dict):
        guard_text = guard_result.get("text", "").strip().upper()
    else:
        guard_text = str(guard_result).strip().upper()

    print(f"🛡 Guardrail: {guard_text}")

    if guard_text in ["НЕТ", "НЕИЗВЕСТНО"]:
        return {
            "result": "💡 Я могу отвечать только на вопросы по пожарной безопасности, нормативам и инструкциям.",
            "source_documents": []
        }

    # RAG: получение релевантных документов и формирование контекста
    retriever_docs = retriever._get_relevant_documents(query)
    context_text = "\n".join([clean_text(doc.page_content) for doc in retriever_docs])
    prompt_text = INSPECTOR_PROMPT.format(context=context_text, question=query)
    response_msg = chat.invoke(prompt_text)

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
