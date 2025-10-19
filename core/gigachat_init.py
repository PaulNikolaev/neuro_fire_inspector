import os
from langchain_community.chat_models.gigachat import GigaChat

# Получаем токен GigaChat из переменной окружения
GIGACHAT_TOKEN = os.getenv("GIGACHAT_API_KEY")

# Инициализация основного чата
chat = GigaChat(
    credentials=GIGACHAT_TOKEN,
    # model="GigaChat-Pro",
    verify_ssl_certs=False,
    streaming=False
)

# Дополнительный чат для валидации / безопасных ответов
guard_chat = GigaChat(
    credentials=GIGACHAT_TOKEN,
    # model="GigaChat",
    verify_ssl_certs=False,
    streaming=False
)

print("✅ GigaChat инициализирован успешно.")
