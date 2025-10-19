from aiogram import Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import F

from core.qa_chain import ask_inspector
from bot.keyboards import main_menu, info_inline


# 1. Обработчик команды /start
async def start_handler(message: Message):
    await message.answer(
        "Привет! Я Нейро-инспектор по пожарной безопасности. 🔥\n"
        "Вы можете задать мне вопросы по нормативам, инструкциям и законам РФ.",
        reply_markup=main_menu
    )


# 2. Обработчик команды /info
async def info_handler(message: Message):
    await message.answer(
        "Я могу помочь с вопросами по пожарной безопасности, нормативам, инструкциям и законам РФ.\n"
        "Задайте вопрос или выберите кнопку меню.",
        reply_markup=info_inline  # Inline-клавиатура для ссылок
    )


# 3. text_handler (основная логика)
async def text_handler(message: Message):
    query = message.text.strip()
    if not query:
        await message.answer("❌ Пустой запрос. Пожалуйста, напишите вопрос.", reply_markup=main_menu)
        return

    result = ask_inspector(query)
    await message.answer(result["result"], reply_markup=main_menu)


def register_handlers(dp: Dispatcher):
    # 1. Обработка команды /start
    dp.message.register(start_handler, Command("start"))

    # 2. Обработка команды /info
    dp.message.register(info_handler, Command("info"))

    # 3. Обработка нажатий на кнопки "Старт" и "Инфо"
    dp.message.register(start_handler, F.text.lower() == "старт")
    dp.message.register(info_handler, F.text.lower() == "инфо")

    # 4. Регистрация для любого другого текста (исключает только Старт и Инфо)
    dp.message.register(
        text_handler,
        F.text.lower().not_in(["старт", "инфо"]),
        F.text
    )
