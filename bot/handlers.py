from aiogram import Router, types
from aiogram.filters import CommandStart
from core.qa_chain import ask_inspector

router = Router()


@router.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Здравствуйте! Я Нейро-инспектор пожарного надзора. Задайте ваш вопрос.")


@router.message()
async def handle_query(message: types.Message):
    query = message.text.strip()
    await message.answer("🔎 Обрабатываю запрос, пожалуйста подождите...")

    result = ask_inspector(query)

    await message.answer(result['result'])


def register_handlers(dp):
    dp.include_router(router)
