from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Старт"), KeyboardButton(text="Инфо")]
    ],
    resize_keyboard=True,
    one_time_keyboard=False
)

# Inline кнопки для информации
info_inline = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Сайт ГОСТ", url="https://www.gost.ru")],
        [InlineKeyboardButton(text="Контакты МЧС", url="https://www.mchs.ru")]
    ]
)
