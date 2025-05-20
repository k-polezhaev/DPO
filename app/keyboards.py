from telegram import ReplyKeyboardMarkup, KeyboardButton

main = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text='Задать вопрос')],
        [KeyboardButton(text='Контакты создателей')],
        [KeyboardButton(text='Главное хранилище')],
        [KeyboardButton(text='Коротко о проекте')]
        ],
    resize_keyboard=True,
    input_field_placeholder='Выберите пункт...'
    )

