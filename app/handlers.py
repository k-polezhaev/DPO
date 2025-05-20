import logging
from telegram import Update
from telegram.ext import ContextTypes

import app.keyboards as kb

from app.AI.AI import QAModel

logger = logging.getLogger(__name__)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Получена команда /start от {user.username} ({user.id}).")
    await update.message.reply_html(
        f"Привет, {user.mention_html()}! Я бот, который может ответить на вопросы по магазину одежды. Спросите меня о чем-нибудь!",
        reply_markup=kb.main
    )

async def handle_button_zadat_vopros(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Нажата кнопка 'Задать вопрос' от {update.effective_user.username}")
    await update.message.reply_text( f"Слушаю твой вопрос!",)


async def handle_button_contacts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Нажата кнопка 'Контакты создателей' от {update.effective_user.username}")
    await update.message.reply_text('Вот контакты этих ребят:')
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id, 'Литвиненко Эммануил - https://vk.com/iamdiedinside')
    await context.bot.send_message(chat_id, 'Попов Виталий - https://vk.com/pvitalys')
    await context.bot.send_message(chat_id, 'Полежаев Константин - https://vk.com/pojodul')
    await context.bot.send_message(chat_id, 'Милько Артём - https://vk.com/aaaaaaa_aaaa_a')
    await context.bot.send_message(chat_id, 'Мурсекаев Ильяс - https://vk.com/mylogin_7958_password_9ukj4ixxxx')


async def handle_button_storage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Нажата кнопка 'Главное хранилище' от {update.effective_user.username}")
    await update.message.reply_text('Вот ссылка на главное хранилище: ссылка[https://drive.google.com/drive/folders/11NH-FT2Mtcy8AcUBanpYeH-vTRZARabO?usp=sharing]')


async def handle_button_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Нажата кнопка 'Коротко о проекте' от {update.effective_user.username}")
    await update.message.reply_text('Наш готовый проект — это Telegram-чат-бот, который отвечает на часто задаваемые вопросы на русском языке с помощью нейросетей. Он написан на Python и использует предобученные модели. Команда распределила роли: Эммануил Литвиненко координировал все этапы как тимлид, Константин Полежаев и Ильяс Мурсекаев занимались кодом и аналитикой, Артём Милько собирал и редактировал датасет, а Виталий Попов оформлял документацию и презентацию. 🚀')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_question = update.message.text
    user = update.effective_user
    logger.info(f"Получен вопрос от @{user.username} ({user.id}): {user_question}")

    if 'qa_model' not in context.application.bot_data or not isinstance(context.application.bot_data['qa_model'], QAModel):
         logger.error("Инстанс QAModel не найден в context.application.bot_data.")
         await update.message.reply_text("Извините, система обработки вопросов недоступна. Пожалуйста, попробуйте позже.")
         return

    qa_model: QAModel = context.application.bot_data['qa_model']

    answer = qa_model.getBestAnswer(user_question)

    await update.message.reply_text(answer)

