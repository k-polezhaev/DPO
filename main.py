import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update

from app.AI.AI import QAModel, RETRIEVER_MODEL_PATH, DATA_PATH, QUESTION_COLUMN, ANSWER_COLUMN, MAX_QUESTIONS_TO_LOAD, TOP_N_LOGGING

from app.handlers import (
    start_command,
    handle_message,
    handle_button_zadat_vopros,
    handle_button_contacts,
    handle_button_storage,
    handle_button_about
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = 'your_token'


def main() -> None:
    """Запускает бота."""
    logger.info("Запуск бота...")

    qa_model_instance = QAModel(
        model_path=RETRIEVER_MODEL_PATH,
        data_path=DATA_PATH,
        question_col=QUESTION_COLUMN,
        answer_col=ANSWER_COLUMN,
        max_questions=MAX_QUESTIONS_TO_LOAD,
        top_n_logging=TOP_N_LOGGING
    )

    if not qa_model_instance.load_resources():
        logger.error("Не удалось загрузить AI ресурсы. Завершение работы.")
        return

    application = Application.builder().token(TOKEN).build()
    logger.info("Application PTB создан.")

    application.bot_data['qa_model'] = qa_model_instance
    logger.info("Инстанс QAModel успешно передан в application.bot_data.")

    application.add_handler(CommandHandler("start", start_command))

    application.add_handler(MessageHandler(filters.Text('Задать вопрос'), handle_button_zadat_vopros))
    application.add_handler(MessageHandler(filters.Text('Контакты создателей'), handle_button_contacts))
    application.add_handler(MessageHandler(filters.Text('Главное хранилище'), handle_button_storage))
    application.add_handler(MessageHandler(filters.Text('Коротко о проекте'), handle_button_about))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    logger.info("Бот готов и запускает polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

    logger.info("Бот остановлен.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Бот выключен вручную (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Произошла ошибка при выполнении бота: {e}", exc_info=True)