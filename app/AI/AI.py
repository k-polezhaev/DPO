import os
import re
import logging # Импортируем logging
from gensim.utils import simple_preprocess
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import random

logger = logging.getLogger(__name__)

DATA_PATH = r'' # <--- Укажите актуальный путь к вашему файлу

QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer'

RETRIEVER_MODEL_PATH = r'' # <--- ВСТАВЬТЕ СЮДА ПУТЬ К ПАПКЕ С ВАШЕЙ ОБУЧЕННОЙ ST МОДЕЛЬЮ

MAX_QUESTIONS_TO_LOAD = 1000
TOP_N_LOGGING = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используемое устройство для AI: {DEVICE}")


def preprocess_text(text: str) -> str:
    """
    Предобрабатывает текст: приводит к нижнему регистру, удаляет пунктуацию и цифры,
    токенезирует (через simple_preprocess) и удаляет диакритические знаки.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = simple_preprocess(text, deacc=True)
    return ' '.join(tokens)

def prepare_question_corpus_and_mapping(data_path: str, question_col: str, answer_col: str, max_questions: int | None = None):
    """
    Загружает датасет, создает корпус уникальных предобработанных вопросов
    и маппинг от этих вопросов к исходным парам (вопрос, ответ).
    """
    try:
        logger.info(f"Загрузка датасета из файла: {data_path}...")
        df = pd.read_csv(data_path)
        if question_col not in df.columns or answer_col not in df.columns:
            logger.error(f"Ошибка: Колонки '{question_col}' или '{answer_col}' не найдены в файле {data_path}.")
            return None, None

        preprocessed_unique_questions_corpus = []
        unique_q_to_raw_pairs = {}

        logger.info("Обработка и создание корпуса вопросов и маппинга к ответам...")
        for index, row in df.iterrows():
            raw_q = str(row[question_col]) if pd.notna(row[question_col]) else ""
            raw_answer = str(row[answer_col]) if pd.notna(row[answer_col]) else ""

            preprocessed_q = preprocess_text(raw_q)

            if preprocessed_q.strip():
                if preprocessed_q not in unique_q_to_raw_pairs:
                    unique_q_to_raw_pairs[preprocessed_q] = []
                    preprocessed_unique_questions_corpus.append(preprocessed_q)

                unique_q_to_raw_pairs[preprocessed_q].append((raw_q, raw_answer))

        if not preprocessed_unique_questions_corpus:
            logger.error(f"Не найдено непустых вопросов для корпуса после предобработки в колонке '{question_col}'.")
            return None, None

        logger.info(f"Создан корпус из {len(preprocessed_unique_questions_corpus)} уникальных предобработанных вопросов.")

        if max_questions is not None and len(preprocessed_unique_questions_corpus) > max_questions:
            logger.info(f"Ограничиваем количество вопросов корпуса до {max_questions} (случайная выборка).")
            random.seed(42)
            sampled_preprocessed_corpus = random.sample(preprocessed_unique_questions_corpus, max_questions)

            sampled_unique_q_to_raw_pairs = {
                q: unique_q_to_raw_pairs[q] for q in sampled_preprocessed_corpus
            }

            return sampled_preprocessed_corpus, sampled_unique_q_to_raw_pairs
        else:
            return preprocessed_unique_questions_corpus, unique_q_to_raw_pairs

    except FileNotFoundError:
        logger.error(f"Ошибка: Файл датасета не найден по пути: {data_path}")
        return None, None
    except Exception as e:
        logger.error(f"Ошибка при чтении или обработке файла датасета: {e}", exc_info=True)
        return None, None


# --- Класс для хранения ресурсов AI и логики поиска ---
class QAModel:
    def __init__(self, model_path: str, data_path: str, question_col: str, answer_col: str, max_questions: int | None, top_n_logging: int):
        self.model_path = model_path
        self.data_path = data_path
        self.question_col = question_col
        self.answer_col = answer_col
        self.max_questions = max_questions
        self.top_n_logging = top_n_logging

        self.model = None
        self.preprocessed_q_corpus = None
        self.q_to_raw_answer_map = None
        self.question_corpus_embeddings = None

    def load_resources(self):
        """
        Загружает модель ретривера, данные, готовит корпус вопросов
        и кодирует его для быстрого поиска. Выполняется один раз при старте.
        """
        logger.info("Загрузка ресурсов AI...")

        if self.model_path is None or not os.path.isdir(self.model_path):
            logger.error(f"ОШИБКА: Не указан или неверен путь к папке с сохраненной моделью ретривера: {self.model_path}")
            return False

        self.preprocessed_q_corpus, self.q_to_raw_answer_map = prepare_question_corpus_and_mapping(
            self.data_path, self.question_col, self.answer_col, self.max_questions
        )

        if not self.preprocessed_q_corpus or not self.q_to_raw_answer_map:
            logger.error("Не удалось подготовить корпус вопросов и маппинг. AI не может быть инициализирован.")
            return False

        try:
            self.model = SentenceTransformer(self.model_path, device=DEVICE)
            logger.info("Модель ретривера успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели ретривера из {self.model_path}: {e}", exc_info=True)
            return False

        logger.info("Кодирование корпуса вопросов...")
        try:
             encode_batch_size = 64 if DEVICE.type == 'cuda' else 32
             self.question_corpus_embeddings = self.model.encode(
                 self.preprocessed_q_corpus,
                 convert_to_tensor=True,
                 device=DEVICE,
                 show_progress_bar=False,
                 batch_size=encode_batch_size
             )
             logger.info(f"{len(self.preprocessed_q_corpus)} вопросов корпуса успешно закодировано.")
        except Exception as e:
             logger.error(f"Ошибка при кодировании корпуса вопросов: {e}", exc_info=True)
             return False

        logger.info("Ресурсы AI загружены успешно.")
        return True

    def getBestAnswer(self, user_question: str) -> str:
        """
        Основная логика поиска ответа по Retrieval-подходу.
        Ищет наиболее похожий вопрос в корпусе и возвращает лучший ответ.
        Топ-N результатов логируется.
        """
        model = self.model
        preprocessed_q_corpus = self.preprocessed_q_corpus
        q_to_raw_answer_map = self.q_to_raw_answer_map
        question_corpus_embeddings = self.question_corpus_embeddings

        if model is None or preprocessed_q_corpus is None or q_to_raw_answer_map is None or question_corpus_embeddings is None:
            logger.error("AI ресурсы не загружены. Невозможно обработать вопрос.")
            return "Извините, система обработки вопросов недоступна."

        preprocessed_user_question = preprocess_text(user_question)
        if not preprocessed_user_question.strip():
            logger.info(f"Вопрос пользователя стал пустым после предобработки: '{user_question}'")
            return "Извините, ваш вопрос не содержит осмысленного текста."

        try:
            user_question_embedding = model.encode(
                preprocessed_user_question,
                convert_to_tensor=True,
                device=DEVICE,
                show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Ошибка при кодировании вопроса пользователя '{user_question}': {e}", exc_info=True)
            return "Произошла внутренняя ошибка при обработке вашего вопроса."

        try:
            similarities = util.pytorch_cos_sim(user_question_embedding, question_corpus_embeddings)
        except Exception as e:
            logger.error(f"Ошибка при вычислении сходства для вопроса '{user_question}': {e}", exc_info=True)
            return "Произошла внутренняя ошибка при поиске ответа."

        if similarities is None or similarities.numel() == 0:
            logger.warning(f"Не удалось рассчитать схожесть для вопроса '{user_question}'.")
            return "Извините, не удалось выполнить поиск в базе."

        scores = similarities[0].cpu().tolist()

        scored_preprocessed_questions = []
        for i, score_val in enumerate(scores):
            scored_preprocessed_questions.append({
                'preprocessed_q': preprocessed_q_corpus[i],
                'score': score_val
            })
        scored_preprocessed_questions.sort(key=lambda x: x['score'], reverse=True)

        best_answer_text = "Извините, не удалось найти подходящий ответ в базе."
        log_results = []

        if scored_preprocessed_questions:
            top1_match = scored_preprocessed_questions[0]
            preprocessed_matched_q_top1 = top1_match['preprocessed_q']
            score_top1 = top1_match['score']

            original_pairs_top1 = q_to_raw_answer_map.get(preprocessed_matched_q_top1, [])

            if original_pairs_top1:
                 best_answer_text = original_pairs_top1[0][1]

            for i in range(min(self.top_n_logging, len(scored_preprocessed_questions))):
                 match = scored_preprocessed_questions[i]
                 preprocessed_q = match['preprocessed_q']
                 score = match['score']
                 original_pairs_for_log = q_to_raw_answer_map.get(preprocessed_q, [])

                 log_results.append({
                     'rank': i + 1,
                     'score': score,
                     'preprocessed_q': preprocessed_q,
                     'original_pairs': original_pairs_for_log
                 })

        logger.info(f"--- Топ {self.top_n_logging} результатов для вопроса '{user_question}' ---")
        if log_results:
            for res in log_results:
                logger.info(f"  Ранг {res['rank']}: Схожесть={res['score']:.4f}")
                logger.info(f"    Предобраб. вопрос: {res['preprocessed_q'][:150]}{'...' if len(res['preprocessed_q']) > 150 else ''}")
                if res['original_pairs']:
                    for j, (raw_q, raw_ans) in enumerate(res['original_pairs']):
                         logger.info(f"    Исходный вопрос ({j+1}): {raw_q[:150]}{'...' if len(raw_q) > 150 else ''}")
                         logger.info(f"    Ответ ({j+1}): {raw_ans[:200]}{'...' if len(raw_ans) > 200 else ''}")
                else:
                    logger.info("    Нет исходных пар в маппинге для этого результата.")
            logger.info("-------------------------------------------------")
        else:
            logger.info("  Не найдено подходящих результатов.")
            logger.info("-------------------------------------------------")

        return best_answer_text

if __name__ == '__main__':
    print("Выполняется проверка AI.py...")

    TEST_MODEL_PATH = None

    if TEST_MODEL_PATH and os.path.isdir(TEST_MODEL_PATH):
        # Убедимся, что DATA_PATH существует для проверки
        if not os.path.exists(DATA_PATH):
             print(f"Ошибка: Файл данных {DATA_PATH} не найден для проверки AI.py.")
        else:
            TEMP_DATA_PATH = DATA_PATH
            TEMP_QUESTION_COLUMN = QUESTION_COLUMN
            TEMP_ANSWER_COLUMN = ANSWER_COLUMN
            TEMP_MAX_QUESTIONS_TO_LOAD = 100 # Ограничим для теста
            TEMP_TOP_N_LOGGING = 5 # Логируем топ 5 для теста

            qa_model_instance = QAModel(
                model_path=TEST_MODEL_PATH,
                data_path=TEMP_DATA_PATH,
                question_col=TEMP_QUESTION_COLUMN,
                answer_col=TEMP_ANSWER_COLUMN,
                max_questions=TEMP_MAX_QUESTIONS_TO_LOAD,
                top_n_logging=TEMP_TOP_N_LOGGING
            )

            if qa_model_instance.load_resources():
                print("\nAI ресурсы успешно загружены для проверки.")
                print("Введите тестовые вопросы (или 'выход' для завершения):")
                while True:
                    test_question = input("> ")
                    if test_question.lower() == 'выход':
                        break
                    if not test_question.strip():
                        continue
                    # getBestAnswer теперь логирует внутри себя
                    answer = qa_model_instance.getBestAnswer(test_question)
                    print(f"Ответ (для пользователя): {answer}")


            else:
                print("\nНе удалось загрузить AI ресурсы для проверки.")
    else:
        print("\nПуть TEST_MODEL_PATH не указан или неверен/не существует. Проверка AI.py пропущена.")