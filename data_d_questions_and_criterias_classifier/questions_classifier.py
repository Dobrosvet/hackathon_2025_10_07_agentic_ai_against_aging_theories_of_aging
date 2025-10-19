"""
Questions and Criterias Classifier using Bioformer-8L
Классифицирует статьи по 9 исследовательским вопросам и 4 критериям
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class QuestionsClassifier:
    """
    Классификатор для ответов на 9 вопросов и 4 критерия
    Использует Bioformer-8L для анализа текста
    """

    def __init__(
        self,
        config_path: str = None,
        use_gpu: bool = True,
        quantize: bool = False
    ):
        """
        Инициализация классификатора

        Args:
            config_path: Путь к config.yaml
            use_gpu: Использовать GPU если доступен
            quantize: Использовать quantization
        """
        # Загрузить конфигурацию
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.questions = self.config.get("questions", {})
        self.criteria = self.config.get("criteria", {})
        self.classifier_config = self.config.get("classifier", {})

        # Bioformer модель (ленивая загрузка)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialized = False
        self._use_gpu = use_gpu
        self._quantize = quantize

        logger.info(f"QuestionsClassifier created (lazy loading)")

    def _initialize(self):
        """Ленивая инициализация Bioformer модели"""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            model_name = self.classifier_config.get("model_name", "bioformers/bioformer-8L")

            # Определить device
            if self._use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Загрузить tokenizer и model
            logger.info(f"Loading Bioformer model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Quantization
            if self._quantize and self.device.type == "cuda":
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    logger.info("Applied int8 quantization")
                except Exception as e:
                    logger.warning(f"Quantization failed: {e}")

            self.model.to(self.device)
            self.model.eval()

            self._initialized = True
            logger.info("Bioformer model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Bioformer: {e}")
            raise

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Получить embedding текста

        Args:
            text: Входной текст

        Returns:
            Numpy array с embedding
        """
        import torch

        # Токенизация
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.classifier_config.get("max_length", 512),
            return_tensors="pt"
        )

        # Перенести на device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Получить embedding
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Использовать [CLS] token
        embedding = outputs.last_hidden_state[:, 0, :]

        # Нормализация
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()[0]

    def _classify_question(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Классифицировать текст по одному вопросу

        Args:
            text: Текст статьи
            question_id: ID вопроса (Q1, Q2, ...)
            question_config: Конфигурация вопроса

        Returns:
            Результат классификации
        """
        self._initialize()

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Создать промпт для вопроса
        prompt = f"Question: {question_text}\n\nPaper text: {text[:2000]}"

        # Получить embedding
        text_emb = self._get_text_embedding(text)

        # Для binary вопросов: создаём embeddings "Yes" и "No"
        if question_type == "binary":
            yes_prompt = f"{prompt}\n\nAnswer: Yes"
            no_prompt = f"{prompt}\n\nAnswer: No"

            yes_emb = self._get_text_embedding(yes_prompt)
            no_emb = self._get_text_embedding(no_prompt)

            # Вычислить similarity
            yes_sim = float(np.dot(text_emb, yes_emb))
            no_sim = float(np.dot(text_emb, no_emb))

            # Выбрать ответ
            if yes_sim > no_sim:
                answer = True
                confidence = yes_sim
            else:
                answer = False
                confidence = no_sim

            # Найти подтверждающие фрагменты
            fragments = self._extract_supporting_fragments(
                text,
                question_text,
                answer
            )

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "fragments": fragments
            }

        # Для multiclass (Q1)
        elif question_type == "multiclass":
            options = question_config.get("options", [])
            option_embeddings = []

            for option in options:
                option_prompt = f"{prompt}\n\nAnswer: {option}"
                option_emb = self._get_text_embedding(option_prompt)
                similarity = float(np.dot(text_emb, option_emb))
                option_embeddings.append((option, similarity))

            # Выбрать вариант с максимальной similarity
            best_option, best_similarity = max(option_embeddings, key=lambda x: x[1])

            fragments = self._extract_supporting_fragments(
                text,
                question_text,
                best_option
            )

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_similarity, 3),
                "fragments": fragments,
                "all_options": [
                    {"option": opt, "similarity": round(sim, 3)}
                    for opt, sim in option_embeddings
                ]
            }

    def _extract_supporting_fragments(
        self,
        text: str,
        question: str,
        answer: Any,
        max_fragments: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Извлечь фрагменты текста, подтверждающие ответ

        Args:
            text: Полный текст
            question: Текст вопроса
            answer: Ответ
            max_fragments: Максимальное количество фрагментов

        Returns:
            Список фрагментов с позициями
        """
        # Разбить текст на параграфы/предложения
        sentences = self._split_into_sentences(text)

        if len(sentences) == 0:
            return []

        # Создать query для поиска релевантных фрагментов
        query = f"{question} Answer: {answer}"
        query_emb = self._get_text_embedding(query)

        # Вычислить similarity каждого предложения с query
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Пропустить короткие
                continue

            sent_emb = self._get_text_embedding(sentence)
            similarity = float(np.dot(query_emb, sent_emb))

            # Найти позицию в оригинальном тексте
            start_pos = text.find(sentence)
            end_pos = start_pos + len(sentence) if start_pos != -1 else -1

            if start_pos != -1:
                sentence_scores.append({
                    "text": sentence,
                    "start_position": start_pos,
                    "end_position": end_pos,
                    "confidence": round(similarity, 3)
                })

        # Сортировать по similarity
        sentence_scores.sort(key=lambda x: x["confidence"], reverse=True)

        # Вернуть топ фрагментов
        return sentence_scores[:max_fragments]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Разбить текст на предложения

        Args:
            text: Входной текст

        Returns:
            Список предложений
        """
        # Простое разбиение по точкам (можно улучшить с помощью nltk)
        import re

        # Разбить по . ! ?
        sentences = re.split(r'[.!?]\s+', text)

        # Очистить
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return sentences

    def classify_paper(self, full_text: str) -> Dict[str, Any]:
        """
        Классифицировать статью по всем вопросам и критериям

        Args:
            full_text: Полный текст статьи

        Returns:
            Результаты классификации
        """
        logger.info("Starting paper classification...")

        results = {
            "questions": {},
            "criteria": {},
            "timestamp": None
        }

        # Классифицировать по вопросам Q1-Q9
        for q_id, q_config in self.questions.items():
            logger.info(f"Classifying {q_id}...")
            result = self._classify_question(full_text, q_id, q_config)
            results["questions"][q_id] = result

        # Классифицировать по критериям C1-C4
        for c_id, c_config in self.criteria.items():
            logger.info(f"Classifying {c_id}...")
            result = self._classify_question(full_text, c_id, c_config)
            results["criteria"][c_id] = result

        from datetime import datetime
        results["timestamp"] = datetime.now().isoformat()

        logger.info("Classification complete")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели"""
        self._initialize()

        import torch

        info = {
            "model_name": self.classifier_config.get("model_name"),
            "device": str(self.device),
            "initialized": self._initialized,
            "num_questions": len(self.questions),
            "num_criteria": len(self.criteria)
        }

        if self._initialized and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)

        return info


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)

    print("=== Questions Classifier Test ===\n")

    classifier = QuestionsClassifier(use_gpu=True)

    test_text = """
    Aging is characterized by progressive decline in cellular function and increased
    susceptibility to disease. Telomere shortening has been proposed as a biomarker
    of biological aging. Studies have shown that telomere length is inversely correlated
    with chronological age and mortality risk across different species. Furthermore,
    interventions such as calorie restriction have been demonstrated to slow telomere
    attrition and extend lifespan in multiple vertebrate species. The mitochondrial
    theory of aging suggests that accumulated damage to mitochondrial DNA contributes
    to age-related functional decline. Large mammals tend to have longer lifespans
    than smaller ones, which may be explained by differences in metabolic rates and
    oxidative stress levels.
    """

    print("Classifying test paper...")
    results = classifier.classify_paper(test_text)

    print(f"\nQuestions Classification Results:")
    for q_id, result in results["questions"].items():
        print(f"\n{q_id}:")
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Fragments found: {len(result['fragments'])}")

    print(f"\nCriteria Classification Results:")
    for c_id, result in results["criteria"].items():
        print(f"\n{c_id}:")
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Fragments found: {len(result['fragments'])}")

    print(f"\nModel info:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
