"""
Bioformer-8L классификатор для анализа теорий старения
Использует реальную модель bioformers/bioformer-8L из HuggingFace
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BioformerClassifier:
    """
    Обертка над Bioformer-8L для классификации текстов и NER
    Поддерживает GPU, batching и quantization
    """

    def __init__(
        self,
        model_name: str = "bioformers/bioformer-8L",
        use_gpu: bool = True,
        quantize: bool = False,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Инициализация Bioformer классификатора

        Args:
            model_name: Название модели на HuggingFace
            use_gpu: Использовать GPU если доступен
            quantize: Использовать int8 quantization
            batch_size: Размер батча для обработки
            max_length: Максимальная длина токенов
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = None
        self.model = None
        self.tokenizer = None
        self.theory_embeddings = None

        # Отложенная загрузка - только при первом использовании
        self._initialized = False
        self._use_gpu = use_gpu
        self._quantize = quantize

        logger.info(f"BioformerClassifier created (lazy loading enabled)")

    def _initialize(self):
        """
        Ленивая инициализация - загрузка модели при первом использовании
        Это экономит память если Bioformer не используется
        """
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            # Определить device
            if self._use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Загрузить tokenizer
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Загрузить модель
            logger.info(f"Loading model from {self.model_name}...")
            self.model = AutoModel.from_pretrained(self.model_name)

            # Quantization для ускорения
            if self._quantize and self.device.type == "cuda":
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    logger.info("Applied int8 quantization")
                except Exception as e:
                    logger.warning(f"Quantization failed: {e}, using full precision")

            self.model.to(self.device)
            self.model.eval()  # Evaluation mode

            # Предвычислить embeddings теорий старения
            self._precompute_theory_embeddings()

            self._initialized = True
            logger.info("Bioformer model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            logger.error("Please install: pip install torch transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Bioformer: {e}")
            raise

    def _precompute_theory_embeddings(self):
        """
        Предвычислить embeddings для всех теорий старения
        Это ускоряет классификацию (вычисляется 1 раз)
        """
        import torch
        from theory_database import theory_db

        logger.info("Precomputing theory embeddings...")

        theory_texts = []
        theory_names = []

        # Создать короткие описания теорий для embedding
        for theory_name, patterns in theory_db.get_all_theories().items():
            # Используем первые 3 паттерна как описание теории
            theory_text = f"aging theory: {', '.join(patterns[:3])}"
            theory_texts.append(theory_text)
            theory_names.append(theory_name)

        # Получить embeddings батчами
        embeddings = []
        for i in range(0, len(theory_texts), self.batch_size):
            batch_texts = theory_texts[i:i + self.batch_size]
            batch_embeddings = self._get_embeddings_batch(batch_texts)
            embeddings.extend(batch_embeddings)

        # Сохранить как numpy array для быстрого поиска
        self.theory_embeddings = {
            "names": theory_names,
            "vectors": np.array(embeddings)
        }

        logger.info(f"Precomputed embeddings for {len(theory_names)} theories")

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Получить embeddings для батча текстов

        Args:
            texts: Список текстов

        Returns:
            Список векторов embeddings
        """
        import torch

        # Токенизация
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Перенести на device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Получить embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Использовать [CLS] token embedding (первый токен)
        # или mean pooling всех токенов
        # Вариант 1: CLS token
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Нормализация для cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Конвертировать в numpy
        embeddings_np = embeddings.cpu().numpy()

        return [embeddings_np[i] for i in range(len(texts))]

    def classify_text(self, text: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Классификация текста: является ли он о теории старения

        Args:
            text: Полный текст статьи
            threshold: Порог similarity для классификации

        Returns:
            Dict с результатами классификации
        """
        self._initialize()  # Ленивая загрузка

        if not text or len(text.strip()) < 100:
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "matched_theories": [],
                "reasoning": "Text too short"
            }

        try:
            # Получить embedding текста
            text_embedding = self._get_embeddings_batch([text])[0]

            # Вычислить similarity со всеми теориями
            similarities = np.dot(
                self.theory_embeddings["vectors"],
                text_embedding
            )

            # Найти top-5 наиболее похожих теорий
            top_indices = np.argsort(similarities)[::-1][:5]
            top_theories = [
                {
                    "theory": self.theory_embeddings["names"][idx],
                    "similarity": float(similarities[idx])
                }
                for idx in top_indices
            ]

            # Классификация: если хотя бы 1 теория с similarity > threshold
            max_similarity = float(similarities.max())
            is_aging_theory = max_similarity >= threshold

            # Confidence = максимальная similarity
            confidence = max_similarity

            result = {
                "is_aging_theory": is_aging_theory,
                "confidence": round(confidence, 3),
                "matched_theories": top_theories,
                "method": "bioformer-embeddings",
                "threshold": threshold
            }

            logger.debug(f"Bioformer classification: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in Bioformer classification: {e}")
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "matched_theories": [],
                "error": str(e)
            }

    def classify_batch(
        self,
        texts: List[str],
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Батчевая классификация текстов (GPU оптимизировано)

        Args:
            texts: Список текстов
            threshold: Порог similarity

        Returns:
            Список результатов классификации
        """
        self._initialize()

        if not texts:
            return []

        try:
            # Получить embeddings для всех текстов батчами
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._get_embeddings_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Вычислить similarities для всех текстов сразу
            text_embeddings_matrix = np.array(all_embeddings)  # [N, embedding_dim]
            theory_embeddings_matrix = self.theory_embeddings["vectors"]  # [M, embedding_dim]

            # Matrix multiplication: [N, M]
            similarities_matrix = np.dot(text_embeddings_matrix, theory_embeddings_matrix.T)

            # Обработать результаты для каждого текста
            results = []
            for i, text in enumerate(texts):
                if len(text.strip()) < 100:
                    results.append({
                        "is_aging_theory": False,
                        "confidence": 0.0,
                        "matched_theories": [],
                        "reasoning": "Text too short"
                    })
                    continue

                similarities = similarities_matrix[i]

                # Top-5 теорий
                top_indices = np.argsort(similarities)[::-1][:5]
                top_theories = [
                    {
                        "theory": self.theory_embeddings["names"][idx],
                        "similarity": float(similarities[idx])
                    }
                    for idx in top_indices
                ]

                max_similarity = float(similarities.max())
                is_aging_theory = max_similarity >= threshold

                results.append({
                    "is_aging_theory": is_aging_theory,
                    "confidence": round(max_similarity, 3),
                    "matched_theories": top_theories,
                    "method": "bioformer-embeddings-batch",
                    "threshold": threshold
                })

            logger.info(f"Batch classified {len(texts)} texts")
            return results

        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            # Fallback: пустые результаты
            return [
                {
                    "is_aging_theory": False,
                    "confidence": 0.0,
                    "matched_theories": [],
                    "error": str(e)
                }
                for _ in texts
            ]

    def extract_entities_with_sliding_window(
        self,
        text: str,
        window_size: int = 512,
        stride: int = 256
    ) -> List[Dict[str, Any]]:
        """
        NER с использованием sliding window для длинных текстов

        Args:
            text: Полный текст
            window_size: Размер окна в токенах
            stride: Шаг окна

        Returns:
            Список найденных упоминаний теорий с позициями
        """
        self._initialize()

        # TODO: Реализовать token-level NER используя Bioformer
        # Пока используем упрощенный подход через keyword matching + scoring

        from theory_database import theory_db

        found_entities = []

        # Разбить текст на chunks с overlap
        chunks = self._create_sliding_windows(text, window_size, stride)

        for chunk_idx, (chunk_text, chunk_start) in enumerate(chunks):
            # Получить embedding chunk'а
            chunk_embedding = self._get_embeddings_batch([chunk_text])[0]

            # Проверить similarity с каждой теорией
            similarities = np.dot(
                self.theory_embeddings["vectors"],
                chunk_embedding
            )

            # Если chunk релевантен хотя бы одной теории
            if similarities.max() > 0.5:
                # Найти упоминания теорий в этом chunk (через keyword)
                for theory_name, patterns in theory_db.get_all_theories().items():
                    theory_idx = self.theory_embeddings["names"].index(theory_name)
                    theory_similarity = similarities[theory_idx]

                    if theory_similarity > 0.5:
                        # Поиск паттернов в chunk
                        for pattern in patterns:
                            pos = chunk_text.lower().find(pattern.lower())
                            if pos != -1:
                                found_entities.append({
                                    "theory_name": theory_name,
                                    "matched_text": chunk_text[pos:pos + len(pattern)],
                                    "start_position": chunk_start + pos,
                                    "end_position": chunk_start + pos + len(pattern),
                                    "confidence": float(theory_similarity),
                                    "method": "bioformer-scored-keyword"
                                })

        # Удалить дубликаты (одна и та же позиция)
        found_entities = self._deduplicate_entities(found_entities)

        logger.info(f"Found {len(found_entities)} entity mentions with Bioformer scoring")
        return found_entities

    def _create_sliding_windows(
        self,
        text: str,
        window_size: int,
        stride: int
    ) -> List[Tuple[str, int]]:
        """
        Создать sliding windows для длинного текста

        Args:
            text: Полный текст
            window_size: Размер окна в символах (примерно)
            stride: Шаг окна

        Returns:
            List of (window_text, start_position)
        """
        # Конвертировать размеры из токенов в символы (грубая оценка: 1 токен ≈ 4 символа)
        char_window = window_size * 4
        char_stride = stride * 4

        windows = []
        start = 0

        while start < len(text):
            end = min(start + char_window, len(text))
            window_text = text[start:end]
            windows.append((window_text, start))

            start += char_stride

            # Если последнее окно, выйти
            if end >= len(text):
                break

        return windows

    def _deduplicate_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Удалить дублирующиеся entities (по позиции)

        Args:
            entities: Список entities

        Returns:
            Deduplicated список
        """
        seen_positions = set()
        unique_entities = []

        for entity in entities:
            position_key = (
                entity["theory_name"],
                entity["start_position"],
                entity["end_position"]
            )

            if position_key not in seen_positions:
                seen_positions.add(position_key)
                unique_entities.append(entity)

        return unique_entities

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получить информацию о модели

        Returns:
            Dict с информацией
        """
        self._initialize()

        import torch

        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "quantized": self._quantize,
            "initialized": self._initialized
        }

        if self._initialized and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

        return info


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)

    print("=== Bioformer Classifier Test ===\n")

    # Создать классификатор (ленивая загрузка)
    classifier = BioformerClassifier(
        use_gpu=True,
        quantize=False,
        batch_size=8
    )

    print(f"Classifier created (not yet loaded)\n")

    # Тестовый текст
    test_text = """
    Aging is associated with mitochondrial dysfunction and telomere shortening.
    The mitochondrial theory of aging suggests that accumulated damage to
    mitochondrial DNA leads to cellular senescence. Telomere attrition is
    another hallmark of aging that triggers replicative senescence.
    """

    print("Classifying test text...")
    result = classifier.classify_text(test_text, threshold=0.6)

    print(f"\nClassification result:")
    print(f"  Is aging theory: {result['is_aging_theory']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Top matched theories:")
    for theory in result['matched_theories'][:3]:
        print(f"    - {theory['theory']}: {theory['similarity']:.3f}")

    print(f"\nModel info:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
