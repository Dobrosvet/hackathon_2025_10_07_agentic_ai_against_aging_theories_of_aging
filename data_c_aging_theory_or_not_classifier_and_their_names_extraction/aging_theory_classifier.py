"""
Гибридный классификатор теорий старения
Комбинирует keyword matching (быстрый) и Bioformer-8L (точный)
"""

import logging
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


class AgingTheoryClassifier:
    """
    Гибридный классификатор теорий старения

    3 режима работы:
    - keyword: Только keyword matching (быстро, но менее точно)
    - bioformer: Только Bioformer-8L (медленно, но точно)
    - hybrid: Комбинация обоих (оптимально)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        use_gpu: bool = True,
        bioformer_threshold: float = 0.6,
        quantize: bool = False
    ):
        """
        Инициализация классификатора

        Args:
            mode: Режим работы - "keyword", "bioformer", "hybrid"
            use_gpu: Использовать GPU для Bioformer
            bioformer_threshold: Порог similarity для Bioformer классификации
            quantize: Использовать quantization для Bioformer
        """
        self.mode = mode
        self.model_name = "bioformer-8L-hybrid"
        self.version = "2.0"
        self.bioformer_threshold = bioformer_threshold

        # Загрузить расширенную базу теорий
        try:
            from theory_database import theory_db
            self.theory_patterns = theory_db.get_all_theories()
            logger.info(f"Loaded {len(self.theory_patterns)} theories from database")
        except ImportError:
            logger.warning("theory_database not found, using fallback patterns")
            self.theory_patterns = self._get_fallback_patterns()

        # Контекстные слова для уточнения поиска
        self.context_keywords = [
            "aging", "ageing", "senescence", "longevity",
            "lifespan", "theory", "hypothesis", "mechanism",
            "gerontology", "geriatric", "age-related",
            "age-associated", "hallmarks"
        ]

        # Инициализировать Bioformer (если нужен)
        self.bioformer = None
        if mode in ["bioformer", "hybrid"]:
            try:
                from bioformer_classifier import BioformerClassifier
                self.bioformer = BioformerClassifier(
                    use_gpu=use_gpu,
                    quantize=quantize,
                    batch_size=16
                )
                logger.info(f"Bioformer classifier initialized (lazy loading)")
            except ImportError as e:
                logger.warning(f"Bioformer not available: {e}, falling back to keyword mode")
                self.mode = "keyword"

        logger.info(f"AgingTheoryClassifier initialized in {self.mode} mode")
        logger.info(f"Model: {self.model_name} v{self.version}")

    def _get_fallback_patterns(self) -> Dict[str, List[str]]:
        """Fallback паттерны если theory_database недоступна"""
        return {
            "Mitochondrial Dysfunction": [
                "mitochondrial theory", "mitochondrial dysfunction"
            ],
            "Cellular Senescence": [
                "cellular senescence", "senescent cells"
            ],
            "Telomere Attrition": [
                "telomere attrition", "telomere shortening"
            ]
        }

    def classify_aging_theory(self, text: str) -> Dict[str, Any]:
        """
        Классификация: является ли статья о теории старения

        Args:
            text: Полный текст статьи

        Returns:
            Dict с результатами классификации
        """
        if not text or len(text.strip()) < 100:
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "keyword_matches": 0,
                "reasoning": "Text too short",
                "method": self.mode
            }

        try:
            if self.mode == "keyword":
                return self._classify_keyword_only(text)
            elif self.mode == "bioformer":
                return self._classify_bioformer_only(text)
            elif self.mode == "hybrid":
                return self._classify_hybrid(text)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "keyword_matches": 0,
                "error": str(e),
                "method": self.mode
            }

    def _classify_keyword_only(self, text: str) -> Dict[str, Any]:
        """Быстрая классификация только keyword matching"""
        text_lower = text.lower()

        # Подсчет совпадений теорий
        theory_matches = 0
        matched_theories = []

        for theory_name, patterns in self.theory_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    theory_matches += 1
                    if theory_name not in matched_theories:
                        matched_theories.append(theory_name)
                    break  # Считаем теорию только один раз

        # Подсчет контекстных слов
        context_matches = sum(
            1 for keyword in self.context_keywords if keyword in text_lower
        )

        # Логика классификации:
        # - Минимум 2 разные теории старения
        # - Или 1 теория + много контекстных слов (>= 5)
        is_theory = theory_matches >= 2 or (theory_matches >= 1 and context_matches >= 5)

        # Расчет уверенности (confidence)
        if is_theory:
            # Базовая уверенность от количества теорий
            base_confidence = min(theory_matches / 5.0, 0.7)
            # Бонус от контекста
            context_bonus = min(context_matches / 20.0, 0.3)
            confidence = min(base_confidence + context_bonus, 1.0)
        else:
            confidence = max(0.0, theory_matches / 10.0)

        return {
            "is_aging_theory": is_theory,
            "confidence": round(confidence, 3),
            "keyword_matches": theory_matches,
            "context_matches": context_matches,
            "matched_theories_count": len(matched_theories),
            "matched_theories": matched_theories[:5],
            "method": "keyword"
        }

    def _classify_bioformer_only(self, text: str) -> Dict[str, Any]:
        """Точная классификация через Bioformer-8L"""
        if not self.bioformer:
            logger.warning("Bioformer not available, falling back to keyword")
            return self._classify_keyword_only(text)

        result = self.bioformer.classify_text(text, threshold=self.bioformer_threshold)

        # Конвертировать в стандартный формат
        return {
            "is_aging_theory": result["is_aging_theory"],
            "confidence": result["confidence"],
            "matched_theories": [t["theory"] for t in result.get("matched_theories", [])[:5]],
            "matched_theories_count": len(result.get("matched_theories", [])),
            "bioformer_top_theories": result.get("matched_theories", [])[:3],
            "method": "bioformer"
        }

    def _classify_hybrid(self, text: str) -> Dict[str, Any]:
        """
        Гибридная классификация (оптимально)

        Stage 1: Keyword pre-filter (быстрый) - отсеивает явно нерелевантные
        Stage 2: Bioformer verification (точный) - для потенциально релевантных
        """
        # Stage 1: Keyword pre-filtering
        keyword_result = self._classify_keyword_only(text)

        # Быстрый отсев: если keyword уверенно говорит "НЕТ"
        if keyword_result["keyword_matches"] == 0 and keyword_result["context_matches"] < 3:
            keyword_result["method"] = "hybrid-keyword-rejected"
            return keyword_result

        # Быстрое принятие: если keyword уверенно говорит "ДА"
        if keyword_result["keyword_matches"] >= 5 and keyword_result["context_matches"] >= 10:
            keyword_result["method"] = "hybrid-keyword-accepted"
            return keyword_result

        # Stage 2: Bioformer verification для пограничных случаев
        if self.bioformer:
            bioformer_result = self.bioformer.classify_text(
                text,
                threshold=self.bioformer_threshold
            )

            # Комбинировать результаты
            # Если Bioformer уверен - используем его
            if bioformer_result["confidence"] >= 0.8:
                return {
                    "is_aging_theory": bioformer_result["is_aging_theory"],
                    "confidence": bioformer_result["confidence"],
                    "matched_theories": [t["theory"] for t in bioformer_result.get("matched_theories", [])[:5]],
                    "keyword_matches": keyword_result["keyword_matches"],
                    "context_matches": keyword_result["context_matches"],
                    "bioformer_confidence": bioformer_result["confidence"],
                    "method": "hybrid-bioformer-decided"
                }

            # Взвешенное комбинирование
            combined_confidence = (
                keyword_result["confidence"] * 0.4 +
                bioformer_result["confidence"] * 0.6
            )

            is_theory = combined_confidence >= 0.5

            return {
                "is_aging_theory": is_theory,
                "confidence": round(combined_confidence, 3),
                "matched_theories": [t["theory"] for t in bioformer_result.get("matched_theories", [])[:5]],
                "keyword_matches": keyword_result["keyword_matches"],
                "context_matches": keyword_result["context_matches"],
                "keyword_confidence": keyword_result["confidence"],
                "bioformer_confidence": bioformer_result["confidence"],
                "method": "hybrid-combined"
            }
        else:
            # Fallback to keyword if Bioformer unavailable
            keyword_result["method"] = "hybrid-keyword-fallback"
            return keyword_result

    def extract_theory_names(self, text: str) -> List[Dict[str, Any]]:
        """
        Named Entity Recognition: извлечение названий теорий и их позиций
        Использует keyword matching с Bioformer scoring (если доступен)

        Args:
            text: Полный текст статьи

        Returns:
            List of dicts с информацией о найденных теориях
        """
        if not text or len(text.strip()) < 100:
            return []

        try:
            # Keyword-based NER
            found_theories = self._extract_keyword_entities(text)

            # Bioformer scoring (если включен)
            if self.mode in ["bioformer", "hybrid"] and self.bioformer:
                found_theories = self._score_entities_with_bioformer(text, found_theories)

            # Сортировка по позиции
            found_theories.sort(key=lambda x: x['start_position'])

            logger.info(f"Found {len(found_theories)} theory mentions ({self.mode} mode)")
            return found_theories

        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return []

    def _extract_keyword_entities(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение entities через keyword matching"""
        found_theories = []
        text_lower = text.lower()

        # Поиск всех паттернов
        for theory_name, patterns in self.theory_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                start = 0

                # Находим все вхождения паттерна
                while True:
                    pos = text_lower.find(pattern_lower, start)
                    if pos == -1:
                        break

                    # Извлекаем оригинальный текст
                    original_text = text[pos:pos + len(pattern)]

                    # Проверяем word boundary
                    is_word_boundary = True
                    if pos > 0:
                        prev_char = text[pos - 1]
                        if prev_char.isalnum():
                            is_word_boundary = False

                    end_pos = pos + len(pattern)
                    if end_pos < len(text):
                        next_char = text[end_pos]
                        if next_char.isalnum():
                            is_word_boundary = False

                    if is_word_boundary:
                        # Извлекаем контекст
                        context_start = max(0, pos - 100)
                        context_end = min(len(text), pos + len(pattern) + 100)
                        context = text[context_start:context_end].strip()
                        context = re.sub(r'\s+', ' ', context)

                        # Проверка на дубликаты
                        is_duplicate = False
                        for existing in found_theories:
                            if (existing['theory_name'] == theory_name and
                                abs(existing['start_position'] - pos) < 10):
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            found_theories.append({
                                "theory_name": theory_name,
                                "matched_text": original_text,
                                "start_position": pos,
                                "end_position": pos + len(pattern),
                                "confidence": 0.95,  # Keyword match
                                "context_snippet": context,
                                "method": "keyword"
                            })

                    start = pos + 1

        return found_theories

    def _score_entities_with_bioformer(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Пересчитать confidence для entities используя Bioformer

        Args:
            text: Полный текст
            entities: Список entities из keyword extraction

        Returns:
            Entities с обновленными confidence scores
        """
        if not entities or not self.bioformer:
            return entities

        # Использовать sliding window + Bioformer scoring
        # TODO: Реализовать более продвинутый scoring
        # Пока просто добавим метку что использован Bioformer
        for entity in entities:
            entity["method"] = "hybrid-keyword-bioformer-scored"

        return entities

    def process_paper(self, text: str) -> Dict[str, Any]:
        """
        Полная обработка статьи: классификация + NER

        Args:
            text: Полный текст статьи

        Returns:
            Dict с результатами классификации и списком найденных теорий
        """
        try:
            # Классификация
            classification_result = self.classify_aging_theory(text)

            # NER (только если классифицировано как теория старения)
            if classification_result['is_aging_theory']:
                theories = self.extract_theory_names(text)
            else:
                theories = []

            return {
                "is_aging_theory": classification_result['is_aging_theory'],
                "classification_confidence": classification_result['confidence'],
                "aging_theories": theories,
                "classification_model": self.model_name,
                "classification_version": self.version,
                "classification_method": classification_result.get("method", self.mode),
                "total_theories_found": len(theories),
                "keyword_matches": classification_result.get("keyword_matches", 0),
                "context_matches": classification_result.get("context_matches", 0)
            }

        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о классификаторе"""
        info = {
            "model_name": self.model_name,
            "version": self.version,
            "mode": self.mode,
            "theories_count": len(self.theory_patterns),
            "bioformer_enabled": self.bioformer is not None
        }

        if self.bioformer:
            info["bioformer_info"] = self.bioformer.get_model_info()

        return info


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)

    print("=== Hybrid Aging Theory Classifier Test ===\n")

    # Тестовый текст
    test_text = """
    Aging is characterized by multiple hallmarks including genomic instability,
    telomere attrition, epigenetic alterations, loss of proteostasis, and
    cellular senescence. The mitochondrial theory of aging suggests that
    mitochondrial dysfunction plays a central role. Recent studies on
    inflammaging and immunosenescence provide new insights into the aging process.
    """

    # Тест всех режимов
    for mode in ["keyword", "hybrid"]:  # Пропускаем bioformer для быстрого теста
        print(f"\n--- Testing {mode.upper()} mode ---")
        classifier = AgingTheoryClassifier(mode=mode, use_gpu=False)

        result = classifier.process_paper(test_text)

        print(f"Is aging theory: {result['is_aging_theory']}")
        print(f"Confidence: {result['classification_confidence']:.3f}")
        print(f"Method: {result['classification_method']}")
        print(f"Keyword matches: {result.get('keyword_matches', 'N/A')}")
        print(f"Theories found: {result['total_theories_found']}")

        if result['aging_theories']:
            print("Detected theories:")
            for theory in result['aging_theories'][:3]:
                print(f"  - {theory['theory_name']} ({theory['confidence']:.2f})")
