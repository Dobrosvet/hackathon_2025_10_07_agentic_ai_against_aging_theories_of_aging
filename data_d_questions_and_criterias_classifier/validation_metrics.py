"""
Validation Metrics Module
Расчет метрик точности модели на валидационных данных
"""

import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ValidationMetrics:
    """Класс для расчета метрик валидации модели"""

    def __init__(self):
        """Инициализация"""
        self.question_ids = [f"Q{i}" for i in range(1, 10)]  # Q1-Q9

    def normalize_answer(self, answer: Any) -> str:
        """
        Нормализовать ответ к единому формату

        Args:
            answer: Ответ (может быть Yes/No/True/False/1/0)

        Returns:
            Нормализованный ответ ("Yes" или "No")
        """
        if answer is None:
            return "No"

        answer_str = str(answer).strip().lower()

        # Варианты "Yes"
        if answer_str in ["yes", "true", "1", "1.0"]:
            return "Yes"

        # Варианты "No"
        if answer_str in ["no", "false", "0", "0.0"]:
            return "No"

        # По умолчанию
        logger.warning(f"Unknown answer format: {answer}, defaulting to 'No'")
        return "No"

    def calculate_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Dict[str, int]:
        """
        Рассчитать confusion matrix для бинарной классификации

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки

        Returns:
            Словарь с TP, TN, FP, FN
        """
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "Yes" and p == "Yes")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "No" and p == "No")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "No" and p == "Yes")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "Yes" and p == "No")

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }

    def calculate_metrics_from_cm(
        self,
        cm: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Рассчитать метрики из confusion matrix

        Args:
            cm: Confusion matrix (tp, tn, fp, fn)

        Returns:
            Словарь с метриками
        """
        tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
        total = tp + tn + fp + fn

        if total == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }

        # Accuracy
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # Precision (для класса "Yes")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall (для класса "Yes")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1-score
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }

    def calculate_per_question_metrics(
        self,
        validation_papers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Рассчитать метрики для каждого вопроса

        Args:
            validation_papers: Список валидационных статей с разметкой и предсказаниями

        Returns:
            Словарь с метриками для каждого вопроса
        """
        metrics_per_question = {}

        for question_id in self.question_ids:
            y_true = []
            y_pred = []

            for paper in validation_papers:
                # Получить истинное значение (ручная разметка)
                manual_answer = paper.get("validation_questions", {}).get(question_id)
                if manual_answer is None:
                    logger.warning(
                        f"Paper {paper.get('paper_url')} has no manual annotation for {question_id}"
                    )
                    continue

                # Получить предсказание модели
                model_answer = None
                questions_classification = paper.get("questions_classification", {})

                if isinstance(questions_classification, dict):
                    question_result = questions_classification.get(question_id)
                    if question_result and isinstance(question_result, dict):
                        model_answer = question_result.get("answer")

                if model_answer is None:
                    logger.warning(
                        f"Paper {paper.get('paper_url')} has no model prediction for {question_id}"
                    )
                    continue

                # Нормализовать ответы
                y_true.append(self.normalize_answer(manual_answer))
                y_pred.append(self.normalize_answer(model_answer))

            # Если есть данные для расчета
            if y_true and y_pred:
                cm = self.calculate_confusion_matrix(y_true, y_pred)
                metrics = self.calculate_metrics_from_cm(cm)

                metrics_per_question[question_id] = {
                    **metrics,
                    "confusion_matrix": cm,
                    "support": len(y_true),
                    "y_true": y_true,
                    "y_pred": y_pred
                }
            else:
                logger.warning(f"No data to calculate metrics for {question_id}")
                metrics_per_question[question_id] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                    "support": 0,
                    "y_true": [],
                    "y_pred": []
                }

        return metrics_per_question

    def calculate_overall_metrics(
        self,
        per_question_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Рассчитать общие метрики по всем вопросам

        Args:
            per_question_metrics: Метрики для каждого вопроса

        Returns:
            Общие метрики
        """
        # Macro-averaged F1 (среднее по всем вопросам)
        f1_scores = [m["f1_score"] for m in per_question_metrics.values() if m["support"] > 0]
        f1_macro = np.mean(f1_scores) if f1_scores else 0.0

        # Micro-averaged метрики (суммируем TP, TN, FP, FN по всем вопросам)
        total_cm = {
            "tp": sum(m["confusion_matrix"]["tp"] for m in per_question_metrics.values()),
            "tn": sum(m["confusion_matrix"]["tn"] for m in per_question_metrics.values()),
            "fp": sum(m["confusion_matrix"]["fp"] for m in per_question_metrics.values()),
            "fn": sum(m["confusion_matrix"]["fn"] for m in per_question_metrics.values())
        }

        micro_metrics = self.calculate_metrics_from_cm(total_cm)

        # Weighted-averaged метрики (взвешенное среднее по support)
        total_support = sum(m["support"] for m in per_question_metrics.values())

        if total_support > 0:
            f1_weighted = sum(
                m["f1_score"] * m["support"]
                for m in per_question_metrics.values()
            ) / total_support

            accuracy_weighted = sum(
                m["accuracy"] * m["support"]
                for m in per_question_metrics.values()
            ) / total_support
        else:
            f1_weighted = 0.0
            accuracy_weighted = 0.0

        return {
            "accuracy_micro": micro_metrics["accuracy"],
            "accuracy_macro": np.mean([m["accuracy"] for m in per_question_metrics.values()]),
            "accuracy_weighted": round(accuracy_weighted, 4),
            "precision_micro": micro_metrics["precision"],
            "recall_micro": micro_metrics["recall"],
            "f1_micro": micro_metrics["f1_score"],
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "total_support": total_support
        }

    def calculate_validation_metrics(
        self,
        validation_papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Рассчитать все валидационные метрики

        Args:
            validation_papers: Список валидационных статей

        Returns:
            Полный набор метрик
        """
        # Метрики по вопросам
        per_question_metrics = self.calculate_per_question_metrics(validation_papers)

        # Общие метрики
        overall_metrics = self.calculate_overall_metrics(per_question_metrics)

        # Детальное сравнение для каждой статьи
        paper_comparisons = []
        for paper in validation_papers:
            comparison = {
                "paper_url": paper.get("paper_url"),
                "paper_name": paper.get("validation_paper_name") or paper.get("title"),
                "paper_year": paper.get("validation_paper_year") or paper.get("year"),
                "questions": {}
            }

            for question_id in self.question_ids:
                manual = paper.get("validation_questions", {}).get(question_id)
                model = None

                questions_classification = paper.get("questions_classification", {})
                if isinstance(questions_classification, dict):
                    question_result = questions_classification.get(question_id)
                    if question_result and isinstance(question_result, dict):
                        model = question_result.get("answer")

                manual_normalized = self.normalize_answer(manual) if manual is not None else None
                model_normalized = self.normalize_answer(model) if model is not None else None

                comparison["questions"][question_id] = {
                    "manual": manual_normalized,
                    "predicted": model_normalized,
                    "match": manual_normalized == model_normalized if (manual_normalized and model_normalized) else None
                }

            paper_comparisons.append(comparison)

        return {
            "overall": overall_metrics,
            "per_question": per_question_metrics,
            "paper_comparisons": paper_comparisons,
            "num_papers": len(validation_papers)
        }

    def format_metrics_for_display(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Форматировать метрики для отображения в UI

        Args:
            metrics: Словарь с метриками

        Returns:
            Форматированные метрики
        """
        # Убрать y_true и y_pred из метрик (слишком большие для JSON)
        formatted = {
            "overall": metrics["overall"],
            "per_question": {},
            "paper_comparisons": metrics["paper_comparisons"],
            "num_papers": metrics["num_papers"]
        }

        for question_id, question_metrics in metrics["per_question"].items():
            formatted["per_question"][question_id] = {
                "accuracy": question_metrics["accuracy"],
                "precision": question_metrics["precision"],
                "recall": question_metrics["recall"],
                "f1_score": question_metrics["f1_score"],
                "support": question_metrics["support"],
                "confusion_matrix": question_metrics["confusion_matrix"]
            }

        return formatted
