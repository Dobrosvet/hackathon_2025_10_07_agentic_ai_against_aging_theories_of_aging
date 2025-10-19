"""
Qdrant Storage для Database Viewer and Export Service
Извлечение данных из Qdrant для формирования таблиц
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Storage handler для извлечения данных из Qdrant для таблиц"""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """
        Инициализация Qdrant storage

        Args:
            qdrant_url: URL Qdrant сервера
        """
        self.qdrant_url = qdrant_url
        self.collection_name = "pubmed_papers"
        self.client = None

    async def initialize(self):
        """Инициализация Qdrant client"""
        try:
            self.client = QdrantClient(url=self.qdrant_url)

            # Проверить существование коллекции
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.error(f"Collection {self.collection_name} does not exist!")
                raise ValueError(f"Collection {self.collection_name} not found")
            else:
                logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    def _extract_year(self, pubdate: str) -> str:
        """
        Извлечь год из pubdate

        Args:
            pubdate: Дата публикации в различных форматах

        Returns:
            Год как строка или 'Unknown'
        """
        if not pubdate:
            return "Unknown"

        # Попробовать извлечь первые 4 цифры
        import re
        match = re.search(r'\d{4}', str(pubdate))
        if match:
            return match.group(0)

        return "Unknown"

    def _generate_theory_id(self, theory_name: str) -> str:
        """
        Генерировать детерминированный ID для теории

        Args:
            theory_name: Название теории

        Returns:
            theory_id как строка
        """
        # Используем hash для детерминированного ID
        hash_obj = hashlib.md5(theory_name.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        return f"T{hash_int % 10000:04d}"

    def get_all_papers_with_theories(self) -> List[Dict[str, Any]]:
        """
        Получить все статьи с теориями старения

        Returns:
            Список статей с полными данными
        """
        try:
            if not self.client:
                return []

            papers = []
            offset = None
            batch_size = 100

            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, offset = result

                if not points:
                    break

                for point in points:
                    # Только статьи с is_aging_theory=True
                    if point.payload.get("is_aging_theory") is True:
                        papers.append({
                            "id": point.id,
                            "pmc_id": point.payload.get("pmc_id", ""),
                            "title": point.payload.get("title", ""),
                            "url": point.payload.get("url", ""),
                            "pubdate": point.payload.get("pubdate", ""),
                            "aging_theories": point.payload.get("aging_theories", []),
                            "questions_classification": point.payload.get("questions_classification"),
                            "criteria_classification": point.payload.get("criteria_classification")
                        })

                if offset is None:
                    break

            logger.info(f"Retrieved {len(papers)} papers with aging theories")
            return papers

        except Exception as e:
            logger.error(f"Error retrieving papers: {e}")
            return []

    def generate_table1_theories(self) -> List[Dict[str, Any]]:
        """
        Генерировать Таблицу 1: Теории старения

        Returns:
            Список словарей с колонками: theory_id, theory_name, number_of_collected_papers
        """
        papers = self.get_all_papers_with_theories()

        # Подсчитать количество статей для каждой теории
        theory_counts = defaultdict(int)

        for paper in papers:
            theories = paper.get("aging_theories", [])
            for theory in theories:
                if theory:  # Пропустить пустые элементы
                    # aging_theories это список словарей с полем theory_name
                    theory_name = theory.get("theory_name") if isinstance(theory, dict) else str(theory)
                    if theory_name:
                        theory_counts[theory_name] += 1

        # Создать таблицу
        table_data = []
        for theory_name, count in sorted(theory_counts.items(), key=lambda x: x[1], reverse=True):
            table_data.append({
                "theory_id": self._generate_theory_id(theory_name),
                "theory_name": theory_name,
                "number_of_collected_papers": count
            })

        logger.info(f"Generated Table 1 with {len(table_data)} theories")
        return table_data

    def generate_table2_papers(self) -> List[Dict[str, Any]]:
        """
        Генерировать Таблицу 2: Собранные статьи

        Returns:
            Список словарей с колонками: theory_id, paper_url, paper_name, paper_year
        """
        papers = self.get_all_papers_with_theories()

        table_data = []

        for paper in papers:
            theories = paper.get("aging_theories", [])
            url = paper.get("url", "")
            title = paper.get("title", "")
            pubdate = paper.get("pubdate", "")
            year = self._extract_year(pubdate)

            # Для каждой статьи берем первую теорию (или создаем запись для каждой)
            # Согласно заданию: одна строка = одна статья
            if theories:
                primary_theory = theories[0]
                # aging_theories это список словарей с полем theory_name
                theory_name = primary_theory.get("theory_name") if isinstance(primary_theory, dict) else str(primary_theory)

                if theory_name:
                    theory_id = self._generate_theory_id(theory_name)

                    table_data.append({
                        "theory_id": theory_id,
                        "paper_url": url,
                        "paper_name": title,
                        "paper_year": year
                    })

        logger.info(f"Generated Table 2 with {len(table_data)} papers")
        return table_data

    def generate_table3_analysis(self) -> List[Dict[str, Any]]:
        """
        Генерировать Таблицу 3: Анализ статей с вопросами и критериями

        Returns:
            Список словарей с колонками: theory_id, paper_url, paper_name, paper_year,
            Q1-Q9, C1-C4
        """
        papers = self.get_all_papers_with_theories()

        table_data = []

        for paper in papers:
            theories = paper.get("aging_theories", [])
            url = paper.get("url", "")
            title = paper.get("title", "")
            pubdate = paper.get("pubdate", "")
            year = self._extract_year(pubdate)

            questions_classification = paper.get("questions_classification")
            criteria_classification = paper.get("criteria_classification")

            # Только статьи с классификацией
            if not questions_classification and not criteria_classification:
                continue

            if theories:
                primary_theory = theories[0]
                # aging_theories это список словарей с полем theory_name
                theory_name = primary_theory.get("theory_name") if isinstance(primary_theory, dict) else str(primary_theory)

                if not theory_name:
                    continue

                theory_id = self._generate_theory_id(theory_name)

                row = {
                    "theory_id": theory_id,
                    "paper_url": url,
                    "paper_name": title,
                    "paper_year": year
                }

                # Добавить Q1-Q9
                for i in range(1, 10):
                    q_key = f"Q{i}"
                    if questions_classification and q_key in questions_classification:
                        answer = questions_classification[q_key]
                        # Преобразовать в строковый формат
                        if isinstance(answer, dict):
                            row[q_key] = answer.get("answer", "N/A")
                        else:
                            row[q_key] = str(answer) if answer is not None else "N/A"
                    else:
                        row[q_key] = "N/A"

                # Добавить C1-C4
                for i in range(1, 5):
                    c_key = f"C{i}"
                    if criteria_classification and c_key in criteria_classification:
                        answer = criteria_classification[c_key]
                        # Преобразовать в строковый формат
                        if isinstance(answer, dict):
                            row[c_key] = answer.get("answer", "N/A")
                        else:
                            row[c_key] = str(answer) if answer is not None else "N/A"
                    else:
                        row[c_key] = "N/A"

                table_data.append(row)

        logger.info(f"Generated Table 3 with {len(table_data)} analyzed papers")
        return table_data

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику по данным

        Returns:
            Словарь со статистикой
        """
        papers = self.get_all_papers_with_theories()

        total_papers = len(papers)

        # Подсчитать уникальные теории
        all_theories = set()
        for paper in papers:
            theories = paper.get("aging_theories", [])
            for theory in theories:
                if theory:
                    # aging_theories это список словарей с полем theory_name
                    theory_name = theory.get("theory_name") if isinstance(theory, dict) else str(theory)
                    if theory_name:
                        all_theories.add(theory_name)

        total_theories = len(all_theories)

        # Подсчитать статьи с классификацией
        papers_with_questions = 0
        papers_with_criteria = 0

        for paper in papers:
            if paper.get("questions_classification"):
                papers_with_questions += 1
            if paper.get("criteria_classification"):
                papers_with_criteria += 1

        return {
            "total_papers": total_papers,
            "total_theories": total_theories,
            "papers_with_questions_classification": papers_with_questions,
            "papers_with_criteria_classification": papers_with_criteria,
            "papers_fully_classified": min(papers_with_questions, papers_with_criteria),
            "classification_progress_percent": round(
                (min(papers_with_questions, papers_with_criteria) / total_papers * 100) if total_papers > 0 else 0,
                2
            )
        }
