"""
Qdrant Storage для Questions and Criterias Classifier
Работа с базой данных для загрузки и сохранения результатов классификации
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Storage handler для Qdrant vector database - Questions Classifier Service"""

    def __init__(self, db_path: str = None, qdrant_url: str = "http://localhost:6333"):
        """
        Инициализация Qdrant storage

        Args:
            db_path: Путь к БД (deprecated)
            qdrant_url: URL Qdrant сервера
        """
        self.db_path = Path(db_path) if db_path else None
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
                raise ValueError(f"Collection {self.collection_name} not found. Run services A, B, C first.")
            else:
                logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    def get_papers_with_aging_theories(self) -> List[Dict[str, Any]]:
        """
        Получить статьи, которые прошли фильтр теорий старения (is_aging_theory=True)

        Returns:
            Список статей для классификации по вопросам
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
                    # Проверяем: есть full_text И is_aging_theory=True
                    if (
                        "full_text" in point.payload
                        and point.payload.get("full_text")
                        and point.payload.get("is_aging_theory") is True
                    ):
                        papers.append({
                            "id": point.id,
                            "pmc_id": point.payload.get("pmc_id", ""),
                            "title": point.payload.get("title", ""),
                            "full_text": point.payload.get("full_text", ""),
                            "has_questions_classification": "questions_classification" in point.payload
                        })

                if offset is None:
                    break

            logger.info(f"Retrieved {len(papers)} papers with aging theories")
            return papers

        except Exception as e:
            logger.error(f"Error getting papers with aging theories: {e}")
            return []

    def get_papers_without_questions_classification(self) -> List[Dict[str, Any]]:
        """
        Получить статьи, которые ещё не классифицированы по вопросам

        Returns:
            Список статей без классификации по вопросам
        """
        try:
            all_papers = self.get_papers_with_aging_theories()
            papers_without_classification = [
                p for p in all_papers if not p["has_questions_classification"]
            ]

            logger.info(
                f"Found {len(papers_without_classification)} papers without questions classification"
            )
            return papers_without_classification

        except Exception as e:
            logger.error(f"Error getting papers without questions classification: {e}")
            return []

    def get_classified_papers_count(self) -> int:
        """
        Подсчитать количество статей с классификацией по вопросам

        Returns:
            Количество классифицированных статей
        """
        try:
            all_papers = self.get_papers_with_aging_theories()
            count = sum(1 for p in all_papers if p["has_questions_classification"])

            logger.info(f"Found {count} papers with questions classification")
            return count

        except Exception as e:
            logger.error(f"Error counting classified papers: {e}")
            return 0

    async def update_paper_with_questions_classification(
        self,
        point_id: int,
        pmc_id: str,
        classification_result: Dict[str, Any]
    ) -> bool:
        """
        Обновить статью с результатами классификации по вопросам

        Args:
            point_id: Qdrant point ID
            pmc_id: PubMed Central ID
            classification_result: Результаты классификации

        Returns:
            True если успешно
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            # Получить существующую точку
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not point:
                logger.error(f"Point {point_id} not found")
                return False

            # Обновить payload
            existing_point = point[0]
            updated_payload = existing_point.payload.copy()

            # Добавить результаты классификации
            updated_payload["questions_classification"] = classification_result.get("questions", {})
            updated_payload["criteria_classification"] = classification_result.get("criteria", {})
            updated_payload["questions_timestamp"] = classification_result.get("timestamp")

            # Обновить точку в Qdrant
            updated_point = PointStruct(
                id=point_id,
                vector=existing_point.vector,
                payload=updated_payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logger.info(f"Updated PMC{pmc_id} with questions classification")
            return True

        except Exception as e:
            logger.error(f"Error updating paper {pmc_id}: {e}")
            return False

    async def update_papers_batch(
        self,
        updates: List[Dict[str, Any]]
    ) -> int:
        """
        Батчевое обновление статей

        Args:
            updates: Список обновлений [{point_id, pmc_id, classification_result}, ...]

        Returns:
            Количество обновлённых статей
        """
        updated_count = 0

        for update in updates:
            success = await self.update_paper_with_questions_classification(
                point_id=update["point_id"],
                pmc_id=update["pmc_id"],
                classification_result=update["classification_result"]
            )

            if success:
                updated_count += 1

        return updated_count

    def get_paper_by_pmc_id(self, pmc_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить статью по PMC ID

        Args:
            pmc_id: PubMed Central ID

        Returns:
            Данные статьи или None
        """
        try:
            if not self.client:
                return None

            # Поиск по pmc_id
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
                    if point.payload.get("pmc_id") == pmc_id:
                        return {
                            "id": point.id,
                            "pmc_id": point.payload.get("pmc_id"),
                            "title": point.payload.get("title"),
                            "full_text": point.payload.get("full_text"),
                            "is_aging_theory": point.payload.get("is_aging_theory"),
                            "questions_classification": point.payload.get("questions_classification"),
                            "criteria_classification": point.payload.get("criteria_classification"),
                            "manual_annotations": point.payload.get("manual_annotations", [])
                        }

                if offset is None:
                    break

            return None

        except Exception as e:
            logger.error(f"Error getting paper {pmc_id}: {e}")
            return None

    async def save_manual_annotation(
        self,
        pmc_id: str,
        annotation: Dict[str, Any]
    ) -> bool:
        """
        Сохранить ручную аннотацию фрагмента текста

        Args:
            pmc_id: PubMed Central ID
            annotation: Аннотация {type: Q1/Q2/.../C1/..., answer: ..., fragment: {...}}

        Returns:
            True если успешно
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            # Найти статью
            paper = self.get_paper_by_pmc_id(pmc_id)

            if not paper:
                logger.error(f"Paper PMC{pmc_id} not found")
                return False

            point_id = paper["id"]

            # Получить точку
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )[0]

            # Обновить payload
            updated_payload = point.payload.copy()

            # Добавить аннотацию
            if "manual_annotations" not in updated_payload:
                updated_payload["manual_annotations"] = []

            # Добавить timestamp
            annotation["timestamp"] = datetime.now().isoformat()

            updated_payload["manual_annotations"].append(annotation)

            # Обновить точку
            updated_point = PointStruct(
                id=point_id,
                vector=point.vector,
                payload=updated_payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logger.info(f"Saved manual annotation for PMC{pmc_id}: {annotation['type']}")
            return True

        except Exception as e:
            logger.error(f"Error saving annotation for {pmc_id}: {e}")
            return False

    async def delete_manual_annotation(
        self,
        pmc_id: str,
        annotation_index: int
    ) -> bool:
        """
        Удалить ручную аннотацию

        Args:
            pmc_id: PubMed Central ID
            annotation_index: Индекс аннотации в списке

        Returns:
            True если успешно
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            # Найти статью
            paper = self.get_paper_by_pmc_id(pmc_id)

            if not paper:
                logger.error(f"Paper PMC{pmc_id} not found")
                return False

            point_id = paper["id"]

            # Получить точку
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )[0]

            # Обновить payload
            updated_payload = point.payload.copy()

            if "manual_annotations" in updated_payload:
                annotations = updated_payload["manual_annotations"]

                if 0 <= annotation_index < len(annotations):
                    del annotations[annotation_index]
                    updated_payload["manual_annotations"] = annotations

                    # Обновить точку
                    updated_point = PointStruct(
                        id=point_id,
                        vector=point.vector,
                        payload=updated_payload
                    )

                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[updated_point]
                    )

                    logger.info(f"Deleted annotation {annotation_index} for PMC{pmc_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error deleting annotation for {pmc_id}: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получить информацию о коллекции

        Returns:
            Информация о коллекции
        """
        try:
            if not self.client:
                return {"count": 0, "error": "Client not initialized"}

            info = self.client.get_collection(self.collection_name)

            return {
                "count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": "ok"
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"count": 0, "error": str(e)}
