"""
Скрипт для загрузки валидационных данных в Qdrant БД
Объединяет CSV с разметкой и полные тексты статей из MD файлов

Промпт
```
Возьми валидационные данные из файла @/d:/Cloud/Projects/hackathon_aaaa_251016/personal_folder/validate_data_fulltext.csv .

Соотнеси каждую строчку полным текстом научной статьи из соответствующих файлов которые я указал ниже. Пусть идентификатором для слияния будет колонка `paper_url` и эти тексты сохранятся в БД:
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC7612201/` — @/d:/Cloud/Projects/hackathon_aaaa_251016/personal_folder/PMC7612201.md
- `https://pubmed.ncbi.nlm.nih.gov/36096982/` — @/d:/Cloud/Projects/hackathon_aaaa_251016/personal_folder/PM36096982.md
- `https://pubmed.ncbi.nlm.nih.gov/40423632/` — @/d:/Cloud/Projects/hackathon_aaaa_251016/personal_folder/PM40423632.md
- `https://pubmed.ncbi.nlm.nih.gov/38102202/` — @/d:/Cloud/Projects/hackathon_aaaa_251016/personal_folder/PM38102202.md
- `https://pubmed.ncbi.nlm.nih.gov/38636560/` — @/d:/Cloud/Projects/hackathon_aaaa_251016/personal_folder/PM38636560.md 
 
Пометь каждую строчку что это валидационные данные размеченные человеком. А также отметку что это является статьёй про теории старения.

Сделай команду которая сохранит эти данные в Qdrant БД. Если в БД уже есть статьи с такими же `paper_url`, пометь данные те что в БД отметкой что это валидационные данные размеченные человеком и обнови остальные поля (9 вопросов).
```
"""

import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationDataUploader:
    """Загрузчик валидационных данных в Qdrant"""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """
        Инициализация загрузчика

        Args:
            qdrant_url: URL Qdrant сервера
        """
        self.qdrant_url = qdrant_url
        self.collection_name = "pubmed_papers"
        self.client = None

        # Маппинг URL -> файл с полным текстом
        self.url_to_file = {
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7612201/": "PMC7612201.md",
            "https://pubmed.ncbi.nlm.nih.gov/36096982/": "PM36096982.md",
            "https://pubmed.ncbi.nlm.nih.gov/40423632/": "PM40423632.md",
            "https://pubmed.ncbi.nlm.nih.gov/38102202/": "PM38102202.md",
            "https://pubmed.ncbi.nlm.nih.gov/38636560/": "PM38636560.md",
        }

    def initialize(self):
        """Инициализация Qdrant client"""
        try:
            self.client = QdrantClient(url=self.qdrant_url)

            # Проверить существование коллекции
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.error(f"Collection {self.collection_name} does not exist!")
                raise ValueError(f"Collection {self.collection_name} not found.")
            else:
                logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    def read_csv_data(self, csv_path: Path) -> List[Dict[str, Any]]:
        """
        Прочитать CSV с валидационными данными

        Args:
            csv_path: Путь к CSV файлу

        Returns:
            Список валидационных данных
        """
        validation_data = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    validation_data.append({
                        'paper_url': row['paper_url'],
                        'paper_name': row['paper_name'],
                        'paper_year': row['paper_year'],
                        'Q1': row['Q1'],
                        'Q2': row['Q2'],
                        'Q3': row['Q3'],
                        'Q4': row['Q4'],
                        'Q5': row['Q5'],
                        'Q6': row['Q6'],
                        'Q7': row['Q7'],
                        'Q8': row['Q8'],
                        'Q9': row['Q9'],
                    })

            logger.info(f"Read {len(validation_data)} validation records from CSV")
            return validation_data

        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []

    def read_fulltext(self, md_path: Path) -> str:
        """
        Прочитать полный текст из MD файла

        Args:
            md_path: Путь к MD файлу

        Returns:
            Полный текст статьи
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading MD file {md_path}: {e}")
            return ""

    def find_paper_by_url(self, paper_url: str) -> Optional[Dict[str, Any]]:
        """
        Найти статью в Qdrant по URL

        Args:
            paper_url: URL статьи

        Returns:
            Данные статьи или None
        """
        try:
            if not self.client:
                return None

            offset = None
            batch_size = 100

            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )

                points, offset = result

                if not points:
                    break

                for point in points:
                    # Проверяем URL в разных форматах
                    payload_url = point.payload.get("paper_url", "")
                    pmc_id = point.payload.get("pmc_id", "")

                    # Нормализуем URLs для сравнения
                    normalized_search = paper_url.lower().strip().rstrip('/')
                    normalized_payload = payload_url.lower().strip().rstrip('/')

                    # Также проверяем совпадение по PMC ID или PubMed ID
                    if normalized_search == normalized_payload:
                        return {
                            "id": point.id,
                            "payload": point.payload,
                            "vector": point.vector
                        }

                    # Дополнительная проверка по ID в URL
                    if pmc_id and pmc_id in paper_url:
                        return {
                            "id": point.id,
                            "payload": point.payload,
                            "vector": point.vector
                        }

                if offset is None:
                    break

            return None

        except Exception as e:
            logger.error(f"Error finding paper by URL {paper_url}: {e}")
            return None

    def update_or_create_paper(
        self,
        validation_record: Dict[str, Any],
        fulltext: str
    ) -> bool:
        """
        Обновить существующую статью или создать новую с валидационными данными

        Args:
            validation_record: Запись из CSV с валидационными данными
            fulltext: Полный текст статьи

        Returns:
            True если успешно
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            paper_url = validation_record['paper_url']

            # Найти существующую статью
            existing_paper = self.find_paper_by_url(paper_url)

            # Подготовить данные для классификации по вопросам
            questions_classification = {
                f"Q{i}": validation_record[f"Q{i}"]
                for i in range(1, 10)
            }

            # Timestamp для отслеживания
            timestamp = datetime.now().isoformat()

            if existing_paper:
                # Обновить существующую статью
                logger.info(f"Found existing paper: {paper_url}")

                updated_payload = existing_paper["payload"].copy()

                # Обновить полный текст, если его не было
                if not updated_payload.get("full_text"):
                    updated_payload["full_text"] = fulltext

                # Добавить/обновить валидационные данные
                updated_payload["is_validation_data"] = True
                updated_payload["is_manually_annotated"] = True
                updated_payload["is_aging_theory"] = True  # Все валидационные статьи про теории старения
                updated_payload["validation_timestamp"] = timestamp

                # Обновить вопросы
                updated_payload["questions_classification"] = questions_classification
                updated_payload["validation_questions"] = questions_classification

                # Сохранить метаданные из CSV
                updated_payload["validation_paper_name"] = validation_record["paper_name"]
                updated_payload["validation_paper_year"] = validation_record["paper_year"]
                updated_payload["paper_url"] = paper_url

                # Обновить точку
                updated_point = PointStruct(
                    id=existing_paper["id"],
                    vector=existing_paper["vector"],
                    payload=updated_payload
                )

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[updated_point]
                )

                logger.info(f"Updated paper: {paper_url}")
                return True

            else:
                # Создать новую статью
                logger.info(f"Creating new paper: {paper_url}")

                # Для новой статьи нужен вектор - используем фиктивный (или можно генерировать)
                # В реальности, лучше создавать эмбеддинг для текста
                # Но для валидационных данных можем использовать нулевой вектор
                # или вектор от заголовка/аннотации

                # Получить размерность вектора из существующей коллекции
                collection_info = self.client.get_collection(self.collection_name)
                vector_size = collection_info.config.params.vectors.size

                # Создаем нулевой вектор (можно улучшить, создав реальный эмбеддинг)
                vector = [0.0] * vector_size

                # Найти максимальный ID для нового point
                max_id = 0
                offset = None
                while True:
                    result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False
                    )
                    points, offset = result

                    if points:
                        for point in points:
                            if isinstance(point.id, int) and point.id > max_id:
                                max_id = point.id

                    if offset is None:
                        break

                new_id = max_id + 1

                payload = {
                    "paper_url": paper_url,
                    "title": validation_record["paper_name"],
                    "year": validation_record["paper_year"],
                    "full_text": fulltext,
                    "is_validation_data": True,
                    "is_manually_annotated": True,
                    "is_aging_theory": True,
                    "validation_timestamp": timestamp,
                    "questions_classification": questions_classification,
                    "validation_questions": questions_classification,
                    "validation_paper_name": validation_record["paper_name"],
                    "validation_paper_year": validation_record["paper_year"],
                }

                new_point = PointStruct(
                    id=new_id,
                    vector=vector,
                    payload=payload
                )

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[new_point]
                )

                logger.info(f"Created new paper with ID {new_id}: {paper_url}")
                return True

        except Exception as e:
            logger.error(f"Error updating/creating paper {paper_url}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def upload_validation_data(
        self,
        csv_path: Path,
        md_folder: Path
    ) -> Dict[str, int]:
        """
        Загрузить все валидационные данные

        Args:
            csv_path: Путь к CSV файлу
            md_folder: Папка с MD файлами

        Returns:
            Статистика загрузки
        """
        stats = {
            "processed": 0,
            "updated": 0,
            "created": 0,
            "failed": 0,
            "skipped": 0
        }

        # Прочитать CSV
        validation_data = self.read_csv_data(csv_path)

        if not validation_data:
            logger.error("No validation data to upload")
            return stats

        # Обработать каждую запись
        for record in validation_data:
            stats["processed"] += 1
            paper_url = record["paper_url"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {stats['processed']}/{len(validation_data)}: {paper_url}")
            logger.info(f"{'='*60}")

            # Найти соответствующий MD файл
            md_filename = self.url_to_file.get(paper_url)

            if not md_filename:
                logger.warning(f"No MD file mapping for URL: {paper_url}")
                stats["skipped"] += 1
                continue

            md_path = md_folder / md_filename

            if not md_path.exists():
                logger.error(f"MD file not found: {md_path}")
                stats["failed"] += 1
                continue

            # Прочитать полный текст
            fulltext = self.read_fulltext(md_path)

            if not fulltext:
                logger.error(f"Empty fulltext for: {md_path}")
                stats["failed"] += 1
                continue

            # Проверить, существует ли уже статья
            existing = self.find_paper_by_url(paper_url)

            # Обновить или создать
            success = self.update_or_create_paper(record, fulltext)

            if success:
                if existing:
                    stats["updated"] += 1
                else:
                    stats["created"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"\n{'='*60}")
        logger.info("Upload complete!")
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Updated: {stats['updated']}")
        logger.info(f"Created: {stats['created']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"{'='*60}\n")

        return stats


def main():
    """Главная функция"""

    # Пути к файлам
    base_folder = Path(__file__).parent
    csv_path = base_folder / "validate_data_fulltext.csv"
    md_folder = base_folder

    # Создать загрузчик
    uploader = ValidationDataUploader(qdrant_url="http://localhost:6333")

    # Инициализировать
    uploader.initialize()

    # Загрузить данные
    stats = uploader.upload_validation_data(csv_path, md_folder)

    logger.info("Done!")


if __name__ == "__main__":
    main()
