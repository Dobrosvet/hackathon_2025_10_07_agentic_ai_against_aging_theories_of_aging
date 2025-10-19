import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Storage handler for Qdrant vector database - Classifier Service"""

    def __init__(self, db_path: str = None, qdrant_url: str = "http://localhost:6333"):
        """
        Initialize Qdrant storage

        Args:
            db_path: Path to Qdrant database directory (deprecated, kept for compatibility)
            qdrant_url: URL of Qdrant server (default: http://localhost:6333)
        """
        self.db_path = Path(db_path) if db_path else None
        self.qdrant_url = qdrant_url
        self.collection_name = "pubmed_papers"
        self.client = None

    async def initialize(self):
        """Initialize Qdrant client"""
        try:
            # Create Qdrant client connected to server
            self.client = QdrantClient(url=self.qdrant_url)

            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.error(f"Collection {self.collection_name} does not exist!")
                raise ValueError(f"Collection {self.collection_name} not found. Run services A and B first.")
            else:
                logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    def get_papers_with_full_text(self) -> List[Dict[str, Any]]:
        """
        Get papers that have full text (for classification)

        Returns:
            List of paper dictionaries with id, pmc_id, title, full_text
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
                    # Проверяем наличие full_text
                    if "full_text" in point.payload and point.payload.get("full_text"):
                        papers.append({
                            "id": point.id,
                            "pmc_id": point.payload.get("pmc_id", ""),
                            "title": point.payload.get("title", ""),
                            "full_text": point.payload.get("full_text", ""),
                            "has_classification": "is_aging_theory" in point.payload
                        })

                if offset is None:
                    break

            logger.info(f"Retrieved {len(papers)} papers with full text")
            return papers

        except Exception as e:
            logger.error(f"Error getting papers with full text: {e}")
            return []

    def get_papers_without_classification(self) -> List[Dict[str, Any]]:
        """
        Get papers that have full text but not yet classified

        Returns:
            List of paper dictionaries without classification
        """
        try:
            all_papers = self.get_papers_with_full_text()
            papers_without_classification = [
                p for p in all_papers if not p["has_classification"]
            ]

            logger.info(f"Found {len(papers_without_classification)} papers without classification")
            return papers_without_classification

        except Exception as e:
            logger.error(f"Error getting papers without classification: {e}")
            return []

    def get_classified_papers_count(self) -> int:
        """
        Count papers that already have classification

        Returns:
            Number of classified papers
        """
        try:
            all_papers = self.get_papers_with_full_text()
            count = sum(1 for p in all_papers if p["has_classification"])

            logger.info(f"Found {count} classified papers")
            return count

        except Exception as e:
            logger.error(f"Error counting classified papers: {e}")
            return 0

    async def update_paper_with_classification(
        self,
        point_id: int,
        pmc_id: str,
        classification_result: Dict[str, Any]
    ) -> bool:
        """
        Update a paper record with classification results

        Args:
            point_id: Qdrant point ID
            pmc_id: PubMed Central ID
            classification_result: Dict with classification data

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            # Get existing point
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not point:
                logger.error(f"Point {point_id} not found")
                return False

            # Update payload with classification results
            existing_point = point[0]
            updated_payload = existing_point.payload.copy()

            # Add classification fields
            updated_payload["is_aging_theory"] = classification_result.get("is_aging_theory", False)
            updated_payload["classification_confidence"] = classification_result.get("classification_confidence", 0.0)
            updated_payload["aging_theories"] = classification_result.get("aging_theories", [])
            updated_payload["classification_model"] = classification_result.get("classification_model", "bioformer-8L")
            updated_payload["classification_timestamp"] = datetime.now().isoformat()
            updated_payload["classification_version"] = classification_result.get("classification_version", "1.0")

            # Update point in Qdrant
            updated_point = PointStruct(
                id=point_id,
                vector=existing_point.vector,
                payload=updated_payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            theories_count = len(classification_result.get("aging_theories", []))
            logger.info(
                f"Updated PMC{pmc_id}: is_theory={classification_result.get('is_aging_theory')}, "
                f"theories_found={theories_count}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating paper {pmc_id} with classification: {e}")
            return False

    async def update_papers_batch(self, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple papers with classification results in batch

        Args:
            updates: List of dicts with keys: point_id, pmc_id, classification_result

        Returns:
            Number of successfully updated papers
        """
        try:
            if not self.client or not updates:
                return 0

            updated_points = []

            for update in updates:
                point_id = update["point_id"]
                classification_result = update["classification_result"]

                # Get existing point
                try:
                    point = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[point_id],
                        with_payload=True,
                        with_vectors=True
                    )

                    if point:
                        existing_point = point[0]
                        updated_payload = existing_point.payload.copy()

                        # Add classification fields
                        updated_payload["is_aging_theory"] = classification_result.get("is_aging_theory", False)
                        updated_payload["classification_confidence"] = classification_result.get("classification_confidence", 0.0)
                        updated_payload["aging_theories"] = classification_result.get("aging_theories", [])
                        updated_payload["classification_model"] = classification_result.get("classification_model", "bioformer-8L")
                        updated_payload["classification_timestamp"] = datetime.now().isoformat()
                        updated_payload["classification_version"] = classification_result.get("classification_version", "1.0")

                        updated_point = PointStruct(
                            id=point_id,
                            vector=existing_point.vector,
                            payload=updated_payload
                        )
                        updated_points.append(updated_point)

                except Exception as e:
                    logger.error(f"Error preparing update for point {point_id}: {e}")

            # Batch update
            if updated_points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=updated_points
                )

                logger.info(f"Updated batch of {len(updated_points)} papers")
                return len(updated_points)

            return 0

        except Exception as e:
            logger.error(f"Error updating papers batch: {e}")
            return 0

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.client:
                return {"error": "Client not initialized", "count": 0}

            collection_info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "count": collection_info.points_count,
                "status": collection_info.status
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e), "count": 0}

    def get_paper_by_pmc_id(self, pmc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper data by PMC ID

        Args:
            pmc_id: PubMed Central ID

        Returns:
            Paper data dict or None
        """
        try:
            if not self.client:
                return None

            # Search by pmc_id in payload
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
                            **point.payload
                        }

                if offset is None:
                    break

            return None

        except Exception as e:
            logger.error(f"Error getting paper {pmc_id}: {e}")
            return None

    def get_unreviewed_papers(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get papers that need manual review (have full text but no manual label)

        Args:
            limit: Maximum number of papers to return
            offset: Number of papers to skip

        Returns:
            List of paper dictionaries without manual labels
        """
        try:
            if not self.client:
                return []

            papers = []
            scroll_offset = None
            batch_size = 100
            current_count = 0

            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, scroll_offset = result

                if not points:
                    break

                for point in points:
                    # Check if has full text and no manual label
                    if ("full_text" in point.payload and
                        point.payload.get("full_text") and
                        "manual_label" not in point.payload):

                        # Skip first 'offset' items
                        if current_count < offset:
                            current_count += 1
                            continue

                        # Add to results
                        papers.append({
                            "id": point.id,
                            "pmc_id": point.payload.get("pmc_id", ""),
                            "title": point.payload.get("title", ""),
                            "full_text": point.payload.get("full_text", ""),
                            "is_aging_theory": point.payload.get("is_aging_theory"),
                            "classification_confidence": point.payload.get("classification_confidence"),
                            "aging_theories": point.payload.get("aging_theories", [])
                        })

                        # Check if we've reached the limit
                        if len(papers) >= limit:
                            logger.info(f"Retrieved {len(papers)} unreviewed papers (offset={offset})")
                            return papers

                if scroll_offset is None:
                    break

            logger.info(f"Retrieved {len(papers)} unreviewed papers (offset={offset})")
            return papers

        except Exception as e:
            logger.error(f"Error getting unreviewed papers: {e}")
            return []

    async def save_manual_label(
        self,
        pmc_id: str,
        label: bool,
        user_comment: str = ""
    ) -> bool:
        """
        Save manual label for a paper

        Args:
            pmc_id: PubMed Central ID
            label: True if aging theory, False if not
            user_comment: Optional comment from user

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            # Find paper by pmc_id
            paper = self.get_paper_by_pmc_id(pmc_id)
            if not paper:
                logger.error(f"Paper {pmc_id} not found")
                return False

            point_id = paper["id"]

            # Get existing point
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not point:
                logger.error(f"Point {point_id} not found")
                return False

            # Update payload with manual label
            existing_point = point[0]
            updated_payload = existing_point.payload.copy()

            # Add manual labeling fields
            updated_payload["manual_label"] = label
            updated_payload["manual_label_timestamp"] = datetime.now().isoformat()
            updated_payload["user_comment"] = user_comment
            updated_payload["review_status"] = "labeled"

            # Update point in Qdrant
            updated_point = PointStruct(
                id=point_id,
                vector=existing_point.vector,
                payload=updated_payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logger.info(f"Saved manual label for PMC{pmc_id}: label={label}")
            return True

        except Exception as e:
            logger.error(f"Error saving manual label for {pmc_id}: {e}")
            return False

    async def skip_paper(self, pmc_id: str) -> bool:
        """
        Mark paper as skipped for manual review

        Args:
            pmc_id: PubMed Central ID

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            # Find paper by pmc_id
            paper = self.get_paper_by_pmc_id(pmc_id)
            if not paper:
                logger.error(f"Paper {pmc_id} not found")
                return False

            point_id = paper["id"]

            # Get existing point
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not point:
                logger.error(f"Point {point_id} not found")
                return False

            # Update payload
            existing_point = point[0]
            updated_payload = existing_point.payload.copy()

            # Mark as skipped
            updated_payload["review_status"] = "skipped"
            updated_payload["skip_timestamp"] = datetime.now().isoformat()

            # Update point in Qdrant
            updated_point = PointStruct(
                id=point_id,
                vector=existing_point.vector,
                payload=updated_payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logger.info(f"Marked PMC{pmc_id} as skipped")
            return True

        except Exception as e:
            logger.error(f"Error skipping paper {pmc_id}: {e}")
            return False

    def get_labeling_stats(self) -> Dict[str, Any]:
        """
        Get statistics about manual labeling

        Returns:
            Dict with labeling statistics
        """
        try:
            if not self.client:
                return {
                    "total_labeled": 0,
                    "labeled_as_theory": 0,
                    "labeled_as_not_theory": 0,
                    "skipped": 0,
                    "unreviewed": 0
                }

            stats = {
                "total_labeled": 0,
                "labeled_as_theory": 0,
                "labeled_as_not_theory": 0,
                "skipped": 0,
                "unreviewed": 0
            }

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
                    # Only count papers with full text
                    if "full_text" in point.payload and point.payload.get("full_text"):
                        if "manual_label" in point.payload:
                            stats["total_labeled"] += 1
                            if point.payload["manual_label"]:
                                stats["labeled_as_theory"] += 1
                            else:
                                stats["labeled_as_not_theory"] += 1
                        elif point.payload.get("review_status") == "skipped":
                            stats["skipped"] += 1
                        else:
                            stats["unreviewed"] += 1

                if offset is None:
                    break

            logger.info(f"Labeling stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting labeling stats: {e}")
            return {
                "total_labeled": 0,
                "labeled_as_theory": 0,
                "labeled_as_not_theory": 0,
                "skipped": 0,
                "unreviewed": 0,
                "error": str(e)
            }
