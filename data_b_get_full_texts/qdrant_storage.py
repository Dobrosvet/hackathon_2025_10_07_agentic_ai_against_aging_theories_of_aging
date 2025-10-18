import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Storage handler for Qdrant vector database - Full Text Service"""

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
                raise ValueError(f"Collection {self.collection_name} not found. Run data_a service first.")
            else:
                logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get all papers from database

        Returns:
            List of paper dictionaries with id and payload
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
                    papers.append({
                        "id": point.id,
                        "pmc_id": point.payload.get("pmc_id", ""),
                        "title": point.payload.get("title", ""),
                        "url": point.payload.get("url", ""),
                        "has_full_text": "full_text" in point.payload and bool(point.payload.get("full_text"))
                    })

                if offset is None:
                    break

            logger.info(f"Retrieved {len(papers)} papers from database")
            return papers

        except Exception as e:
            logger.error(f"Error getting all papers: {e}")
            return []

    def get_papers_without_full_text(self) -> List[Dict[str, Any]]:
        """
        Get papers that don't have full text yet

        Returns:
            List of paper dictionaries without full_text field
        """
        try:
            all_papers = self.get_all_papers()
            papers_without_text = [p for p in all_papers if not p["has_full_text"]]

            logger.info(f"Found {len(papers_without_text)} papers without full text")
            return papers_without_text

        except Exception as e:
            logger.error(f"Error getting papers without full text: {e}")
            return []

    def get_papers_with_full_text_count(self) -> int:
        """
        Count papers that already have full text

        Returns:
            Number of papers with full_text field
        """
        try:
            all_papers = self.get_all_papers()
            count = sum(1 for p in all_papers if p["has_full_text"])

            logger.info(f"Found {count} papers with full text")
            return count

        except Exception as e:
            logger.error(f"Error counting papers with full text: {e}")
            return 0

    async def update_paper_with_full_text(self, point_id: int, pmc_id: str, full_text: str) -> bool:
        """
        Update a paper record with full text

        Args:
            point_id: Qdrant point ID
            pmc_id: PubMed Central ID
            full_text: Full text content

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

            # Update payload with full_text
            existing_point = point[0]
            updated_payload = existing_point.payload.copy()
            updated_payload["full_text"] = full_text

            # Update point in Qdrant
            from qdrant_client.models import PointStruct

            updated_point = PointStruct(
                id=point_id,
                vector=existing_point.vector,
                payload=updated_payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logger.info(f"Updated PMC{pmc_id} with full text ({len(full_text)} chars)")
            return True

        except Exception as e:
            logger.error(f"Error updating paper {pmc_id} with full text: {e}")
            return False

    async def update_papers_batch(self, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple papers with full text in batch

        Args:
            updates: List of dicts with keys: point_id, pmc_id, full_text

        Returns:
            Number of successfully updated papers
        """
        try:
            if not self.client or not updates:
                return 0

            from qdrant_client.models import PointStruct

            updated_points = []

            for update in updates:
                point_id = update["point_id"]
                full_text = update["full_text"]

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
                        updated_payload["full_text"] = full_text

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
