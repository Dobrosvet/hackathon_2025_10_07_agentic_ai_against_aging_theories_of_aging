import logging
from typing import Dict, Any, List
from pathlib import Path
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Storage handler for Qdrant vector database"""

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
        """Initialize Qdrant client and create collection if needed"""
        try:
            # Create Qdrant client connected to server
            self.client = QdrantClient(url=self.qdrant_url)

            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Create collection with simple vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    def _create_simple_embedding(self, text: str, size: int = 384) -> List[float]:
        """Create a simple deterministic embedding from text using hash"""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(size):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)

        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    async def store_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Store paper URL and metadata in Qdrant"""
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")

            text_for_embedding = paper_data.get('title', '')
            if not text_for_embedding.strip():
                text_for_embedding = f"PMC{paper_data.get('pmc_id', '')}"

            embedding = self._create_simple_embedding(text_for_embedding)
            point_id = int(paper_data.get('pmc_id', '0'))

            payload = {
                "pmc_id": paper_data.get("pmc_id", ""),
                "title": paper_data.get("title", ""),
                "authors": paper_data.get("authors", []),
                "source": paper_data.get("source", ""),
                "pubdate": paper_data.get("pubdate", ""),
                "doi": paper_data.get("doi", ""),
                "pmid": paper_data.get("pmid", ""),
                "url": paper_data.get("url", ""),
                "pdf_url": paper_data.get("pdf_url", "")
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            return True

        except Exception as e:
            logger.error(f"Error storing paper URL in Qdrant: {e}")
            return False

    async def store_papers_batch(self, papers: List[Dict[str, Any]]) -> int:
        """Store multiple papers at once (batch operation)"""
        try:
            if not self.client or not papers:
                return 0

            points = []
            for paper_data in papers:
                text_for_embedding = paper_data.get('title', '')
                if not text_for_embedding.strip():
                    text_for_embedding = f"PMC{paper_data.get('pmc_id', '')}"

                embedding = self._create_simple_embedding(text_for_embedding)
                point_id = int(paper_data.get('pmc_id', '0'))

                payload = {
                    "pmc_id": paper_data.get("pmc_id", ""),
                    "title": paper_data.get("title", ""),
                    "authors": paper_data.get("authors", []),
                    "source": paper_data.get("source", ""),
                    "pubdate": paper_data.get("pubdate", ""),
                    "doi": paper_data.get("doi", ""),
                    "pmid": paper_data.get("pmid", ""),
                    "url": paper_data.get("url", ""),
                    "pdf_url": paper_data.get("pdf_url", "")
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Stored batch of {len(points)} papers")
            return len(points)

        except Exception as e:
            logger.error(f"Error storing paper batch: {e}")
            return 0

    def get_existing_ids(self) -> set:
        """Get all existing PMC IDs from database"""
        try:
            if not self.client:
                return set()

            existing_ids = set()
            offset = None
            batch_size = 1000

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
                    pmc_id = point.payload.get("pmc_id", "")
                    if pmc_id:
                        existing_ids.add(pmc_id)

                if offset is None:
                    break

            logger.info(f"Found {len(existing_ids)} existing papers in database")
            return existing_ids

        except Exception as e:
            logger.error(f"Error getting existing IDs: {e}")
            return set()

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
