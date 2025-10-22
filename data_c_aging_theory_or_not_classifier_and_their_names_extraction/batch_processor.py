"""
Batch processor для пакетной классификации статей об теориях старения.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BatchProcessor:
    def __init__(self, classifier, batch_size: int = 32, max_workers: int = 2):
        self.classifier = classifier
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(
            "BatchProcessor initialized (batch_size=%d, max_workers=%d)",
            batch_size,
            max_workers,
        )

    def process_batch(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not papers:
            return []

        results: List[Dict[str, Any]] = []

        if getattr(self.classifier, "supports_batch_embeddings", False) and getattr(self.classifier, "bioformer", None):
            filtered_papers: List[Dict[str, Any]] = []
            for paper in papers:
                keyword_result = self.classifier._classify_keyword_only(paper.get("text", ""))
                if keyword_result["keyword_matches"] > 0:
                    filtered_papers.append({**paper, "_keyword_result": keyword_result})
                else:
                    results.append(
                        {
                            "pmc_id": paper.get("pmc_id"),
                            "is_aging_theory": False,
                            "confidence": 0.0,
                            "method": "batch-keyword-rejected",
                        }
                    )

            if filtered_papers:
                texts = [p.get("text", "") for p in filtered_papers]
                embedding_results = self.classifier.bioformer.classify_batch(texts)  # type: ignore[union-attr]

                for paper, embedding_result in zip(filtered_papers, embedding_results):
                    result = dict(embedding_result)
                    result["pmc_id"] = paper.get("pmc_id")
                    results.append(result)
        else:
            for paper in papers:
                result = self.classifier.process_paper(paper.get("text", ""))
                result["pmc_id"] = paper.get("pmc_id")
                results.append(result)

        logger.info("Processed batch of %d papers", len(papers))
        return results

    async def process_papers_async(
        self,
        papers: List[Dict[str, Any]],
        callback: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []

        for start in range(0, len(papers), self.batch_size):
            batch = papers[start : start + self.batch_size]
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                self.executor,
                self.process_batch,
                batch,
            )
            all_results.extend(batch_results)

            if callback:
                await callback(batch_results, start + len(batch), len(papers))

        return all_results

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
