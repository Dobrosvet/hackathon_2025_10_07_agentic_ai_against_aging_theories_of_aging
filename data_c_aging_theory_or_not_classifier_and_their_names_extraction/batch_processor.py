"""
Batch processor для эффективной обработки больших объемов статей
Поддерживает GPU batching и асинхронную обработку
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Процессор для батчевой обработки статей
    Оптимизирован для GPU throughput
    """

    def __init__(
        self,
        classifier,
        batch_size: int = 32,
        max_workers: int = 2
    ):
        """
        Args:
            classifier: Экземпляр AgingTheoryClassifier
            batch_size: Размер батча для GPU
            max_workers: Количество CPU worker'ов
        """
        self.classifier = classifier
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"BatchProcessor initialized (batch_size={batch_size})")

    def process_batch(
        self,
        papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Обработать батч статей

        Args:
            papers: Список статей с полями text, pmc_id, etc.

        Returns:
            Список результатов классификации
        """
        if not papers:
            return []

        results = []

        # Если classifier поддерживает batch mode (Bioformer)
        if (hasattr(self.classifier, 'mode') and
            self.classifier.mode in ['bioformer', 'hybrid'] and
            self.classifier.bioformer):

            # Stage 1: Keyword pre-filtering
            filtered_papers = []
            for paper in papers:
                # Быстрая проверка через keyword
                keyword_result = self.classifier._classify_keyword_only(
                    paper.get('text', '')
                )
                if keyword_result['keyword_matches'] > 0:
                    filtered_papers.append(paper)
                else:
                    # Отклонено keyword фильтром
                    results.append({
                        'pmc_id': paper.get('pmc_id'),
                        'is_aging_theory': False,
                        'confidence': 0.0,
                        'method': 'batch-keyword-rejected'
                    })

            # Stage 2: Batch Bioformer classification
            if filtered_papers:
                texts = [p.get('text', '') for p in filtered_papers]
                bioformer_results = self.classifier.bioformer.classify_batch(texts)

                for i, paper in enumerate(filtered_papers):
                    result = bioformer_results[i]
                    result['pmc_id'] = paper.get('pmc_id')
                    results.append(result)

        else:
            # Fallback: обычная последовательная обработка
            for paper in papers:
                result = self.classifier.process_paper(paper.get('text', ''))
                result['pmc_id'] = paper.get('pmc_id')
                results.append(result)

        logger.info(f"Processed batch of {len(papers)} papers")
        return results

    async def process_papers_async(
        self,
        papers: List[Dict[str, Any]],
        callback=None
    ) -> List[Dict[str, Any]]:
        """
        Асинхронная обработка с callback для обновления UI

        Args:
            papers: Список статей
            callback: Функция для callback после каждого батча

        Returns:
            Список всех результатов
        """
        all_results = []

        # Разбить на батчи
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i + self.batch_size]

            # Обработать батч в отдельном потоке
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                self.executor,
                self.process_batch,
                batch
            )

            all_results.extend(batch_results)

            # Callback для обновления UI
            if callback:
                await callback(batch_results, i + len(batch), len(papers))

        return all_results

    def shutdown(self):
        """Завершить работу"""
        self.executor.shutdown(wait=True)
