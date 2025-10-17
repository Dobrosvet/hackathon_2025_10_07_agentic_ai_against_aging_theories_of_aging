import asyncio
from typing import List, Dict, Any, Optional
import httpx
import logging

logger = logging.getLogger(__name__)


class PubMedFetcher:
    """Fetcher for PubMed Central using E-utilities API"""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str, batch_size: int = 10000) -> List[str]:
        """
        Search PubMed Central and return list of ALL PMC IDs (no limit)

        Args:
            query: Search query string
            batch_size: Number of IDs to fetch per request

        Returns:
            List of all PMC IDs matching the query
        """
        try:
            all_ids = []
            retstart = 0

            # First request to get total count
            params = {
                "db": "pmc",
                "term": query,
                "retmax": 0,
                "retmode": "json",
                "usehistory": "y"
            }

            response = await self.client.get(f"{self.base_url}esearch.fcgi", params=params)
            response.raise_for_status()

            data = response.json()
            total_count = int(data.get("esearchresult", {}).get("count", 0))

            logger.info(f"Total papers found for query: {total_count}")

            # Fetch all IDs in batches
            while retstart < total_count:
                params = {
                    "db": "pmc",
                    "term": query,
                    "retstart": retstart,
                    "retmax": batch_size,
                    "retmode": "json",
                    "usehistory": "y"
                }

                response = await self.client.get(f"{self.base_url}esearch.fcgi", params=params)
                response.raise_for_status()

                data = response.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])

                all_ids.extend(id_list)
                retstart += len(id_list)

                logger.info(f"Fetched {len(all_ids)}/{total_count} paper IDs")

                # Small delay to avoid overwhelming the API
                if retstart < total_count:
                    await asyncio.sleep(0.5)

            logger.info(f"Total {len(all_ids)} paper IDs retrieved")
            return all_ids

        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            raise

    async def fetch_papers_batch(self, pmc_ids: List[str], retry_on_429: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch multiple papers metadata in a single batch request (up to 200 IDs)

        Args:
            pmc_ids: List of PubMed Central IDs
            retry_on_429: Whether to retry on rate limit errors

        Returns:
            List of paper data dictionaries
        """
        if not pmc_ids:
            return []

        try:
            # Join IDs with comma (NCBI API accepts comma-separated IDs)
            ids_string = ",".join(pmc_ids)

            params = {
                "db": "pmc",
                "id": ids_string,
                "retmode": "json"
            }

            # Respect rate limits: 3 requests per second
            await asyncio.sleep(0.34)

            response = await self.client.get(f"{self.base_url}esummary.fcgi", params=params)

            # Handle rate limiting
            if response.status_code == 429:
                if retry_on_429:
                    logger.warning("Rate limit hit (429), waiting 2 seconds before retry...")
                    await asyncio.sleep(2.0)
                    return await self.fetch_papers_batch(pmc_ids, retry_on_429=False)
                else:
                    logger.error("Rate limit hit (429) on retry, skipping batch")
                    return []

            response.raise_for_status()
            data = response.json()

            papers = []
            if "result" in data:
                for pmc_id in pmc_ids:
                    if pmc_id in data["result"]:
                        paper_info = data["result"][pmc_id]

                        # Safely extract fields
                        title = paper_info.get("title", "")

                        # Authors can be a list of dicts or just empty
                        authors = []
                        authors_data = paper_info.get("authors", [])
                        if isinstance(authors_data, list):
                            for author in authors_data:
                                if isinstance(author, dict):
                                    name = author.get("name", "")
                                    if name:
                                        authors.append(name)
                                elif isinstance(author, str):
                                    authors.append(author)

                        # Extract article IDs
                        pmid = ""
                        doi = ""
                        articleids = paper_info.get("articleids", [])
                        if isinstance(articleids, list):
                            for article_id in articleids:
                                if isinstance(article_id, dict):
                                    id_type = article_id.get("idtype", "")
                                    value = article_id.get("value", "")
                                    if id_type == "pmid":
                                        pmid = value
                                    elif id_type == "doi":
                                        doi = value

                        paper_data = {
                            "pmc_id": pmc_id,
                            "title": title,
                            "authors": authors,
                            "source": paper_info.get("source", ""),
                            "pubdate": paper_info.get("pubdate", ""),
                            "pmid": pmid,
                            "doi": doi,
                            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/",
                            "pdf_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
                        }

                        papers.append(paper_data)

            return papers

        except Exception as e:
            logger.error(f"Error fetching batch of {len(pmc_ids)} papers: {e}")
            return []

    async def fetch_paper_url(self, pmc_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch single paper URL and metadata (wrapper for batch method)

        Args:
            pmc_id: PubMed Central ID

        Returns:
            Dictionary with paper URL and metadata or None if error
        """
        results = await self.fetch_papers_batch([pmc_id])
        return results[0] if results else None

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
