import asyncio
from typing import Optional, Dict, Any
import httpx
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class PMCTextFetcher:
    """Fetcher for full text articles from PubMed Central using E-utilities API"""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.client = httpx.AsyncClient(timeout=60.0)

    async def fetch_full_text(self, pmc_id: str, retry_on_429: bool = True) -> Optional[str]:
        """
        Fetch full text of an article from PubMed Central

        Args:
            pmc_id: PubMed Central ID (without 'PMC' prefix)
            retry_on_429: Whether to retry on rate limit errors

        Returns:
            Full text as string or None if error
        """
        try:
            # Respect rate limits: 3 requests per second
            await asyncio.sleep(0.34)

            params = {
                "db": "pmc",
                "id": pmc_id,
                "retmode": "xml"
            }

            response = await self.client.get(f"{self.base_url}efetch.fcgi", params=params)

            # Handle rate limiting
            if response.status_code == 429:
                if retry_on_429:
                    logger.warning(f"Rate limit hit for PMC{pmc_id}, waiting 2 seconds...")
                    await asyncio.sleep(2.0)
                    return await self.fetch_full_text(pmc_id, retry_on_429=False)
                else:
                    logger.error(f"Rate limit hit on retry for PMC{pmc_id}")
                    return None

            response.raise_for_status()
            xml_content = response.text

            # Parse XML and extract text
            full_text = self._extract_text_from_xml(xml_content)

            if full_text:
                logger.info(f"Successfully fetched full text for PMC{pmc_id} ({len(full_text)} chars)")
                return full_text
            else:
                logger.warning(f"No text content found in PMC{pmc_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching full text for PMC{pmc_id}: {e}")
            return None

    def _extract_text_from_xml(self, xml_content: str) -> Optional[str]:
        """
        Extract plain text from PMC XML format

        Args:
            xml_content: XML string from PMC API

        Returns:
            Plain text string or None
        """
        try:
            root = ET.fromstring(xml_content)

            # Find article body - PMC XML has structure: pmc-articleset > article > body
            text_parts = []

            # Try to find article body
            body = root.find(".//body")
            if body is not None:
                text_parts.append(self._get_element_text(body))

            # Also try to get abstract if body is not available
            if not text_parts:
                abstract = root.find(".//abstract")
                if abstract is not None:
                    text_parts.append(self._get_element_text(abstract))

            # Get title as well
            title = root.find(".//article-title")
            if title is not None:
                title_text = self._get_element_text(title)
                if title_text:
                    text_parts.insert(0, f"Title: {title_text}")

            if text_parts:
                full_text = "\n\n".join(text_parts)
                # Clean up excessive whitespace
                full_text = " ".join(full_text.split())
                return full_text

            return None

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from XML: {e}")
            return None

    def _get_element_text(self, element: ET.Element) -> str:
        """
        Recursively extract all text from an XML element

        Args:
            element: XML Element

        Returns:
            Concatenated text content
        """
        text_parts = []

        # Get direct text
        if element.text:
            text_parts.append(element.text.strip())

        # Recursively get text from children
        for child in element:
            child_text = self._get_element_text(child)
            if child_text:
                text_parts.append(child_text)

            # Get tail text (text after closing tag)
            if child.tail:
                text_parts.append(child.tail.strip())

        return " ".join(filter(None, text_parts))

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
