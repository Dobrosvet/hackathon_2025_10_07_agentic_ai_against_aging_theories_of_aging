"""
PubMedBERT-based sentence embedding classifier for aging theory detection.

The classifier relies on the ``pritamdeka/S-PubMedBert-MS-MARCO`` sentence-transformer
model to build dense embeddings for both reference aging theories and incoming texts.
Cosine similarity between vectors is then used to produce classification scores and
rank potentially relevant theories. The implementation keeps backward-compatible
behaviour with the legacy Bioformer wrapper, exposing the same public methods and
result schema used by the surrounding microservice.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PubmedBertClassifier:
    """
    Wrapper around a SentenceTransformer model (PubMedBERT fine-tuned for sentence
    similarity) that provides utility methods for single-text classification, batch
    classification and lightweight entity extraction via sliding windows.
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        use_gpu: bool = True,
        quantize: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = None
        self.model = None
        self.theory_embeddings: Optional[Dict[str, Any]] = None

        self._initialized = False
        self._use_gpu = use_gpu
        self._quantize = quantize

        if self._quantize:
            logger.warning(
                "Quantization is not supported for SentenceTransformer models; "
                "ignoring quantize=True."
            )

        logger.info(
            "PubmedBertClassifier created (lazy initialization). Model: %s",
            self.model_name,
        )

    def _initialize(self) -> None:
        """Load the SentenceTransformer model and pre-compute embeddings for theories."""
        if self._initialized:
            return

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for PubMedBERT classifier. "
                "Install sentence-transformers and torch inside the poetry environment."
            ) from exc

        if self._use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for PubMedBERT embeddings")

        logger.info("Loading sentence-transformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        if self.max_length:
            self.model.max_seq_length = self.max_length

        # mark as initialized before precomputing to avoid recursive initialization
        self._initialized = True
        try:
            self._precompute_theory_embeddings()
        except Exception:
            self._initialized = False
            raise

        logger.info("PubMedBERT model loaded successfully")

    def _precompute_theory_embeddings(self) -> None:
        """Generate embeddings for all reference aging theories."""
        from theory_database import theory_db

        theory_texts: List[str] = []
        theory_names: List[str] = []

        for theory_name, patterns in theory_db.get_all_theories().items():
            theory_text = f"aging theory: {', '.join(patterns[:3])}"
            theory_texts.append(theory_text)
            theory_names.append(theory_name)

        embeddings: List[np.ndarray] = []
        for start in range(0, len(theory_texts), self.batch_size):
            batch = theory_texts[start : start + self.batch_size]
            embeddings.extend(self._get_embeddings_batch(batch))

        self.theory_embeddings = {
            "names": theory_names,
            "vectors": np.array(embeddings),
        }
        logger.info("Precomputed embeddings for %d theories", len(theory_names))

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts into normalized embedding vectors."""
        self._initialize()

        embeddings = self.model.encode(
            texts,
            batch_size=min(self.batch_size, max(1, len(texts))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if isinstance(embeddings, np.ndarray):
            return [embeddings[i] for i in range(len(texts))]
        return embeddings  # type: ignore[return-value]

    def classify_text(self, text: str, threshold: float = 0.6) -> Dict[str, Any]:
        """Classify a single text as an aging theory or not."""
        self._initialize()

        if not text or len(text.strip()) < 100:
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "matched_theories": [],
                "reasoning": "Text too short",
            }

        try:
            text_embedding = self._get_embeddings_batch([text])[0]
            similarities = np.dot(self.theory_embeddings["vectors"], text_embedding)  # type: ignore[index]

            top_indices = np.argsort(similarities)[::-1][:5]
            top_theories = [
                {
                    "theory": self.theory_embeddings["names"][idx],  # type: ignore[index]
                    "similarity": float(similarities[idx]),
                }
                for idx in top_indices
            ]

            max_similarity = float(similarities.max())
            is_aging_theory = max_similarity >= threshold

            return {
                "is_aging_theory": is_aging_theory,
                "confidence": round(max_similarity, 3),
                "matched_theories": top_theories,
                "method": "pubmedbert-sbert",
                "threshold": threshold,
            }
        except Exception as exc:
            logger.error("Error in PubMedBERT classification: %s", exc)
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "matched_theories": [],
                "error": str(exc),
            }

    def classify_batch(
        self,
        texts: List[str],
        threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of texts."""
        self._initialize()

        if not texts:
            return []

        try:
            embeddings = self._get_embeddings_batch(texts)
            text_matrix = np.array(embeddings)
            theory_matrix = self.theory_embeddings["vectors"]  # type: ignore[index]
            similarities_matrix = np.dot(text_matrix, theory_matrix.T)

            results: List[Dict[str, Any]] = []
            for idx, text in enumerate(texts):
                if len(text.strip()) < 100:
                    results.append(
                        {
                            "is_aging_theory": False,
                            "confidence": 0.0,
                            "matched_theories": [],
                            "reasoning": "Text too short",
                        }
                    )
                    continue

                similarities = similarities_matrix[idx]
                top_indices = np.argsort(similarities)[::-1][:5]
                top_theories = [
                    {
                        "theory": self.theory_embeddings["names"][ti],  # type: ignore[index]
                        "similarity": float(similarities[ti]),
                    }
                    for ti in top_indices
                ]

                max_similarity = float(similarities.max())
                results.append(
                    {
                        "is_aging_theory": max_similarity >= threshold,
                        "confidence": round(max_similarity, 3),
                        "matched_theories": top_theories,
                        "method": "pubmedbert-sbert-batch",
                        "threshold": threshold,
                    }
                )

            logger.info("Batch classified %d texts with PubMedBERT", len(texts))
            return results
        except Exception as exc:
            logger.error("Error in batch classification: %s", exc)
            return [
                {
                    "is_aging_theory": False,
                    "confidence": 0.0,
                    "matched_theories": [],
                    "error": str(exc),
                }
                for _ in texts
            ]

    def extract_entities_with_sliding_window(
        self,
        text: str,
        window_size: int = 512,
        stride: int = 256,
    ) -> List[Dict[str, Any]]:
        """Approximate entity extraction by sliding window similarity scoring."""
        self._initialize()

        from theory_database import theory_db

        chunks = self._create_sliding_windows(text, window_size, stride)
        entities: List[Dict[str, Any]] = []

        for chunk_text, chunk_start in chunks:
            embedding = self._get_embeddings_batch([chunk_text])[0]
            similarities = np.dot(self.theory_embeddings["vectors"], embedding)  # type: ignore[index]

            if similarities.max() <= 0.5:
                continue

            for theory_name, patterns in theory_db.get_all_theories().items():
                theory_idx = self.theory_embeddings["names"].index(theory_name)  # type: ignore[index]
                theory_similarity = similarities[theory_idx]
                if theory_similarity <= 0.5:
                    continue

                for pattern in patterns:
                    pos = chunk_text.lower().find(pattern.lower())
                    if pos == -1:
                        continue

                    entities.append(
                        {
                            "theory_name": theory_name,
                            "matched_text": chunk_text[pos : pos + len(pattern)],
                            "start_position": chunk_start + pos,
                            "end_position": chunk_start + pos + len(pattern),
                            "confidence": float(theory_similarity),
                            "method": "pubmedbert-scored-keyword",
                        }
                    )

        deduped = self._deduplicate_entities(entities)
        logger.info("Found %d entity mentions with PubMedBERT scoring", len(deduped))
        return deduped

    def _create_sliding_windows(
        self,
        text: str,
        window_size: int,
        stride: int,
    ) -> List[Tuple[str, int]]:
        """Create overlapping character windows for the provided text."""
        char_window = window_size * 4
        char_stride = stride * 4

        windows: List[Tuple[str, int]] = []
        start = 0

        while start < len(text):
            end = min(start + char_window, len(text))
            windows.append((text[start:end], start))
            start += char_stride
            if end >= len(text):
                break

        return windows

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicated entity mentions based on theory and span."""
        seen = set()
        unique: List[Dict[str, Any]] = []

        for entity in entities:
            key = (entity["theory_name"], entity["start_position"], entity["end_position"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)

        return unique

    def get_model_info(self, force_initialize: bool = False) -> Dict[str, Any]:
        """Return model metadata without forcing heavy initialization by default."""
        if force_initialize and not self._initialized:
            try:
                self._initialize()
            except Exception as exc:
                logger.warning("Failed to eagerly initialize PubMedBERT: %s", exc)

        info = {
            "model_name": self.model_name,
            "initialized": self._initialized,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": str(self.device) if self.device else None,
        }

        if self._initialized:
            try:
                import torch

                if self.device and self.device.type == "cuda":
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    info["gpu_memory"] = f"{gpu_memory:.2f} GB"
            except Exception:
                logger.debug("Failed to fetch GPU information", exc_info=True)

        return info


# Backwards compatibility alias for legacy imports
BioformerClassifier = PubmedBertClassifier
