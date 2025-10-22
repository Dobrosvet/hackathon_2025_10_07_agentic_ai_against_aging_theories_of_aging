import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from pubmedbert_classifier import PubmedBertClassifier

logger = logging.getLogger(__name__)


class AgingTheoryClassifier:
    """
    Классификатор теорий старения с переключаемыми режимами:
    - keyword  : быстрый эвристический поиск по шаблонам
    - embedding: sentence-transformers (PubMedBERT) + косинусное сходство
    - hybrid   : keyword + embedding
    - gpt4o    : GPT-4o-mini через OpenAI API
    """

    def __init__(
        self,
        mode: str = "embedding",
        use_gpu: bool = True,
        bioformer_threshold: float = 0.6,
        quantize: bool = False,
        embedding_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        llm_model_name: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 256,
        openai_api_key: Optional[str] = None,
    ):
        normalized_mode = (mode or "embedding").lower()
        if normalized_mode == "bioformer":
            normalized_mode = "embedding"

        self.mode = normalized_mode
        self.version = "3.0"
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._openai_client = None

        self.bioformer_threshold = bioformer_threshold
        self.embedding_threshold = bioformer_threshold

        try:
            from theory_database import theory_db

            self.theory_patterns = theory_db.get_all_theories()
            logger.info("Loaded %d theory patterns", len(self.theory_patterns))
        except ImportError:
            logger.warning("theory_database not found, using fallback patterns")
            self.theory_patterns = self._get_fallback_patterns()

        self.context_keywords = [
            "aging",
            "ageing",
            "senescence",
            "longevity",
            "lifespan",
            "theory",
            "hypothesis",
            "mechanism",
            "gerontology",
            "age-related",
            "age-associated",
            "hallmarks",
        ]

        self.bioformer: Optional[PubmedBertClassifier] = None
        if self.mode in ["embedding", "hybrid"]:
            try:
                self.bioformer = PubmedBertClassifier(
                    model_name=self.embedding_model_name,
                    use_gpu=use_gpu,
                    quantize=quantize,
                    batch_size=16,
                    max_length=512,
                )
                logger.info("PubMedBERT embedding classifier ready (lazy loading)")
            except Exception as exc:
                logger.warning(
                    "Failed to initialize PubMedBERT classifier (%s). Switching to keyword mode.",
                    exc,
                )
                self.mode = "keyword"
                self.bioformer = None

        self.supports_batch_embeddings = self.mode in {"embedding", "hybrid"} and self.bioformer is not None

        if self.mode == "gpt4o":
            self.model_name = self.llm_model_name
        elif self.mode == "keyword":
            self.model_name = "keyword-matcher"
        elif self.mode == "hybrid":
            self.model_name = "keyword+pubmedbert"
        else:
            self.model_name = "pubmedbert-sbert"

        logger.info(
            "AgingTheoryClassifier initialized. mode=%s, model=%s, version=%s",
            self.mode,
            self.model_name,
            self.version,
        )

    # ---------------------------------------------------------------------
    # Classification entrypoints
    # ---------------------------------------------------------------------
    def classify_aging_theory(self, text: str) -> Dict[str, Any]:
        try:
            if self.mode == "keyword":
                return self._classify_keyword_only(text)
            if self.mode == "embedding":
                return self._classify_embedding_only(text)
            if self.mode == "hybrid":
                return self._classify_hybrid(text)
            if self.mode == "gpt4o":
                return self._classify_gpt4o_only(text)
            raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as exc:
            logger.error("Classification error: %s", exc)
            return {
                "is_aging_theory": False,
                "confidence": 0.0,
                "keyword_matches": 0,
                "error": str(exc),
                "method": self.mode,
            }

    # Backwards compatibility aliases
    def _classify_bioformer_only(self, text: str) -> Dict[str, Any]:
        return self._classify_embedding_only(text)

    # ---------------------------------------------------------------------
    # Keyword classification
    # ---------------------------------------------------------------------
    def _classify_keyword_only(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        theory_matches = 0
        matched_theories: List[str] = []

        for theory_name, patterns in self.theory_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    theory_matches += 1
                    if theory_name not in matched_theories:
                        matched_theories.append(theory_name)
                    break

        context_matches = sum(1 for keyword in self.context_keywords if keyword in text_lower)
        is_theory = theory_matches >= 2 or (theory_matches >= 1 and context_matches >= 5)

        if is_theory:
            base_confidence = min(theory_matches / 5.0, 0.7)
            context_bonus = min(context_matches / 20.0, 0.3)
            confidence = min(base_confidence + context_bonus, 1.0)
        else:
            confidence = max(0.0, theory_matches / 10.0)

        return {
            "is_aging_theory": is_theory,
            "confidence": round(confidence, 3),
            "keyword_matches": theory_matches,
            "context_matches": context_matches,
            "matched_theories_count": len(matched_theories),
            "matched_theories": matched_theories[:5],
            "method": "keyword",
        }

    # ---------------------------------------------------------------------
    # Embedding classification
    # ---------------------------------------------------------------------
    def _classify_embedding_only(self, text: str) -> Dict[str, Any]:
        if not self.bioformer:
            logger.warning("Embedding classifier unavailable, falling back to keywords")
            return self._classify_keyword_only(text)

        result = self.bioformer.classify_text(text, threshold=self.embedding_threshold)
        top_theories = result.get("matched_theories", [])

        return {
            "is_aging_theory": result.get("is_aging_theory", False),
            "confidence": result.get("confidence", 0.0),
            "matched_theories": [item.get("theory") for item in top_theories[:5]],
            "matched_theories_count": len(top_theories),
            "pubmedbert_top_theories": top_theories[:3],
            "method": "pubmedbert",
        }

    # ---------------------------------------------------------------------
    # Hybrid classification
    # ---------------------------------------------------------------------
    def _classify_hybrid(self, text: str) -> Dict[str, Any]:
        keyword_result = self._classify_keyword_only(text)

        if keyword_result["keyword_matches"] == 0 and keyword_result["context_matches"] < 3:
            keyword_result["method"] = "hybrid-keyword-rejected"
            return keyword_result

        if keyword_result["keyword_matches"] >= 5 and keyword_result["context_matches"] >= 10:
            keyword_result["method"] = "hybrid-keyword-accepted"
            return keyword_result

        if not self.bioformer:
            keyword_result["method"] = "hybrid-keyword-fallback"
            return keyword_result

        embedding_result = self.bioformer.classify_text(text, threshold=self.embedding_threshold)
        embedding_conf = embedding_result.get("confidence", 0.0)

        if embedding_conf >= 0.8:
            return {
                "is_aging_theory": embedding_result.get("is_aging_theory", False),
                "confidence": embedding_conf,
                "matched_theories": [
                    item.get("theory")
                    for item in embedding_result.get("matched_theories", [])[:5]
                ],
                "keyword_matches": keyword_result["keyword_matches"],
                "context_matches": keyword_result["context_matches"],
                "pubmedbert_confidence": embedding_conf,
                "method": "hybrid-pubmedbert-decided",
            }

        combined_confidence = (
            keyword_result["confidence"] * 0.4 + embedding_conf * 0.6
        )
        is_theory = combined_confidence >= 0.5

        return {
            "is_aging_theory": is_theory,
            "confidence": round(combined_confidence, 3),
            "matched_theories": [
                item.get("theory")
                for item in embedding_result.get("matched_theories", [])[:5]
            ],
            "keyword_matches": keyword_result["keyword_matches"],
            "context_matches": keyword_result["context_matches"],
            "keyword_confidence": keyword_result["confidence"],
            "pubmedbert_confidence": embedding_conf,
            "method": "hybrid-combined",
        }

    # ---------------------------------------------------------------------
    # GPT-4o classification
    # ---------------------------------------------------------------------
    def _classify_gpt4o_only(self, text: str) -> Dict[str, Any]:
        client = self._ensure_openai_client()

        system_prompt = (
            "You are a senior longevity researcher. "
            "Analyse the provided passage and decide whether it proposes or "
            "discusses a scientific theory of aging. Respond in JSON."
        )
        user_prompt = (
            "Return a JSON object with keys: "
            "`is_aging_theory` (boolean), "
            "`confidence` (float 0-1), "
            "`matched_theories` (array of theory names or empty array), "
            "`rationale` (string). "
            "Text:\n"
            f"{text.strip()}"
        )

        response = client.chat.completions.create(
            model=self.llm_model_name,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("GPT-4o response is not valid JSON: %s", content)
            parsed = {}

        matched_theories = parsed.get("matched_theories") or []
        if isinstance(matched_theories, str):
            matched_theories = [matched_theories]

        return {
            "is_aging_theory": bool(parsed.get("is_aging_theory")),
            "confidence": float(parsed.get("confidence", 0.0)),
            "matched_theories": matched_theories[:5],
            "matched_theories_count": len(matched_theories),
            "rationale": parsed.get("rationale", ""),
            "method": "gpt4o-mini",
        }

    def _ensure_openai_client(self):
        if self._openai_client:
            return self._openai_client

        if not self._openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for GPT-4o classification")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for GPT-4o classification. "
                "Install it inside the poetry environment."
            ) from exc

        self._openai_client = OpenAI(api_key=self._openai_api_key)
        logger.info("OpenAI client initialized for model %s", self.llm_model_name)
        return self._openai_client

    # ---------------------------------------------------------------------
    # Entity extraction
    # ---------------------------------------------------------------------
    def extract_theory_names(self, text: str) -> List[Dict[str, Any]]:
        if not text or len(text.strip()) < 100:
            return []

        keyword_entities = self._extract_keyword_entities(text)

        if self.mode in {"embedding", "hybrid"} and self.bioformer:
            embedding_entities = self.bioformer.extract_entities_with_sliding_window(text)
            merged = self._merge_entities(keyword_entities, embedding_entities)
        else:
            merged = keyword_entities

        merged.sort(key=lambda ent: ent["start_position"])
        logger.info("Found %d theory mentions (mode=%s)", len(merged), self.mode)
        return merged

    def _extract_keyword_entities(self, text: str) -> List[Dict[str, Any]]:
        text_lower = text.lower()
        entities: List[Dict[str, Any]] = []

        for theory_name, patterns in self.theory_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                for match in re.finditer(re.escape(pattern_lower), text_lower):
                    start, end = match.start(), match.end()
                    matched_text = text[start:end]
                    entities.append(
                        {
                            "theory_name": theory_name,
                            "matched_text": matched_text,
                            "start_position": start,
                            "end_position": end,
                            "confidence": 0.3,
                            "method": "keyword",
                        }
                    )

        return entities

    def _merge_entities(self, keyword_entities, embedding_entities):
        merged = {(
            entity["theory_name"],
            entity["start_position"],
            entity["end_position"],
        ): entity for entity in keyword_entities}

        for entity in embedding_entities:
            key = (
                entity["theory_name"],
                entity["start_position"],
                entity["end_position"],
            )
            if key in merged:
                merged_entity = merged[key]
                merged_entity["confidence"] = max(
                    merged_entity.get("confidence", 0.0),
                    entity.get("confidence", 0.0),
                )
                merged_entity["method"] = "keyword+pubmedbert"
            else:
                merged[key] = entity

        return list(merged.values())

    # Backwards compatibility alias
    def _score_entities_with_bioformer(self, text: str, entities: List[Dict[str, Any]]):
        if not self.bioformer:
            return entities

        embedding_entities = self.bioformer.extract_entities_with_sliding_window(text)
        return self._merge_entities(entities, embedding_entities)

    # ---------------------------------------------------------------------
    # End-to-end processing
    # ---------------------------------------------------------------------
    def process_paper(self, text: str) -> Dict[str, Any]:
        classification = self.classify_aging_theory(text)
        theories = self.extract_theory_names(text) if classification["is_aging_theory"] else []

        return {
            "is_aging_theory": classification["is_aging_theory"],
            "classification_confidence": classification["confidence"],
            "aging_theories": theories,
            "classification_model": self.model_name,
            "classification_version": self.version,
            "classification_method": classification.get("method", self.mode),
            "total_theories_found": len(theories),
            "keyword_matches": classification.get("keyword_matches", 0),
            "context_matches": classification.get("context_matches", 0),
        }

    # ---------------------------------------------------------------------
    # Info helpers
    # ---------------------------------------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_name": self.model_name,
            "version": self.version,
            "mode": self.mode,
            "theories_count": len(self.theory_patterns),
            "supports_batch_embeddings": self.supports_batch_embeddings,
        }

        if self.bioformer:
            info["embedding_model"] = self.bioformer.get_model_info()

        if self.mode == "gpt4o":
            info["llm_model_name"] = self.llm_model_name
            info["llm_temperature"] = self.llm_temperature

        return info

    @staticmethod
    def _get_fallback_patterns() -> Dict[str, List[str]]:
        return {
            "Mitochondrial Dysfunction": ["mitochondrial theory", "mitochondrial dysfunction"],
            "Cellular Senescence": ["cellular senescence", "senescent cells"],
            "Telomere Attrition": ["telomere attrition", "telomere shortening"],
        }
