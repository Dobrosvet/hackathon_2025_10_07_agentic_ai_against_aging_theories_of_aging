"""
Questions and Criterias Classifier v2
Supports multiple approaches: NLI, Sentence-BERT, Cross-Encoder, and traditional embeddings
"""

import logging
import os
import importlib.util
import random
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"


def _load_env_from_file():
    if not ENV_PATH.exists():
        return

    try:
        content = ENV_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Failed to read .env file at {ENV_PATH}: {exc}")
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class QuestionsClassifierV2:
    """
    Enhanced classifier supporting multiple approaches:
    - NLI (Natural Language Inference) for zero-shot classification
    - Sentence-BERT for proper semantic embeddings
    - Cross-Encoder for pairwise scoring
    - Traditional embeddings (baseline)
    """

    def __init__(
        self,
        config_path: str = None,
        model_name: str = None,
        approach: str = "auto",
        use_gpu: bool = True,
        quantize: bool = False,
        requires_auth: Optional[bool] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize classifier

        Args:
            config_path: Path to config.yaml
            model_name: Model name (if None, uses config)
            approach: Classification approach (auto, nli, sentence_bert, cross_encoder, embedding)
            use_gpu: Use GPU if available
            quantize: Use quantization
            requires_auth: Explicit flag that model requires Hugging Face authentication
            random_seed: Seed for deterministic random predictions (used in random approach)
        """
        # Load configuration
        _load_env_from_file()

        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.questions = self.config.get("questions", {})
        self.criteria = self.config.get("criteria", {})
        self.classifier_config = self.config.get("classifier", {})
        self.huggingface_config = self.config.get("huggingface", {})
        self._hf_env_var = self.huggingface_config.get("token_env_var")
        self._progress_enabled = bool(self.classifier_config.get("show_progress", True))

        # Model settings
        self.model_name = model_name or self.classifier_config.get("model_name", "bioformers/bioformer-8L")
        self.approach = approach
        self._use_gpu = use_gpu
        self._quantize = quantize
        self.requires_auth = requires_auth if requires_auth is not None else self._model_requires_auth()
        self._hf_token = self._resolve_hf_token()
        self._hf_transformers_kwargs = self._build_transformers_kwargs()
        self.random_seed = random_seed if random_seed is not None else self.classifier_config.get("random_seed", 1337)

        # Models (lazy loading)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialized = False

        # Approach-specific attributes
        self.nli_pipeline = None
        self.sbert_model = None
        self.cross_encoder = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.generation_config = None
        self._checked_model_access = False
        self._random = random.Random(self.random_seed)
        self._paper_counter = 0

        logger.info(f"QuestionsClassifierV2 created: model={self.model_name}, approach={self.approach}")

    def _model_requires_auth(self) -> bool:
        required_models = self.huggingface_config.get("require_token_for_models", []) or []
        model_name_lower = (self.model_name or "").lower()
        return any(model_name_lower == item.lower() for item in required_models)

    def _resolve_hf_token(self) -> Optional[str]:
        token_value: Optional[str] = None

        if self._hf_env_var:
            raw_value = os.environ.get(self._hf_env_var)
            if raw_value:
                token_value = raw_value.strip() or None

        if self.requires_auth and not token_value:
            env_hint = self._hf_env_var or "HF_TOKEN"
            raise RuntimeError(
                f"Hugging Face token is required to load model '{self.model_name}'. "
                f"Set the environment variable {env_hint} before running the service."
            )

        if not token_value and self.huggingface_config.get("fail_if_missing", False):
            env_hint = self._hf_env_var or "HF_TOKEN"
            raise RuntimeError(
                f"Hugging Face token is not set (expected in {env_hint}). "
                "Update your environment or disable private models."
            )

        if token_value:
            logger.debug("Hugging Face token resolved from environment variable.")
        else:
            logger.debug("No Hugging Face token detected; using public model loading.")

        return token_value

    def _build_transformers_kwargs(self) -> Dict[str, Any]:
        if not self._hf_token:
            return {}
        return {"token": self._hf_token}

    def _ensure_model_access(self):
        if self._checked_model_access:
            return

        if not self.model_name or Path(self.model_name).exists():
            self._checked_model_access = True
            return

        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
            from requests.exceptions import HTTPError
        except ImportError:
            logger.warning("huggingface_hub not available; skipping remote model access validation.")
            self._checked_model_access = True
            return

        api = HfApi()
        try:
            api.model_info(self.model_name, token=self._hf_token)
        except (GatedRepoError, RepositoryNotFoundError, HTTPError, OSError) as err:
            env_hint = self._hf_env_var or "HF_TOKEN"
            raise RuntimeError(
                f"Hugging Face access denied for '{self.model_name}'. "
                f"Verify that the token ({env_hint}) has permissions or request access at "
                f"https://huggingface.co/{self.model_name}."
            ) from err

        self._checked_model_access = True

    def _detect_approach(self, model_name: str) -> str:
        """
        Auto-detect best approach based on model name

        Args:
            model_name: Model name

        Returns:
            Approach name
        """
        model_lower = model_name.lower()

        # LLM Generation models (3-4B autoregressive models)
        if any(keyword in model_lower for keyword in ['llama', 'mistral', 'phi', 'qwen', 'gemma', 'biomedlm', 'ministral']):
            return "llm_generation"

        # NLI models
        if any(keyword in model_lower for keyword in ['mnli', 'nli', 'bart-large-mnli', 'deberta']):
            return "nli"

        # Sentence-BERT models
        if any(keyword in model_lower for keyword in ['sentence-transformers', 'sbert', 'specter', 's-pubmedbert']):
            return "sentence_bert"

        # Cross-encoder models
        if 'cross-encoder' in model_lower:
            return "cross_encoder"

        # Random baseline
        if 'random' in model_lower:
            return "random"

        # Default to embedding
        return "embedding"

    def _initialize(self):
        """Lazy initialization of model"""
        if self._initialized:
            return

        try:
            import torch

            # Detect approach if auto
            if self.approach == "auto":
                self.approach = self._detect_approach(self.model_name)
                logger.info(f"Auto-detected approach: {self.approach}")

            # Determine device
            if self._use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            needs_remote_access = self.approach in {
                "nli",
                "sentence_bert",
                "cross_encoder",
                "embedding",
                "llm_generation",
            }
            if needs_remote_access:
                self._ensure_model_access()

            # Initialize based on approach
            if self.approach == "nli":
                self._initialize_nli()
            elif self.approach == "sentence_bert":
                self._initialize_sentence_bert()
            elif self.approach == "cross_encoder":
                self._initialize_cross_encoder()
            elif self.approach == "embedding":
                self._initialize_embedding()
            elif self.approach == "llm_generation":
                self._initialize_llm_generation()
            elif self.approach == "random":
                self._initialize_random()
            else:
                raise ValueError(f"Unknown approach: {self.approach}")

            self._initialized = True
            logger.info(f"Model initialized successfully with {self.approach} approach")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _initialize_nli(self):
        """Initialize NLI (zero-shot classification) model"""
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading NLI model: {self.model_name}...")

        # Create zero-shot classification pipeline
        device_idx = 0 if self.device.type == "cuda" else -1

        # DeBERTa models require slow tokenizer
        if "deberta" in self.model_name.lower():
            logger.info("Detected DeBERTa model, using slow tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                **self._hf_transformers_kwargs
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **self._hf_transformers_kwargs
            )

            pipeline_kwargs = {
                "task": "zero-shot-classification",
                "model": model,
                "tokenizer": tokenizer,
                "device": device_idx,
            }
            self.nli_pipeline = pipeline(**pipeline_kwargs)
        else:
            # Other models can use fast tokenizer
            pipeline_kwargs = {
                "task": "zero-shot-classification",
                "model": self.model_name,
                "device": device_idx,
            }
            if self._hf_token:
                pipeline_kwargs["token"] = self._hf_token
            self.nli_pipeline = pipeline(**pipeline_kwargs)

        logger.info("NLI pipeline loaded successfully")

    def _initialize_sentence_bert(self):
        """Initialize Sentence-BERT model"""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading Sentence-BERT model: {self.model_name}...")

        sbert_kwargs = {}
        if self._hf_token:
            sbert_kwargs["token"] = self._hf_token

        self.sbert_model = SentenceTransformer(self.model_name, **sbert_kwargs)

        # Move to device
        if self.device.type == "cuda":
            self.sbert_model = self.sbert_model.to(self.device)

        logger.info("Sentence-BERT model loaded successfully")

    def _initialize_cross_encoder(self):
        """Initialize Cross-Encoder model"""
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading Cross-Encoder model: {self.model_name}...")

        cross_kwargs = {}
        if self._hf_token:
            cross_kwargs["token"] = self._hf_token

        self.cross_encoder = CrossEncoder(self.model_name, **cross_kwargs)

        logger.info("Cross-Encoder loaded successfully")

    def _initialize_embedding(self):
        """Initialize traditional embedding model"""
        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info(f"Loading embedding model: {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **self._hf_transformers_kwargs
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            **self._hf_transformers_kwargs
        )

        # Quantization
        if self._quantize and self.device.type == "cuda":
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Applied int8 quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        self.model.to(self.device)
        self.model.eval()

        logger.info("Embedding model loaded successfully")

    def _initialize_llm_generation(self):
        """Initialize LLM for generation-based classification"""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
        import torch

        logger.info(f"Loading LLM generation model: {self.model_name}...")

        quantization_config = None

        if self._quantize:
            if self.device.type != "cuda":
                logger.info(
                    "Quantization requested but running on CPU; disabling quantization and using float32 weights."
                )
            else:
                if importlib.util.find_spec("bitsandbytes") is None:
                    raise RuntimeError(
                        "bitsandbytes is required for 8-bit quantization but is not installed. "
                        "Install it inside the poetry environment before running the classifier."
                    )
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                logger.info("Using 8-bit quantization via bitsandbytes")

        # Load tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **self._hf_transformers_kwargs
        )

        # Set padding token if not present
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Load model with quantization
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self._hf_transformers_kwargs:
            model_kwargs.update(self._hf_transformers_kwargs)

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            if self.device.type == "cuda":
                device_index = self.device.index if self.device.index is not None else 0
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = {"": device_index}
                model_kwargs["max_memory"] = {
                    f"cuda:{device_index}": "7GiB",
                    "cpu": "8GiB"
                }
            else:
                model_kwargs["torch_dtype"] = torch.float32

        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.llm_model.to(self.device)

            self.llm_model.eval()

            # Configure generation
            self.generation_config = GenerationConfig(
                max_new_tokens=5,  # Only need "Yes" or "No"
                temperature=0.0,   # Deterministic
                do_sample=False,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )

            logger.info("LLM generation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def _initialize_random(self):
        """Initialize random baseline (no external resources)."""
        logger.info(f"Using deterministic random baseline with seed {self.random_seed}")

    def _classify_question_nli(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using NLI (zero-shot classification)

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate text to avoid memory issues
        max_text_length = 2000
        text_truncated = text[:max_text_length]

        if question_type == "binary":
            # Binary classification with NLI
            # Hypothesis: "This paper {question}"
            hypothesis = f"suggests {question_text.lower()}"

            result = self.nli_pipeline(
                text_truncated,
                candidate_labels=["yes", "no"],
                hypothesis_template=f"This paper {{}}"
            )

            # Get scores
            labels = result['labels']
            scores = result['scores']

            # Find yes/no scores
            yes_idx = labels.index('yes') if 'yes' in labels else 0
            yes_score = scores[yes_idx]

            answer = yes_score > 0.5
            confidence = yes_score if answer else (1 - yes_score)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "scores": {
                    "yes": round(yes_score, 3),
                    "no": round(1 - yes_score, 3)
                },
                "fragments": []  # TODO: implement fragment extraction
            }

        elif question_type == "multiclass":
            # Multiclass with NLI
            options = question_config.get("options", [])

            result = self.nli_pipeline(
                text_truncated,
                candidate_labels=options,
                hypothesis_template=f"This paper suggests {{}}"
            )

            best_label = result['labels'][0]
            best_score = result['scores'][0]

            return {
                "question_id": question_id,
                "answer": best_label,
                "confidence": round(best_score, 3),
                "all_options": [
                    {"option": label, "score": round(score, 3)}
                    for label, score in zip(result['labels'], result['scores'])
                ],
                "fragments": []
            }

    def _classify_question_sentence_bert(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using Sentence-BERT (semantic similarity)

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        from sentence_transformers import util

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate text
        max_text_length = 5000
        text_truncated = text[:max_text_length]

        # Get text embedding
        text_emb = self.sbert_model.encode(text_truncated, convert_to_tensor=True)

        if question_type == "binary":
            # Create contextualized prompts for yes/no
            yes_prompt = f"A scientific paper about aging that {question_text.lower().replace('does it', 'does').replace('?', '')}"
            no_prompt = f"A scientific paper about aging that does NOT {question_text.lower().replace('does it', '').replace('?', '')}"

            yes_emb = self.sbert_model.encode(yes_prompt, convert_to_tensor=True)
            no_emb = self.sbert_model.encode(no_prompt, convert_to_tensor=True)

            # Calculate similarities
            yes_sim = util.cos_sim(text_emb, yes_emb).item()
            no_sim = util.cos_sim(text_emb, no_emb).item()

            # Normalize scores to [0, 1]
            yes_score = (yes_sim + 1) / 2
            no_score = (no_sim + 1) / 2

            # Decision based on relative scores
            answer = yes_score > no_score
            confidence = max(yes_score, no_score)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "scores": {
                    "yes": round(yes_score, 3),
                    "no": round(no_score, 3),
                    "yes_sim": round(yes_sim, 3),
                    "no_sim": round(no_sim, 3)
                },
                "fragments": []
            }

        elif question_type == "multiclass":
            # Multiclass with sentence embeddings
            options = question_config.get("options", [])

            option_prompts = [
                f"A scientific paper about aging where {question_text.lower().replace('?', '')}: {opt}"
                for opt in options
            ]

            option_embs = self.sbert_model.encode(option_prompts, convert_to_tensor=True)

            # Calculate similarities
            similarities = util.cos_sim(text_emb, option_embs)[0]

            # Normalize
            scores = [(sim.item() + 1) / 2 for sim in similarities]

            # Get best option
            best_idx = np.argmax(scores)
            best_option = options[best_idx]
            best_score = scores[best_idx]

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_score, 3),
                "all_options": [
                    {"option": opt, "score": round(score, 3)}
                    for opt, score in zip(options, scores)
                ],
                "fragments": []
            }

    def _classify_question_cross_encoder(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using Cross-Encoder

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate
        max_text_length = 2000
        text_truncated = text[:max_text_length]

        if question_type == "binary":
            # Score with cross-encoder
            pairs = [
                [text_truncated, f"This paper suggests: {question_text}"],
                [text_truncated, f"This paper does NOT suggest: {question_text}"]
            ]

            scores = self.cross_encoder.predict(pairs)
            yes_score = (scores[0] + 1) / 2  # Normalize
            no_score = (scores[1] + 1) / 2

            answer = yes_score > no_score
            confidence = max(yes_score, no_score)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "scores": {
                    "yes": round(yes_score, 3),
                    "no": round(no_score, 3)
                },
                "fragments": []
            }

        elif question_type == "multiclass":
            options = question_config.get("options", [])

            pairs = [
                [text_truncated, f"This paper suggests: {question_text} Answer: {opt}"]
                for opt in options
            ]

            scores = self.cross_encoder.predict(pairs)
            normalized_scores = [(s + 1) / 2 for s in scores]

            best_idx = np.argmax(normalized_scores)
            best_option = options[best_idx]
            best_score = normalized_scores[best_idx]

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_score, 3),
                "all_options": [
                    {"option": opt, "score": round(score, 3)}
                    for opt, score in zip(options, normalized_scores)
                ],
                "fragments": []
            }

    def _classify_question_embedding(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using traditional embeddings (baseline approach)
        """
        self._initialize()

        import torch

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Get text embedding
        def get_embedding(txt: str) -> np.ndarray:
            inputs = self.tokenizer(
                txt,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy()[0]

        text_emb = get_embedding(text[:2000])

        if question_type == "binary":
            yes_prompt = f"Question: {question_text}\n\nAnswer: Yes"
            no_prompt = f"Question: {question_text}\n\nAnswer: No"

            yes_emb = get_embedding(yes_prompt)
            no_emb = get_embedding(no_prompt)

            yes_sim = float(np.dot(text_emb, yes_emb))
            no_sim = float(np.dot(text_emb, no_emb))

            answer = yes_sim > no_sim
            confidence = max(yes_sim, no_sim)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "fragments": []
            }

        elif question_type == "multiclass":
            options = question_config.get("options", [])
            option_embeddings = []

            for option in options:
                option_prompt = f"Question: {question_text}\n\nAnswer: {option}"
                option_emb = get_embedding(option_prompt)
                similarity = float(np.dot(text_emb, option_emb))
                option_embeddings.append((option, similarity))

            best_option, best_similarity = max(option_embeddings, key=lambda x: x[1])

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_similarity, 3),
                "all_options": [
                    {"option": opt, "similarity": round(sim, 3)}
                    for opt, sim in option_embeddings
                ],
                "fragments": []
            }

    def _classify_question_random(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate deterministic random answers"""
        self._initialize()

        question_type = question_config.get("type", "binary")

        if question_type == "binary":
            answer = self._random.choice([True, False])
            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": 0.5,
                "scores": {
                    "yes": 0.5,
                    "no": 0.5
                },
                "fragments": []
            }
        elif question_type == "multiclass":
            options = question_config.get("options", [])
            if not options:
                return {
                    "question_id": question_id,
                    "answer": None,
                    "confidence": 0.0,
                    "all_options": [],
                    "fragments": []
                }
            answer = self._random.choice(options)
            uniform_score = round(1.0 / len(options), 3)
            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": uniform_score,
                "all_options": [
                    {"option": opt, "score": uniform_score}
                    for opt in options
                ],
                "fragments": []
            }
        else:
            answer = self._random.choice([True, False])
            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": 0.5,
                "fragments": []
            }

    def _parse_llm_response(self, response_text: str) -> Tuple[Optional[bool], float]:
        """
        Parse LLM response to extract Yes/No answer

        Args:
            response_text: Generated text from LLM

        Returns:
            Tuple of (answer: bool or None, confidence: float)
        """
        response_lower = response_text.lower().strip()

        # Try to extract Yes/No from response
        # Look for exact matches first
        if response_lower == "yes":
            return True, 1.0
        elif response_lower == "no":
            return False, 1.0

        # Look for Yes/No at the beginning
        if response_lower.startswith("yes"):
            return True, 0.9
        elif response_lower.startswith("no"):
            return False, 0.9

        # Look for Yes/No anywhere in response
        if "yes" in response_lower and "no" not in response_lower:
            return True, 0.7
        elif "no" in response_lower and "yes" not in response_lower:
            return False, 0.7

        # Could not determine answer
        logger.warning(f"Could not parse LLM response: {response_text}")
        return None, 0.0

    def _classify_question_llm(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using LLM generation

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        import torch

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate text to fit context
        max_text_length = 2000
        text_truncated = text[:max_text_length]

        if question_type == "binary":
            # Create prompt for binary question
            prompt = f"""You are a scientific paper analyzer specialized in aging research.

Context: {text_truncated}

Question: {question_text}

Answer with ONLY "Yes" or "No": """

            # Tokenize
            tokenized = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            target_device = self.device
            if hasattr(self.llm_model, "hf_device_map") and self.llm_model.hf_device_map:
                first_device = next(
                    (device for device in self.llm_model.hf_device_map.values() if isinstance(device, str) and device not in {"disk"}),  # type: ignore[attr-defined]
                    "cpu"
                )
                target_device = torch.device(first_device) if isinstance(first_device, str) else self.device

            inputs = {k: v.to(target_device) for k, v in tokenized.items()}

            # Generate
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode response
            generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
            response_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse response
            answer, confidence = self._parse_llm_response(response_text)

            # Fallback to default if parsing failed
            if answer is None:
                logger.warning(f"LLM returned unparseable response for {question_id}: '{response_text}', defaulting to No")
                answer = False
                confidence = 0.1

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "llm_response": response_text,
                "fragments": []
            }

        elif question_type == "multiclass":
            options = question_config.get("options", [])

            # Create prompt for multiclass question
            options_str = ", ".join(options)
            prompt = f"""You are a scientific paper analyzer specialized in aging research.

Context: {text_truncated}

Question: {question_text}
Options: {options_str}

Answer with ONLY one of the options: """

            # Tokenize
            tokenized = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            target_device = self.device
            if hasattr(self.llm_model, "hf_device_map") and self.llm_model.hf_device_map:
                first_device = next(
                    (device for device in self.llm_model.hf_device_map.values() if isinstance(device, str) and device not in {"disk"}),  # type: ignore[attr-defined]
                    "cpu"
                )
                target_device = torch.device(first_device) if isinstance(first_device, str) else self.device

            inputs = {k: v.to(target_device) for k, v in tokenized.items()}

            # Generate
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

            # Decode response
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            response_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Find best matching option
            response_lower = response_text.lower().strip()
            best_option = None
            best_score = 0.0

            for option in options:
                if option.lower() in response_lower:
                    # Exact match
                    best_option = option
                    best_score = 1.0
                    break
                elif response_lower in option.lower():
                    # Partial match
                    if best_score < 0.7:
                        best_option = option
                        best_score = 0.7

            # Fallback to first option if no match
            if best_option is None:
                logger.warning(f"Could not match LLM response '{response_text}' to any option, using first option")
                best_option = options[0]
                best_score = 0.1

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_score, 3),
                "llm_response": response_text,
                "all_options": [{"option": opt, "score": 1.0 if opt == best_option else 0.0} for opt in options],
                "fragments": []
            }

    def classify_question(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify text by question using the selected approach

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        if self.approach == "nli":
            return self._classify_question_nli(text, question_id, question_config)
        elif self.approach == "sentence_bert":
            return self._classify_question_sentence_bert(text, question_id, question_config)
        elif self.approach == "cross_encoder":
            return self._classify_question_cross_encoder(text, question_id, question_config)
        elif self.approach == "embedding":
            return self._classify_question_embedding(text, question_id, question_config)
        elif self.approach == "llm_generation":
            return self._classify_question_llm(text, question_id, question_config)
        elif self.approach == "random":
            return self._classify_question_random(text, question_id, question_config)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def classify_paper(self, full_text: str) -> Dict[str, Any]:
        """
        Classify paper by all questions and criteria

        Args:
            full_text: Full paper text

        Returns:
            Classification results
        """
        logger.info(f"Starting paper classification with {self.approach} approach...")

        if self.approach == "random":
            paper_seed = (self.random_seed + self._paper_counter) & 0xFFFFFFFF
            self._paper_counter += 1
            self._random = random.Random(paper_seed)

        results = {
            "questions": {},
            "criteria": {},
            "timestamp": None,
            "approach": self.approach,
            "model_name": self.model_name
        }

        # Classify questions
        question_items = list(self.questions.items())
        question_iterator = tqdm(
            question_items,
            desc="Questions",
            leave=False,
            disable=not self._progress_enabled
        )
        for q_id, q_config in question_iterator:
            logger.info(f"Classifying {q_id}...")
            result = self.classify_question(full_text, q_id, q_config)
            results["questions"][q_id] = result

        # Classify criteria
        criteria_items = list(self.criteria.items())
        criteria_iterator = tqdm(
            criteria_items,
            desc="Criteria",
            leave=False,
            disable=not self._progress_enabled
        )
        for c_id, c_config in criteria_iterator:
            logger.info(f"Classifying {c_id}...")
            result = self.classify_question(full_text, c_id, c_config)
            results["criteria"][c_id] = result

        from datetime import datetime
        results["timestamp"] = datetime.now().isoformat()

        logger.info("Classification complete")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        self._initialize()

        import torch

        info = {
            "model_name": self.model_name,
            "approach": self.approach,
            "device": str(self.device),
            "initialized": self._initialized,
            "num_questions": len(self.questions),
            "num_criteria": len(self.criteria)
        }

        if self._initialized and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)

        return info


if __name__ == "__main__":
    # Test different approaches
    logging.basicConfig(level=logging.INFO)

    test_text = """
    Aging is characterized by progressive decline in cellular function and increased
    susceptibility to disease. Telomere shortening has been proposed as a biomarker
    of biological aging. Studies have shown that telomere length is inversely correlated
    with chronological age and mortality risk across different species.
    """

    # Test NLI approach
    print("\n=== Testing NLI Approach ===")
    try:
        classifier_nli = QuestionsClassifierV2(
            model_name="facebook/bart-large-mnli",
            approach="nli",
            use_gpu=False
        )
        results = classifier_nli.classify_paper(test_text)
        print(f"Q1 result: {results['questions']['Q1']}")
    except Exception as e:
        print(f"NLI test failed: {e}")

    # Test Sentence-BERT approach
    print("\n=== Testing Sentence-BERT Approach ===")
    try:
        classifier_sbert = QuestionsClassifierV2(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            approach="sentence_bert",
            use_gpu=False
        )
        results = classifier_sbert.classify_paper(test_text)
        print(f"Q1 result: {results['questions']['Q1']}")
    except Exception as e:
        print(f"Sentence-BERT test failed: {e}")
