"""
Questions and Criterias Classifier v2
Supports multiple approaches: NLI, Sentence-BERT, Cross-Encoder, and traditional embeddings
"""

import logging
import os
import importlib.util
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import httpx
try:
    import numpy as np
except ImportError:
    np = None
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

    _openrouter_global_state = {"last_call": None, "extra_delay": 0.0}
    _openrouter_state_lock = threading.RLock()
    _google_genai_global_state = {"last_call": None, "extra_delay": 0.0, "call_times": deque()}
    _google_genai_state_lock = threading.RLock()
    _anthropic_global_state = {"last_call": None, "extra_delay": 0.0, "call_times": deque()}
    _anthropic_state_lock = threading.RLock()
    _openai_global_state = {"last_call": None, "extra_delay": 0.0, "call_times": deque()}
    _openai_state_lock = threading.RLock()

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
        self.model_name = model_name or self.classifier_config.get("model_name", "pritamdeka/S-PubMedBert-MS-MARCO")
        self.approach = approach
        self._use_gpu = use_gpu
        self._quantize = quantize
        self.requires_auth = requires_auth if requires_auth is not None else self._model_requires_auth()
        self.provider = str(self.classifier_config.get("provider", "huggingface")).lower()
        if self.provider != "huggingface":
            if self.requires_auth:
                logger.debug(
                    "Provider '%s' does not use Hugging Face token; skipping token requirement.",
                    self.provider,
                )
            self.requires_auth = False
            self._hf_token = None
        else:
            self._hf_token = self._resolve_hf_token()
        self._hf_transformers_kwargs = self._build_transformers_kwargs()
        self.api_config = self.classifier_config.get("api", {}) or {}
        self._generation_overrides = self.classifier_config.get("generation_config", {}) or {}

        approaches_config = self.config.get("approaches", {})
        self.llm_generation_templates = approaches_config.get("llm_generation", {}) if isinstance(approaches_config, dict) else {}
        self.openrouter_config = self.config.get("openrouter", {}) or {}
        self._openrouter_env_var = self.openrouter_config.get("token_env_var", "OPENROUTER_API_KEY")
        self._openrouter_api_key: Optional[str] = None
        self._openrouter_client: Optional[httpx.Client] = None
        self._openrouter_base_url = (
            self.api_config.get("base_url")
            or self.openrouter_config.get("base_url")
            or "https://openrouter.ai/api/v1"
        )
        self._openrouter_endpoint = self.api_config.get("endpoint") or self.openrouter_config.get("endpoint") or "/chat/completions"
        timeout_value = self.api_config.get("timeout") or self.openrouter_config.get("timeout") or 60.0
        self._openrouter_timeout = float(timeout_value)
        if isinstance(self._openrouter_base_url, str):
            self._openrouter_base_url = self._openrouter_base_url.rstrip("/")
        if isinstance(self._openrouter_endpoint, str):
            self._openrouter_endpoint = "/" + self._openrouter_endpoint.lstrip("/")
        self._openrouter_max_retries = int(self.api_config.get("max_retries") or self.openrouter_config.get("max_retries") or 5)
        self._openrouter_backoff_base = float(self.api_config.get("backoff_base") or self.openrouter_config.get("backoff_base") or 1.5)
        self._openrouter_backoff_cap = float(self.api_config.get("backoff_cap") or self.openrouter_config.get("backoff_cap") or 30.0)
        self._openrouter_backoff_jitter = float(self.api_config.get("backoff_jitter") or self.openrouter_config.get("backoff_jitter") or 0.5)
        self._openrouter_retry_statuses = {429, 500, 502, 503, 504}
        self._openrouter_throttle_seconds = float(self.api_config.get("throttle_seconds") or self.openrouter_config.get("throttle_seconds") or 1.0)

        self.google_genai_config = self.config.get("google_genai", {}) or {}
        self._google_genai_env_var = self.google_genai_config.get("token_env_var", "GOOGLE_GENAI_API_KEY")
        self._google_genai_api_key: Optional[str] = None
        self._google_genai_client: Optional[Any] = None  # late import from google.genai
        
        # Anthropic configuration
        self.anthropic_config = self.config.get("anthropic", {}) or {}
        self._anthropic_env_var = self.anthropic_config.get("token_env_var", "ANTHROPIC_API_KEY")
        self._anthropic_api_key: Optional[str] = None
        self._anthropic_client: Optional[Any] = None  # late import from anthropic
        self._anthropic_throttle_seconds = float(self.anthropic_config.get("throttle_seconds", 1.0))
        self._anthropic_max_retries = int(self.anthropic_config.get("max_retries", 6))
        self._anthropic_backoff_base = float(self.anthropic_config.get("backoff_base", 2.0))
        self._anthropic_backoff_cap = float(self.anthropic_config.get("backoff_cap", 180.0))
        self._anthropic_backoff_jitter = float(self.anthropic_config.get("backoff_jitter", 0.5))
        self._anthropic_retry_statuses = self.anthropic_config.get("retry_statuses", [429, 500, 502, 503, 504])
        
        # OpenAI configuration
        self.openai_config = self.config.get("openai", {}) or {}
        self._openai_env_var = self.openai_config.get("token_env_var", "OPENAI_API_KEY")
        self._openai_api_key: Optional[str] = None
        self._openai_client: Optional[Any] = None  # late import from openai
        self._openai_throttle_seconds = float(self.openai_config.get("throttle_seconds", 1.0))
        self._openai_max_retries = int(self.openai_config.get("max_retries", 6))
        self._openai_backoff_base = float(self.openai_config.get("backoff_base", 2.0))
        self._openai_backoff_cap = float(self.openai_config.get("backoff_cap", 180.0))
        self._openai_backoff_jitter = float(self.openai_config.get("backoff_jitter", 0.5))
        self._openai_retry_statuses = self.openai_config.get("retry_statuses", [429, 500, 502, 503, 504])
        api_rate_limit = self.api_config.get("rate_limit_per_minute")
        if api_rate_limit is None:
            api_rate_limit = self.google_genai_config.get("rate_limit_per_minute")
        self._google_genai_rate_limit_per_minute = int(api_rate_limit or 60)
        throttle_value = self.api_config.get("throttle_seconds")
        if throttle_value is None:
            throttle_value = self.google_genai_config.get("throttle_seconds")
        self._google_genai_throttle_seconds = float(throttle_value or 1.0)
        max_retries_value = self.api_config.get("max_retries")
        if max_retries_value is None:
            max_retries_value = self.google_genai_config.get("max_retries")
        self._google_genai_max_retries = int(max_retries_value or 5)
        backoff_base_value = self.api_config.get("backoff_base")
        if backoff_base_value is None:
            backoff_base_value = self.google_genai_config.get("backoff_base")
        self._google_genai_backoff_base = float(backoff_base_value or 1.5)
        backoff_cap_value = self.api_config.get("backoff_cap")
        if backoff_cap_value is None:
            backoff_cap_value = self.google_genai_config.get("backoff_cap")
        self._google_genai_backoff_cap = float(backoff_cap_value or 60.0)
        backoff_jitter_value = self.api_config.get("backoff_jitter")
        if backoff_jitter_value is None:
            backoff_jitter_value = self.google_genai_config.get("backoff_jitter")
        self._google_genai_backoff_jitter = float(backoff_jitter_value or 0.5)
        retry_statuses_value = self.api_config.get("retry_statuses")
        if retry_statuses_value is None:
            retry_statuses_value = self.google_genai_config.get("retry_statuses")
        if retry_statuses_value:
            self._google_genai_retry_statuses = {int(status) for status in retry_statuses_value}
        else:
            self._google_genai_retry_statuses = {429, 500, 502, 503, 504}
        http_options_value = self.api_config.get("http_options")
        if http_options_value is None:
            http_options_value = self.google_genai_config.get("http_options")
        self._google_genai_http_options = http_options_value if isinstance(http_options_value, dict) else {}
        vertexai_value = self.api_config.get("vertexai")
        if vertexai_value is None:
            vertexai_value = self.google_genai_config.get("vertexai")
        self._google_genai_vertexai = bool(vertexai_value) if vertexai_value is not None else None
        self._google_genai_project = self.api_config.get("project") or self.google_genai_config.get("project")
        self._google_genai_location = self.api_config.get("location") or self.google_genai_config.get("location")
        self._google_genai_safety_settings = self.api_config.get("safety_settings") or self.google_genai_config.get("safety_settings")
        self._google_genai_response_mime_type = self.api_config.get("response_mime_type") or self.google_genai_config.get("response_mime_type")
        generation_defaults = self.google_genai_config.get("generation_config")
        self._google_genai_generation_defaults = generation_defaults if isinstance(generation_defaults, dict) else {}
        self._google_genai_rate_limit_window = 60.0

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

    @classmethod
    def reset_openrouter_state(cls):
        with cls._openrouter_state_lock:
            cls._openrouter_global_state["last_call"] = None
            cls._openrouter_global_state["extra_delay"] = 0.0

    @classmethod
    def reset_google_genai_state(cls):
        with cls._google_genai_state_lock:
            state = cls._google_genai_global_state
            state["last_call"] = None
            state["extra_delay"] = 0.0
            call_times = state.get("call_times")
            if isinstance(call_times, deque):
                call_times.clear()
            else:
                state["call_times"] = deque()

    def _openrouter_wait_for_slot(self) -> None:
        throttle = max(self._openrouter_throttle_seconds, 0.0)
        with self._openrouter_state_lock:
            state = self._openrouter_global_state
            last_call = state["last_call"]
            extra = state["extra_delay"]

        wait_target = throttle + max(0.0, extra)
        if wait_target > 0.0 and last_call is not None:
            elapsed = time.perf_counter() - last_call
            if elapsed < wait_target:
                time.sleep(wait_target - elapsed)

        with self._openrouter_state_lock:
            self._openrouter_global_state["last_call"] = time.perf_counter()

    def _openrouter_record_call(self) -> None:
        with self._openrouter_state_lock:
            self._openrouter_global_state["last_call"] = time.perf_counter()

    def _openrouter_increase_delay(self) -> None:
        increment = max(0.5, self._openrouter_throttle_seconds * 0.5, self._openrouter_backoff_base * 0.25)
        with self._openrouter_state_lock:
            state = self._openrouter_global_state
            extra = max(0.0, state["extra_delay"])
            new_extra = min(self._openrouter_backoff_cap, extra * 1.4 + increment)
            state["extra_delay"] = new_extra

    def _openrouter_decrease_delay(self) -> None:
        with self._openrouter_state_lock:
            state = self._openrouter_global_state
            extra = state["extra_delay"]
            if extra <= 0.0:
                return
            reduce_factor = 0.6
            decrement = max(0.2, self._openrouter_throttle_seconds * 0.25)
            new_extra = max(0.0, extra * reduce_factor - decrement)
            state["extra_delay"] = new_extra

    def _google_wait_for_slot(self) -> None:
        rate_limit = max(0, int(self._google_genai_rate_limit_per_minute))
        window = max(1.0, float(self._google_genai_rate_limit_window))
        throttle = max(self._google_genai_throttle_seconds, 0.0)

        with self._google_genai_state_lock:
            state = self._google_genai_global_state
            last_call = state["last_call"]
            extra = state["extra_delay"]
            call_times = state.get("call_times")
            if not isinstance(call_times, deque):
                call_times = deque()
                state["call_times"] = call_times
            now = time.perf_counter()
            rate_wait = 0.0
            if rate_limit > 0:
                while call_times and now - call_times[0] >= window:
                    call_times.popleft()
                if len(call_times) >= rate_limit:
                    earliest = call_times[0]
                    rate_wait = max(0.0, window - (now - earliest))
            throttle_wait = 0.0
            if last_call is not None:
                elapsed = now - last_call
                throttle_wait = max(0.0, throttle + max(0.0, extra) - elapsed)
            else:
                throttle_wait = throttle + max(0.0, extra)
            wait_time = max(rate_wait, throttle_wait, 0.0)

        if wait_time > 0.0:
            time.sleep(wait_time)

        with self._google_genai_state_lock:
            self._google_genai_global_state["last_call"] = time.perf_counter()

    def _google_record_call(self) -> None:
        with self._google_genai_state_lock:
            state = self._google_genai_global_state
            now = time.perf_counter()
            state["last_call"] = now
            call_times = state.get("call_times")
            if not isinstance(call_times, deque):
                call_times = deque()
                state["call_times"] = call_times
            window = max(1.0, float(self._google_genai_rate_limit_window))
            while call_times and now - call_times[0] >= window:
                call_times.popleft()
            if self._google_genai_rate_limit_per_minute > 0:
                call_times.append(now)

    def _google_increase_delay(self) -> None:
        increment = max(
            0.5,
            self._google_genai_throttle_seconds * 0.5,
            self._google_genai_backoff_base * 0.25,
        )
        with self._google_genai_state_lock:
            state = self._google_genai_global_state
            extra = max(0.0, state["extra_delay"])
            new_extra = min(self._google_genai_backoff_cap, extra * 1.4 + increment)
            state["extra_delay"] = new_extra

    def _google_decrease_delay(self) -> None:
        with self._google_genai_state_lock:
            state = self._google_genai_global_state
            extra = state["extra_delay"]
            if extra <= 0.0:
                return
            reduce_factor = 0.6
            decrement = max(0.2, self._google_genai_throttle_seconds * 0.25)
            new_extra = max(0.0, extra * reduce_factor - decrement)
            state["extra_delay"] = new_extra

    def _google_adjust_rate_limit_on_429(self, response: Optional[Any]) -> None:
        """
        Reduce effective Google GenAI rate limit after repeated HTTP 429 responses.

        Args:
            response: Optional HTTP response object carrying rate-limit headers.
        """
        limit_hint: Optional[int] = None
        headers: Optional[Any] = getattr(response, "headers", None)
        if headers:
            header_map = {str(key).lower(): str(value) for key, value in dict(headers).items()}
            for key in (
                "x-ratelimit-limit",
                "x-ratelimit-limit-minute",
                "x-ratelimit-limit-requests",
                "ratelimit-limit",
                "x-ac-rate-limit",
            ):
                raw_value = header_map.get(key)
                if not raw_value:
                    continue
                match = re.search(r"\d+", raw_value)
                if match:
                    try:
                        limit_hint = int(match.group())
                        break
                    except ValueError:
                        continue

        if limit_hint is not None and limit_hint <= 0:
            limit_hint = None

        fallback_limit = max(1, int(self._google_genai_rate_limit_per_minute * 0.5)) or 1
        if self._google_genai_rate_limit_per_minute > 10:
            fallback_limit = min(fallback_limit, 10)

        effective_limit = limit_hint or fallback_limit
        if (
            limit_hint is not None
            and self._google_genai_rate_limit_per_minute > 10
            and limit_hint > 10
        ):
            effective_limit = min(limit_hint, 10)

        with self._google_genai_state_lock:
            current_limit = self._google_genai_rate_limit_per_minute
            if effective_limit < current_limit:
                self._google_genai_rate_limit_per_minute = max(1, effective_limit)
                logger.warning(
                    "Google GenAI rate limit adjusted from %d to %d rpm after HTTP 429.",
                    current_limit,
                    self._google_genai_rate_limit_per_minute,
                )
    def _compute_google_backoff(self, response: Optional[Any], attempt: int) -> float:
        retry_after_seconds: Optional[float] = None
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers:
                header_value = headers.get("Retry-After") or headers.get("retry-after")
                if header_value:
                    header_value = str(header_value).strip()
                    if header_value.isdigit():
                        retry_after_seconds = float(header_value)
                    else:
                        try:
                            parsed = parsedate_to_datetime(header_value)
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            now = datetime.now(timezone.utc)
                            retry_after_seconds = max(0.0, (parsed - now).total_seconds())
                        except (TypeError, ValueError, OverflowError):
                            retry_after_seconds = None

        if retry_after_seconds is not None:
            return max(max(self._google_genai_throttle_seconds, 0.0), retry_after_seconds)

        base_delay = self._google_genai_backoff_base * (2 ** max(0, attempt - 1))
        jitter = random.random() * self._google_genai_backoff_jitter
        delay = min(self._google_genai_backoff_cap, base_delay + jitter)
        return max(max(self._google_genai_throttle_seconds, 0.0), delay)

    def _create_google_generation_config(self) -> Dict[str, Any]:
        allowed_keys = {
            "max_output_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "candidate_count",
            "stop_sequences",
            "response_mime_type",
            "safety_settings",
            "seed",
            "response_logprobs",
            "logprobs",
        }
        config: Dict[str, Any] = {}
        if isinstance(self._google_genai_generation_defaults, dict):
            for key, value in self._google_genai_generation_defaults.items():
                mapped_key = "max_output_tokens" if key in {"max_new_tokens", "max_tokens"} else key
                if mapped_key in allowed_keys and value is not None:
                    config[mapped_key] = value

        overrides = self._generation_overrides or {}
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                if value is None:
                    continue
                mapped_key = "max_output_tokens" if key in {"max_new_tokens", "max_tokens"} else key
                if mapped_key in allowed_keys:
                    config[mapped_key] = value

        config.setdefault("temperature", 0.0)
        config.setdefault("max_output_tokens", 16)

        typed_config: Dict[str, Any] = {}
        for key, value in config.items():
            if key in {"max_output_tokens", "top_k", "candidate_count", "seed"}:
                typed_config[key] = int(value)
            elif key in {"temperature", "top_p", "frequency_penalty", "presence_penalty"}:
                typed_config[key] = float(value)
            else:
                typed_config[key] = value

        return typed_config

    @staticmethod
    def _extract_google_response_text(response: Any) -> Optional[str]:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            for candidate in candidates:
                candidate_text = getattr(candidate, "text", None)
                if isinstance(candidate_text, str) and candidate_text.strip():
                    return candidate_text.strip()
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if parts and len(parts) > 0:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            return part_text.strip()

        # Fallback to __str__ if available
        if isinstance(response, dict):
            return str(response)
        if hasattr(response, "__str__"):
            representation = str(response)
            return representation.strip() if representation else None
        return None

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

        if self.provider != "huggingface":
            self._checked_model_access = True
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
            if needs_remote_access and self.provider == "huggingface":
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
                if self.provider == "openrouter":
                    self._initialize_llm_openrouter()
                elif self.provider == "google_genai":
                    self._initialize_llm_google()
                elif self.provider == "anthropic":
                    self._initialize_llm_anthropic()
                elif self.provider == "openai":
                    self._initialize_llm_openai()
                else:
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

            overrides = self._generation_overrides
            if overrides:
                if "max_new_tokens" in overrides:
                    self.generation_config.max_new_tokens = int(overrides["max_new_tokens"])
                if "temperature" in overrides:
                    self.generation_config.temperature = float(overrides["temperature"])
                if "do_sample" in overrides:
                    self.generation_config.do_sample = bool(overrides["do_sample"])
                if "top_p" in overrides:
                    self.generation_config.top_p = float(overrides["top_p"])
                if "repetition_penalty" in overrides:
                    self.generation_config.repetition_penalty = float(overrides["repetition_penalty"])

            logger.info("LLM generation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def _initialize_llm_openrouter(self):
        """Initialize OpenRouter client for generation-based classification."""
        if self._openrouter_api_key is None:
            self._openrouter_api_key = os.environ.get(self._openrouter_env_var)

        if not self._openrouter_api_key:
            raise RuntimeError(
                f"OpenRouter API key is required for model '{self.model_name}'. "
                f"Set the environment variable {self._openrouter_env_var} before running the classifier."
            )

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
        }

        user_agent = self.api_config.get("user_agent") or self.openrouter_config.get("user_agent")
        if user_agent:
            headers["User-Agent"] = user_agent

        default_headers = {}
        openrouter_headers = self.openrouter_config.get("default_headers")
        if isinstance(openrouter_headers, dict):
            default_headers.update(openrouter_headers)
        api_headers = self.api_config.get("headers")
        if isinstance(api_headers, dict):
            default_headers.update(api_headers)
        headers.update(default_headers)

        self._openrouter_client = httpx.Client(
            base_url=self._openrouter_base_url,
            headers=headers,
            timeout=self._openrouter_timeout,
        )

        max_new_tokens = self._generation_overrides.get("max_new_tokens", 5)
        temperature = self._generation_overrides.get("temperature", 0.0)
        self.generation_config = {
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }

        optional_params = ("top_p", "frequency_penalty", "presence_penalty")
        for param in optional_params:
            value = self._generation_overrides.get(param)
            if value is not None:
                self.generation_config[param] = float(value)

        logger.info(
            "OpenRouter client initialized for %s (endpoint: %s%s, timeout: %.1fs)",
            self.model_name,
            self._openrouter_base_url,
            self._openrouter_endpoint,
            self._openrouter_timeout,
        )

    def _initialize_llm_google(self) -> None:
        """Initialize Google GenAI client for generation-based classification."""
        try:
            from google import genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for Google Gemini models. "
                "Install it inside the poetry environment before running the classifier."
            ) from exc

        if self._google_genai_api_key is None:
            self._google_genai_api_key = os.environ.get(self._google_genai_env_var)

        if not self._google_genai_api_key:
            raise RuntimeError(
                f"Google GenAI API key is required for model '{self.model_name}'. "
                f"Set the environment variable {self._google_genai_env_var} before running the classifier."
            )

        client_kwargs: Dict[str, Any] = {}
        if self._google_genai_vertexai is not None:
            client_kwargs["vertexai"] = bool(self._google_genai_vertexai)
        if self._google_genai_project:
            client_kwargs["project"] = self._google_genai_project
        if self._google_genai_location:
            client_kwargs["location"] = self._google_genai_location
        if self._google_genai_http_options:
            client_kwargs["http_options"] = self._google_genai_http_options

        self._google_genai_client = genai.Client(
            api_key=self._google_genai_api_key,
            **client_kwargs,
        )

        config_dict = self._create_google_generation_config()
        if self._google_genai_response_mime_type and "response_mime_type" not in config_dict:
            config_dict["response_mime_type"] = self._google_genai_response_mime_type
        if self._google_genai_safety_settings and "safety_settings" not in config_dict:
            config_dict["safety_settings"] = self._google_genai_safety_settings
        self.generation_config = config_dict

        logger.info(
            "Google GenAI client initialized for %s (rate limit %d rpm, throttle %.2fs)",
            self.model_name,
            self._google_genai_rate_limit_per_minute,
            self._google_genai_throttle_seconds,
        )

    def _initialize_llm_anthropic(self) -> None:
        """Initialize Anthropic Claude client for generation-based classification."""
        try:
            from anthropic import Anthropic  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package is required for Anthropic Claude models. "
                "Install it inside the poetry environment before running the classifier."
            ) from exc

        if self._anthropic_api_key is None:
            self._anthropic_api_key = os.environ.get(self._anthropic_env_var)

        if not self._anthropic_api_key:
            raise RuntimeError(
                f"Anthropic API key is required for model '{self.model_name}'. "
                f"Set the environment variable {self._anthropic_env_var} before running the classifier."
            )

        self._anthropic_client = Anthropic(api_key=self._anthropic_api_key)

        logger.info(
            "Anthropic client initialized for %s (throttle %.2fs)",
            self.model_name,
            self._anthropic_throttle_seconds,
        )

    def _initialize_llm_openai(self) -> None:
        """Initialize OpenAI client for generation-based classification."""
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAI models. "
                "Install it inside the poetry environment before running the classifier."
            ) from exc

        if self._openai_api_key is None:
            self._openai_api_key = os.environ.get(self._openai_env_var)

        if not self._openai_api_key:
            raise RuntimeError(
                f"OpenAI API key is required for model '{self.model_name}'. "
                f"Set the environment variable {self._openai_env_var} before running the classifier."
            )

        self._openai_client = OpenAI(api_key=self._openai_api_key)

        overrides = self._generation_overrides if isinstance(self._generation_overrides, dict) else {}
        max_tokens_override = overrides.get("max_tokens")
        if max_tokens_override is None:
            max_tokens_override = overrides.get("max_new_tokens")
        if max_tokens_override is None:
            max_tokens_override = 16

        temperature_override = overrides.get("temperature", 0.0)

        generation_config: Dict[str, Any] = {
            "max_tokens": int(max_tokens_override),
            "temperature": float(temperature_override),
        }

        optional_numeric_params = ("top_p", "frequency_penalty", "presence_penalty")
        for param in optional_numeric_params:
            value = overrides.get(param)
            if value is None:
                continue
            generation_config[param] = float(value)

        for passthrough_key in ("logprobs", "top_logprobs", "response_format"):
            if passthrough_key in overrides and overrides[passthrough_key] is not None:
                generation_config[passthrough_key] = overrides[passthrough_key]

        self.generation_config = generation_config

        logger.info(
            "OpenAI client initialized for %s (throttle %.2fs)",
            self.model_name,
            self._openai_throttle_seconds,
        )

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

            if np is None:
                raise RuntimeError("Numpy is not available. Please install numpy: pip install numpy")
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

    @staticmethod
    def _format_guidance_section(section: Any, fallback: str) -> str:
        """
        Normalize guidance snippets from configuration into a readable block.

        Args:
            section: Raw value from configuration (string, list, or dict).
            fallback: Text to use when section is empty.

        Returns:
            Formatted guidance text.
        """
        if isinstance(section, dict):
            for key in ("rules", "points", "items", "bullets", "summary"):
                if key in section:
                    return QuestionsClassifierV2._format_guidance_section(section[key], fallback)
        if isinstance(section, (list, tuple, set)):
            lines: List[str] = []
            for item in section:
                if item is None:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                if text.startswith(("-", "*")):
                    lines.append(text)
                else:
                    lines.append(f"- {text}")
            if lines:
                return "\n".join(lines)
        elif isinstance(section, str):
            text = section.strip()
            if text:
                return text
        return fallback

    def _build_llm_prompt(
        self,
        question_type: str,
        question_text: str,
        text_truncated: str,
        options: Optional[List[str]] = None,
        question_id: Optional[str] = None
    ) -> str:
        """
        Build prompt text for LLM classification.

        Args:
            question_type: Type of question (binary or multiclass)
            question_text: Text of the question
            text_truncated: Truncated paper text
            options: Options list for multiclass questions
            question_id: Identifier of the current question/criterion

        Returns:
            Prompt string
        """
        templates = self.llm_generation_templates or {}
        system_prompt = templates.get(
            "system_prompt",
            "You are a scientific paper analyzer specialized in aging research."
        )

        general_guidance_raw = templates.get("general_guidelines")
        general_guidelines = self._format_guidance_section(
            general_guidance_raw,
            "- Base your judgement strictly on the provided context.\n"
            "- Prefer 'No' when the context does not supply compelling evidence."
        )

        question_guidance_map = templates.get("question_guidelines", {}) or {}
        question_guidance_raw = None
        if question_id and isinstance(question_guidance_map, dict):
            question_guidance_raw = question_guidance_map.get(question_id)
        question_guidelines = self._format_guidance_section(
            question_guidance_raw,
            "- Apply the general rubric for this question."
        )

        answer_format_binary = templates.get(
            "answer_format_binary",
            'Answer format: start with "Yes" or "No" followed by a short justification grounded in the context.'
        )
        answer_format_multiclass = templates.get(
            "answer_format_multiclass",
            "Answer format: start with one of the provided options exactly as written, then add a short justification."
        )

        if question_type == "binary":
            template = templates.get("prompt_template_binary")
            if template:
                prompt = template.format(
                    system_prompt=system_prompt,
                    paper_text=text_truncated,
                    question_text=question_text,
                    question_id=question_id or "",
                    general_guidelines=general_guidelines,
                    question_guidelines=question_guidelines,
                    answer_format_binary=answer_format_binary,
                )
            else:
                prompt = (
                    f"{system_prompt}\n\n"
                    "General rubric:\n"
                    f"{general_guidelines}\n\n"
                    f"Question-specific rubric for {question_id or 'this task'}:\n"
                    f"{question_guidelines}\n\n"
                    f"Context: {text_truncated}\n\n"
                    f"Question: {question_text}\n\n"
                    f"{answer_format_binary}"
                )
            return prompt.strip() + " "

        options = options or []
        template = templates.get("prompt_template_multiclass")
        if template:
            prompt = template.format(
                system_prompt=system_prompt,
                paper_text=text_truncated,
                question_text=question_text,
                options=", ".join(options),
                question_id=question_id or "",
                general_guidelines=general_guidelines,
                question_guidelines=question_guidelines,
                answer_format_multiclass=answer_format_multiclass,
            )
        else:
            options_str = ", ".join(options)
            prompt = (
                f"{system_prompt}\n\n"
                "General rubric:\n"
                f"{general_guidelines}\n\n"
                f"Question-specific rubric for {question_id or 'this task'}:\n"
                f"{question_guidelines}\n\n"
                f"Context: {text_truncated}\n\n"
                f"Question: {question_text}\n"
                f"Options: {options_str}\n\n"
                f"{answer_format_multiclass}"
            )
        return prompt.strip() + " "

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

    def _get_max_context_length(self) -> int:
        candidate_values: List[Any] = []
        templates = self.llm_generation_templates or {}
        if isinstance(self._generation_overrides, dict):
            candidate_values.append(self._generation_overrides.get("max_context_length"))
            candidate_values.append(self._generation_overrides.get("max_text_length"))
        if templates:
            candidate_values.append(templates.get("max_context_length"))
        if isinstance(self.generation_config, dict):
            candidate_values.append(self.generation_config.get("max_context_length"))
        if isinstance(self._google_genai_generation_defaults, dict):
            candidate_values.append(self._google_genai_generation_defaults.get("max_context_length"))

        for value in candidate_values:
            if value is None:
                continue
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                continue
        return 2000

    def _extract_relevant_context(
        self,
        full_text: str,
        question_id: Optional[str],
        max_length: int
    ) -> str:
        if not full_text:
            return ""

        templates = self.llm_generation_templates or {}
        keywords_map = templates.get("question_keywords") if isinstance(templates, dict) else None

        keywords: List[str] = []
        if isinstance(keywords_map, dict) and question_id:
            raw_keywords = keywords_map.get(question_id)
            if isinstance(raw_keywords, str):
                keywords = [raw_keywords]
            elif isinstance(raw_keywords, (list, tuple, set)):
                keywords = [str(item) for item in raw_keywords if item]

        normalized_text = full_text
        if max_length <= 0:
            max_length = 2000

        if not keywords:
            return normalized_text[:max_length]

        text_lower = normalized_text.lower()
        match_windows: List[Tuple[int, int, int]] = []

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if not keyword_lower:
                continue
            start = 0
            while start < len(text_lower):
                idx = text_lower.find(keyword_lower, start)
                if idx == -1:
                    break
                window_start = max(0, idx - 400)
                window_end = min(len(normalized_text), idx + len(keyword_lower) + 400)
                match_windows.append((window_start, window_end, idx))
                start = idx + len(keyword_lower)

        if not match_windows:
            return normalized_text[:max_length]

        match_windows.sort()
        filtered_windows: List[Tuple[int, int, int]] = []
        last_end = -1
        for start_idx, end_idx, keyword_idx in match_windows:
            if start_idx <= last_end + 50 and filtered_windows:
                # Extend previous window if overlapping significantly
                prev_start, prev_end, prev_keyword = filtered_windows[-1]
                new_start = min(prev_start, start_idx)
                new_end = max(prev_end, end_idx)
                # Prefer keyword that lies inside extended window; ensure we keep closest center
                if abs(keyword_idx - (new_start + new_end) // 2) < abs(prev_keyword - (new_start + new_end) // 2):
                    chosen_keyword = keyword_idx
                else:
                    chosen_keyword = prev_keyword
                filtered_windows[-1] = (new_start, new_end, chosen_keyword)
                last_end = new_end
            else:
                filtered_windows.append((start_idx, end_idx, keyword_idx))
                last_end = end_idx

        segments: List[str] = []
        used = 0

        head_slice_len = max_length // 3
        if head_slice_len > 0:
            head_segment = normalized_text[:min(len(normalized_text), head_slice_len)].strip()
            if head_segment:
                segments.append(head_segment)
                used += len(head_segment)

        for start_idx, end_idx, keyword_idx in filtered_windows:
            if used >= max_length:
                break
            segment = normalized_text[start_idx:end_idx].strip()
            if not segment:
                continue
            remaining = max_length - used
            if len(segment) > remaining:
                keyword_offset = keyword_idx - start_idx
                if keyword_offset < 0:
                    keyword_offset = 0
                if keyword_offset > len(segment):
                    keyword_offset = len(segment) // 2
                half_window = remaining // 2
                start_offset = max(0, keyword_offset - half_window)
                if start_offset + remaining > len(segment):
                    start_offset = max(0, len(segment) - remaining)
                segment = segment[start_offset:start_offset + remaining]
            segments.append(segment)
            used += len(segment)

        if not segments:
            return normalized_text[:max_length]

        # Preserve order but remove duplicates
        unique_segments = list(dict.fromkeys(seg for seg in segments if seg))
        context = "\n---\n".join(unique_segments)
        if len(context) > max_length:
            context = context[:max_length]
        return context

    def _prepare_llm_context(self, full_text: str, question_id: Optional[str]) -> str:
        max_length = self._get_max_context_length()
        context = self._extract_relevant_context(full_text, question_id, max_length)
        if context:
            return context
        return full_text[:max_length]

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

        if self.provider == "openrouter":
            return self._classify_question_llm_openrouter(text, question_id, question_config)
        if self.provider == "google_genai":
            return self._classify_question_llm_google(text, question_id, question_config)
        if self.provider == "anthropic":
            return self._classify_question_llm_anthropic(text, question_id, question_config)
        if self.provider == "openai":
            return self._classify_question_llm_openai(text, question_id, question_config)

        import torch

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        text_truncated = self._prepare_llm_context(text, question_id)

        if question_type == "binary":
            prompt = self._build_llm_prompt(
                question_type=question_type,
                question_text=question_text,
                text_truncated=text_truncated,
                question_id=question_id,
            )

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

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
            response_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

            answer, confidence = self._parse_llm_response(response_text)

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

        options = question_config.get("options", [])
        prompt = self._build_llm_prompt(
            question_type=question_type,
            question_text=question_text,
            text_truncated=text_truncated,
            options=options,
            question_id=question_id,
        )

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

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                generation_config=self.generation_config
            )

        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        response_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

        response_lower = response_text.lower().strip()
        best_option = None
        best_score = 0.0

        for option in options:
            if option.lower() in response_lower:
                best_option = option
                best_score = 1.0
                break
            if response_lower in option.lower() and best_score < 0.7:
                best_option = option
                best_score = 0.7

        if best_option is None and options:
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

    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        if self._openrouter_client is None:
            raise RuntimeError("OpenRouter client is not initialized.")

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        generation_payload: Dict[str, Any] = {}
        if isinstance(self.generation_config, dict):
            generation_payload.update(self.generation_config)
        else:
            generation_payload["max_tokens"] = getattr(self.generation_config, "max_new_tokens", 5)
            generation_payload["temperature"] = getattr(self.generation_config, "temperature", 0.0)
        payload.update(generation_payload)

        extra_payload = self.api_config.get("payload")
        if isinstance(extra_payload, dict):
            payload.update(extra_payload)

        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max(1, self._openrouter_max_retries):
            attempt += 1

            self._openrouter_wait_for_slot()

            try:
                response = self._openrouter_client.post(self._openrouter_endpoint, json=payload)
                status_code = getattr(response, "status_code", None)
                if status_code in self._openrouter_retry_statuses:
                    if attempt >= self._openrouter_max_retries:
                        response.raise_for_status()
                    delay = self._compute_openrouter_backoff(response, attempt)
                    logger.warning(
                        "OpenRouter rate limit/status %s for %s on attempt %d/%d. Retrying in %.2fs.",
                        status_code,
                        self.model_name,
                        attempt,
                        self._openrouter_max_retries,
                        delay,
                    )
                    self._openrouter_increase_delay()
                    logger.debug(
                        "OpenRouter retry-after header: %s (computed delay %.2fs)",
                        getattr(response, "headers", {}).get("Retry-After") if hasattr(response, "headers") else None,
                        delay,
                    )
                    time.sleep(delay)
                    last_error = RuntimeError(f"HTTP {status_code}")
                    continue

                response.raise_for_status()

                data = response.json()
                content = self._extract_openrouter_content(data)
                logger.debug("OpenRouter response for %s: %s", self.model_name, content)
                self._openrouter_decrease_delay()
                return content

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in self._openrouter_retry_statuses and attempt < self._openrouter_max_retries:
                    delay = self._compute_openrouter_backoff(exc.response, attempt)
                    logger.warning(
                        "OpenRouter HTTP %s for %s on attempt %d/%d. Retrying in %.2fs.",
                        status,
                        self.model_name,
                        attempt,
                        self._openrouter_max_retries,
                        delay,
                    )
                    self._openrouter_increase_delay()
                    if exc.response is not None:
                        logger.debug(
                            "OpenRouter retry-after header: %s (computed delay %.2fs)",
                            exc.response.headers.get("Retry-After"),
                            delay,
                        )
                    time.sleep(delay)
                    last_error = exc
                    continue
                self._openrouter_increase_delay()
                raise RuntimeError(f"OpenRouter request failed for '{self.model_name}': {exc}") from exc

            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                if attempt < self._openrouter_max_retries:
                    delay = self._compute_openrouter_backoff(None, attempt)
                    logger.warning(
                        "OpenRouter call error for %s (%s) on attempt %d/%d. Retrying in %.2fs.",
                        self.model_name,
                        exc,
                        attempt,
                        self._openrouter_max_retries,
                        delay,
                    )
                    self._openrouter_increase_delay()
                    time.sleep(delay)
                    last_error = exc
                    continue
                self._openrouter_increase_delay()
                raise RuntimeError(f"OpenRouter request failed for '{self.model_name}': {exc}") from exc

            finally:
                self._openrouter_record_call()

        if last_error:
            self._openrouter_increase_delay()
            raise RuntimeError(f"OpenRouter request failed for '{self.model_name}' after retries: {last_error}") from last_error
        self._openrouter_increase_delay()
        raise RuntimeError(f"OpenRouter request failed for '{self.model_name}' after retries.")

    def _call_google_genai(self, prompt: str) -> str:
        if self._google_genai_client is None:
            raise RuntimeError("Google GenAI client is not initialized.")

        try:
            from google.genai import errors as genai_errors  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for Google Gemini models. "
                "Install it inside the poetry environment before running the classifier."
            ) from exc

        max_retries = max(1, self._google_genai_max_retries)
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_retries:
            attempt += 1
            self._google_wait_for_slot()

            try:
                request_payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "contents": prompt,
                }
                config_payload: Optional[Dict[str, Any]] = None
                if isinstance(self.generation_config, dict) and self.generation_config:
                    config_payload = dict(self.generation_config)
                if self._google_genai_safety_settings:
                    if config_payload is None:
                        config_payload = {"safety_settings": self._google_genai_safety_settings}
                    else:
                        config_payload.setdefault("safety_settings", self._google_genai_safety_settings)
                if config_payload:
                    request_payload["config"] = config_payload

                response = self._google_genai_client.models.generate_content(**request_payload)
                text = self._extract_google_response_text(response)
                if not text:
                    raise RuntimeError(
                        f"Google GenAI response for '{self.model_name}' did not include text content."
                    )
                self._google_decrease_delay()
                return text

            except genai_errors.APIError as exc:
                response_obj = getattr(exc, "response", None)
                status_code = getattr(exc, "code", None)
                if status_code is None and hasattr(response_obj, "status_code"):
                    status_code = getattr(response_obj, "status_code")
                if status_code in self._google_genai_retry_statuses and attempt < max_retries:
                    if status_code == 429:
                        self._google_adjust_rate_limit_on_429(response_obj)
                    delay = self._compute_google_backoff(response_obj, attempt)
                    logger.warning(
                        "Google GenAI HTTP %s for %s on attempt %d/%d. Retrying in %.2fs.",
                        status_code,
                        self.model_name,
                        attempt,
                        max_retries,
                        delay,
                    )
                    self._google_increase_delay()
                    time.sleep(delay)
                    last_error = exc
                    continue
                self._google_increase_delay()
                raise RuntimeError(
                    f"Google GenAI request failed for '{self.model_name}': {exc}"
                ) from exc

            except Exception as exc:
                if attempt < max_retries:
                    delay = self._compute_google_backoff(None, attempt)
                    logger.warning(
                        "Google GenAI error for %s (%s) on attempt %d/%d. Retrying in %.2fs.",
                        self.model_name,
                        exc,
                        attempt,
                        max_retries,
                        delay,
                    )
                    self._google_increase_delay()
                    time.sleep(delay)
                    last_error = exc
                    continue
                self._google_increase_delay()
                raise RuntimeError(f"Google GenAI request failed for '{self.model_name}': {exc}") from exc

            finally:
                self._google_record_call()

        if last_error:
            raise RuntimeError(
                f"Google GenAI request failed for '{self.model_name}' after retries: {last_error}"
            ) from last_error

        raise RuntimeError(f"Google GenAI request failed for '{self.model_name}' after retries.")

    def _compute_openrouter_backoff(self, response: Optional[httpx.Response], attempt: int) -> float:
        retry_after: Optional[float] = None
        header_value: Optional[str] = None

        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            header_value = headers.get("Retry-After") if hasattr(headers, "get") else None
            if header_value:
                header_value = str(header_value).strip()
                try:
                    retry_after = float(header_value)
                except ValueError:
                    try:
                        retry_dt = parsedate_to_datetime(header_value)
                    except (TypeError, ValueError, OverflowError):
                        retry_dt = None

                    if retry_dt is not None:
                        if retry_dt.tzinfo is None:
                            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        delta_seconds = (retry_dt - now).total_seconds()
                        if delta_seconds > 0:
                            retry_after = float(delta_seconds)

        if retry_after is None:
            delay = self._openrouter_backoff_base * (2 ** (attempt - 1))
            jitter = random.random() * self._openrouter_backoff_jitter
            delay = min(delay + jitter, self._openrouter_backoff_cap)
        else:
            delay = float(retry_after)
            jitter = random.random() * self._openrouter_backoff_jitter
            delay += jitter

        min_delay = max(self._openrouter_throttle_seconds, 0.0)
        delay = max(delay, min_delay)
        return delay

    def _extract_openrouter_content(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if not choices:
            raise ValueError("OpenRouter response missing 'choices' field.")

        choice = choices[0] or {}
        message = choice.get("message", {}) or {}
        content = message.get("content")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if text_value:
                        parts.append(str(text_value))
                elif isinstance(item, str):
                    parts.append(item)
            content = "".join(parts).strip()
        elif isinstance(content, str):
            content = content.strip()
        else:
            content = ""

        if not content:
            text_field = choice.get("text")
            if isinstance(text_field, list):
                text_field = "".join(str(elem) for elem in text_field if elem)
            if isinstance(text_field, str):
                content = text_field.strip()

        if not content:
            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_text = delta.get("content") or delta.get("text")
                if isinstance(delta_text, list):
                    delta_text = "".join(str(elem) for elem in delta_text if elem)
                if isinstance(delta_text, str):
                    content = delta_text.strip()

        if not content:
            raise ValueError(f"OpenRouter response contains no text content: {data}")

        return content

    # Anthropic helper methods
    def _anthropic_wait_for_slot(self) -> None:
        throttle = max(self._anthropic_throttle_seconds, 0.0)
        with self._anthropic_state_lock:
            state = self._anthropic_global_state
            last_call = state["last_call"]
            extra = state["extra_delay"]

        wait_target = throttle + max(0.0, extra)
        if wait_target > 0.0 and last_call is not None:
            elapsed = time.perf_counter() - last_call
            if elapsed < wait_target:
                time.sleep(wait_target - elapsed)

        with self._anthropic_state_lock:
            self._anthropic_global_state["last_call"] = time.perf_counter()

    def _anthropic_increase_delay(self) -> None:
        with self._anthropic_state_lock:
            state = self._anthropic_global_state
            extra = state["extra_delay"]
            base = self._anthropic_backoff_base
            cap = self._anthropic_backoff_cap
            jitter = self._anthropic_backoff_jitter
            new_extra = min(cap, extra * base)
            if jitter > 0.0:
                import random
                new_extra *= (1.0 + random.uniform(-jitter, jitter))
            state["extra_delay"] = max(0.0, new_extra)

    def _anthropic_decrease_delay(self) -> None:
        with self._anthropic_state_lock:
            state = self._anthropic_global_state
            extra = state["extra_delay"]
            if extra <= 0.0:
                return
            state["extra_delay"] = max(0.0, extra * 0.5)

    def _compute_anthropic_backoff(self, exc: Exception, attempt: int) -> float:
        base = self._anthropic_backoff_base
        cap = self._anthropic_backoff_cap
        jitter = self._anthropic_backoff_jitter
        delay = min(cap, base ** attempt)
        if jitter > 0.0:
            import random
            delay *= (1.0 + random.uniform(-jitter, jitter))
        return max(0.0, delay)

    # OpenAI helper methods
    def _openai_wait_for_slot(self) -> None:
        throttle = max(self._openai_throttle_seconds, 0.0)
        with self._openai_state_lock:
            state = self._openai_global_state
            last_call = state["last_call"]
            extra = state["extra_delay"]

        wait_target = throttle + max(0.0, extra)
        if wait_target > 0.0 and last_call is not None:
            elapsed = time.perf_counter() - last_call
            if elapsed < wait_target:
                time.sleep(wait_target - elapsed)

        with self._openai_state_lock:
            self._openai_global_state["last_call"] = time.perf_counter()

    def _openai_increase_delay(self) -> None:
        with self._openai_state_lock:
            state = self._openai_global_state
            extra = state["extra_delay"]
            base = self._openai_backoff_base
            cap = self._openai_backoff_cap
            jitter = self._openai_backoff_jitter
            new_extra = min(cap, extra * base)
            if jitter > 0.0:
                import random
                new_extra *= (1.0 + random.uniform(-jitter, jitter))
            state["extra_delay"] = max(0.0, new_extra)

    def _openai_decrease_delay(self) -> None:
        with self._openai_state_lock:
            state = self._openai_global_state
            extra = state["extra_delay"]
            if extra <= 0.0:
                return
            state["extra_delay"] = max(0.0, extra * 0.5)

    def _compute_openai_backoff(self, exc: Exception, attempt: int) -> float:
        base = self._openai_backoff_base
        cap = self._openai_backoff_cap
        jitter = self._openai_backoff_jitter
        delay = min(cap, base ** attempt)
        if jitter > 0.0:
            import random
            delay *= (1.0 + random.uniform(-jitter, jitter))
        return max(0.0, delay)

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client is not initialized.")

        try:
            from anthropic import Anthropic, APIError
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package is required for Anthropic Claude models. "
                "Install it inside the poetry environment before running the classifier."
            ) from exc

        max_retries = max(1, self._anthropic_max_retries)
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_retries:
            attempt += 1
            self._anthropic_wait_for_slot()

            try:
                response = self._anthropic_client.messages.create(
                    model=self.model_name,
                    max_tokens=self.generation_config.get("max_tokens", 16),
                    temperature=self.generation_config.get("temperature", 0.0),
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = response.content[0].text if response.content else ""
                if not content:
                    raise RuntimeError("Anthropic response contains no text content.")
                
                self._anthropic_decrease_delay()
                return content.strip()

            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code in self._anthropic_retry_statuses and attempt < max_retries:
                    delay = self._compute_anthropic_backoff(exc, attempt)
                    logger.warning(
                        "Anthropic API error %s for %s on attempt %d/%d. Retrying in %.2fs.",
                        status_code,
                        self.model_name,
                        attempt,
                        max_retries,
                        delay,
                    )
                    self._anthropic_increase_delay()
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise exc
            except Exception as exc:
                if attempt >= max_retries:
                    raise exc
                delay = self._compute_anthropic_backoff(exc, attempt)
                logger.warning(
                    "Anthropic error for %s on attempt %d/%d: %s. Retrying in %.2fs.",
                    self.model_name,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                self._anthropic_increase_delay()
                time.sleep(delay)
                last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError(f"Anthropic API failed after {max_retries} attempts.")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        try:
            from openai import OpenAI, APIError
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAI models. "
                "Install it inside the poetry environment before running the classifier."
            ) from exc

        max_retries = max(1, self._openai_max_retries)
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_retries:
            attempt += 1
            self._openai_wait_for_slot()

            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.generation_config.get("max_tokens", 16),
                    temperature=self.generation_config.get("temperature", 0.0)
                )
                
                content = response.choices[0].message.content if response.choices else ""
                if not content:
                    raise RuntimeError("OpenAI response contains no text content.")
                
                self._openai_decrease_delay()
                return content.strip()

            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code in self._openai_retry_statuses and attempt < max_retries:
                    delay = self._compute_openai_backoff(exc, attempt)
                    logger.warning(
                        "OpenAI API error %s for %s on attempt %d/%d. Retrying in %.2fs.",
                        status_code,
                        self.model_name,
                        attempt,
                        max_retries,
                        delay,
                    )
                    self._openai_increase_delay()
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise exc
            except Exception as exc:
                if attempt >= max_retries:
                    raise exc
                delay = self._compute_openai_backoff(exc, attempt)
                logger.warning(
                    "OpenAI error for %s on attempt %d/%d: %s. Retrying in %.2fs.",
                    self.model_name,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                self._openai_increase_delay()
                time.sleep(delay)
                last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError(f"OpenAI API failed after {max_retries} attempts.")

    def _classify_question_llm_google(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")
        options = question_config.get("options", []) if question_type == "multiclass" else []

        text_truncated = self._prepare_llm_context(text, question_id)
        prompt = self._build_llm_prompt(
            question_type=question_type,
            question_text=question_text,
            text_truncated=text_truncated,
            options=options,
            question_id=question_id,
        )

        response_text = self._call_google_genai(prompt)

        if question_type == "binary":
            answer, confidence = self._parse_llm_response(response_text)
            if answer is None:
                logger.warning(
                    "Google GenAI returned unparseable response for %s (%s): '%s'. Defaulting to No.",
                    question_id,
                    self.model_name,
                    response_text,
                )
                answer = False
                confidence = 0.1

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "llm_response": response_text,
                "fragments": []
            }

        response_lower = response_text.lower().strip()
        best_option = None
        best_score = 0.0

        for option in options:
            option_lower = option.lower()
            if option_lower in response_lower:
                best_option = option
                best_score = 1.0
                break
            if response_lower in option_lower and best_score < 0.7:
                best_option = option
                best_score = 0.7

        if best_option is None and options:
            logger.warning(
                "Google GenAI response '%s' did not match provided options for %s. Defaulting to first option.",
                response_text,
                question_id,
            )
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

    def _classify_question_llm_openrouter(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")
        options = question_config.get("options", []) if question_type == "multiclass" else []

        text_truncated = self._prepare_llm_context(text, question_id)
        prompt = self._build_llm_prompt(
            question_type=question_type,
            question_text=question_text,
            text_truncated=text_truncated,
            options=options,
            question_id=question_id,
        )

        messages: List[Dict[str, str]] = []
        system_prompt = self.llm_generation_templates.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_text = self._call_openrouter(messages)

        if question_type == "binary":
            answer, confidence = self._parse_llm_response(response_text)
            if answer is None:
                logger.warning(
                    "OpenRouter returned unparseable response for %s (%s): '%s'. Defaulting to No.",
                    question_id,
                    self.model_name,
                    response_text,
                )
                answer = False
                confidence = 0.1

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "llm_response": response_text,
                "fragments": []
            }

        response_lower = response_text.lower().strip()
        best_option = None
        best_score = 0.0

        for option in options:
            option_lower = option.lower()
            if option_lower in response_lower:
                best_option = option
                best_score = 1.0
                break
            if response_lower in option_lower and best_score < 0.7:
                best_option = option
                best_score = 0.7

        if best_option is None and options:
            logger.warning(
                "OpenRouter response '%s' did not match provided options for %s. Defaulting to first option.",
                response_text,
                question_id,
            )
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

    def _classify_question_llm_anthropic(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify using Anthropic Claude API"""
        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")
        options = question_config.get("options", []) if question_type == "multiclass" else []

        text_truncated = self._prepare_llm_context(text, question_id)
        prompt = self._build_llm_prompt(
            question_type=question_type,
            question_text=question_text,
            text_truncated=text_truncated,
            options=options,
            question_id=question_id,
        )

        response_text = self._call_anthropic(prompt)

        if question_type == "binary":
            answer, confidence = self._parse_llm_response(response_text)
            if answer is None:
                logger.warning(
                    "Anthropic returned unparseable response for %s (%s): '%s'. Defaulting to No.",
                    question_id,
                    self.model_name,
                    response_text,
                )
                answer = False
                confidence = 0.1

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "llm_response": response_text,
                "fragments": []
            }

        response_lower = response_text.lower().strip()
        best_option = None
        best_score = 0.0

        for option in options:
            option_lower = option.lower()
            if option_lower in response_lower:
                best_option = option
                best_score = 1.0
                break
            if response_lower in option_lower and best_score < 0.7:
                best_option = option
                best_score = 0.7

        if best_option is None and options:
            logger.warning(
                "Anthropic response '%s' did not match provided options for %s. Defaulting to first option.",
                response_text,
                question_id,
            )
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

    def _classify_question_llm_openai(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify using OpenAI API"""
        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")
        options = question_config.get("options", []) if question_type == "multiclass" else []

        text_truncated = self._prepare_llm_context(text, question_id)
        prompt = self._build_llm_prompt(
            question_type=question_type,
            question_text=question_text,
            text_truncated=text_truncated,
            options=options,
            question_id=question_id,
        )

        response_text = self._call_openai(prompt)

        if question_type == "binary":
            answer, confidence = self._parse_llm_response(response_text)
            if answer is None:
                logger.warning(
                    "OpenAI returned unparseable response for %s (%s): '%s'. Defaulting to No.",
                    question_id,
                    self.model_name,
                    response_text,
                )
                answer = False
                confidence = 0.1

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "llm_response": response_text,
                "fragments": []
            }

        response_lower = response_text.lower().strip()
        best_option = None
        best_score = 0.0

        for option in options:
            option_lower = option.lower()
            if option_lower in response_lower:
                best_option = option
                best_score = 1.0
                break
            if response_lower in option_lower and best_score < 0.7:
                best_option = option
                best_score = 0.7

        if best_option is None and options:
            logger.warning(
                "OpenAI response '%s' did not match provided options for %s. Defaulting to first option.",
                response_text,
                question_id,
            )
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
