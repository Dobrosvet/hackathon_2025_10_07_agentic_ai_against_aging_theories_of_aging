import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Any

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from questions_classifier_v2 import QuestionsClassifierV2  # noqa: E402


def _write_config(
    tmp_path,
    model_name: str,
    require_token: bool = True,
    provider: str = "huggingface",
    classifier_overrides: dict | None = None,
    openrouter_section: dict | None = None,
    google_section: dict | None = None,
    approaches_section: dict | None = None,
) -> str:
    hf_requirements = [model_name] if (require_token and provider == "huggingface") else []
    config = {
        "classifier": {
            "model_name": model_name,
            "use_gpu": False,
            "quantize": False,
            "show_progress": False,
        },
        "questions": {},
        "criteria": {},
        "huggingface": {
            "token_env_var": "HF_TOKEN_TEST",
            "fail_if_missing": False,
            "require_token_for_models": hf_requirements,
        },
    }
    if provider != "huggingface":
        config["classifier"]["provider"] = provider

    if classifier_overrides:
        config["classifier"].update(classifier_overrides)

    if approaches_section:
        config["approaches"] = approaches_section

    if provider == "openrouter":
        default_section = {
            "base_url": "https://openrouter.ai/api/v1",
            "endpoint": "/chat/completions",
            "token_env_var": "OPENROUTER_API_KEY_TEST",
            "timeout": 30,
            "throttle_seconds": 0,
            "max_retries": 1,
            "backoff_base": 0.1,
            "backoff_jitter": 0.0,
        }
        merged_section = default_section
        if openrouter_section:
            merged_section = {**default_section, **openrouter_section}
        config["openrouter"] = merged_section
    if provider == "google_genai":
        default_google = {
            "token_env_var": "GOOGLE_GENAI_API_KEY_TEST",
            "timeout": 30,
            "max_retries": 2,
            "backoff_base": 0.5,
            "backoff_cap": 10.0,
            "backoff_jitter": 0.0,
            "throttle_seconds": 1.0,
            "rate_limit_per_minute": 60,
            "http_options": {},
            "retry_statuses": [429, 500, 502, 503, 504],
        }
        merged_google = default_google
        if google_section:
            merged_google = {**default_google, **google_section}
        config["google_genai"] = merged_google

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(config_path)


def test_llm_requires_token_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN_TEST", raising=False)
    config_path = _write_config(tmp_path, "google/medgemma-4b-it")

    with pytest.raises(RuntimeError) as error:
        QuestionsClassifierV2(
            config_path=config_path,
            model_name="google/medgemma-4b-it",
            approach="llm_generation",
            use_gpu=False,
            quantize=False,
        )

    assert "Hugging Face token is required" in str(error.value)


def test_llm_initialization_passes_token(monkeypatch, tmp_path):
    import transformers
    import huggingface_hub

    config_path = _write_config(tmp_path, "meta-llama/Llama-3.2-3B-Instruct")
    monkeypatch.setenv("HF_TOKEN_TEST", "hf_test_token")

    recorded = {}

    class DummyTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.eos_token_id = 0

        def __call__(self, *args, **kwargs):
            return {"input_ids": [[0, 1, 2]], "attention_mask": [[1, 1, 1]]}

        def decode(self, *_args, **_kwargs):
            return "Yes"

    class DummyModel:
        def __init__(self):
            self.device = types.SimpleNamespace(type="cpu", index=None)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        recorded["tokenizer"] = {"name": model_name, "kwargs": kwargs}
        return DummyTokenizer()

    def fake_model_from_pretrained(model_name, **kwargs):
        recorded["model"] = {"name": model_name, "kwargs": kwargs}
        return DummyModel()

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        lambda: types.SimpleNamespace(model_info=lambda *_, **__: None),
    )

    classifier = QuestionsClassifierV2(
        config_path=config_path,
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    classifier._initialize()

    assert recorded["tokenizer"]["kwargs"]["token"] == "hf_test_token"
    assert recorded["model"]["kwargs"]["token"] == "hf_test_token"


def test_token_loaded_from_dotenv(monkeypatch, tmp_path):
    import questions_classifier_v2 as qc

    config_path = _write_config(tmp_path, "google/medgemma-4b-it")
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN_TEST=hf_from_env_file\n", encoding="utf-8")

    monkeypatch.delenv("HF_TOKEN_TEST", raising=False)
    monkeypatch.setattr(qc, "ENV_PATH", env_file)

    # Instantiate should succeed because token is provided via .env
    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="google/medgemma-4b-it",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
        requires_auth=True,
    )

    assert classifier._hf_token == "hf_from_env_file"
    monkeypatch.delenv("HF_TOKEN_TEST", raising=False)


def test_model_access_denied(monkeypatch, tmp_path):
    import huggingface_hub  # noqa: F401
    import questions_classifier_v2 as qc

    class DummyApi:
        def model_info(self, *args, **kwargs):
            raise OSError("denied")

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: DummyApi())
    monkeypatch.setenv("HF_TOKEN_TEST", "hf_test_token")

    config_path = _write_config(tmp_path, "google/medgemma-4b-it")
    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="google/medgemma-4b-it",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
        requires_auth=True,
    )

    with pytest.raises(RuntimeError) as err:
        classifier._ensure_model_access()

    assert "huggingface.co/google/medgemma-4b-it" in str(err.value)


def test_openrouter_requires_api_key(monkeypatch, tmp_path):
    QuestionsClassifierV2.reset_openrouter_state()

    config_path = _write_config(
        tmp_path,
        "google/gemini-2.0-flash-exp:free",
        require_token=False,
        provider="openrouter",
        classifier_overrides={"generation_config": {"max_new_tokens": 6}},
    )

    monkeypatch.delenv("OPENROUTER_API_KEY_TEST", raising=False)

    classifier = QuestionsClassifierV2(
        config_path=config_path,
        model_name="google/gemini-2.0-flash-exp:free",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    with pytest.raises(RuntimeError) as err:
        classifier._initialize()

    assert "OpenRouter API key is required" in str(err.value)


def test_openrouter_invocation(monkeypatch, tmp_path):
    import questions_classifier_v2 as qc

    qc.QuestionsClassifierV2.reset_openrouter_state()

    config_path = _write_config(
        tmp_path,
        "openai/gpt-oss-20b:free",
        require_token=False,
        provider="openrouter",
        classifier_overrides={
            "generation_config": {"max_new_tokens": 7, "temperature": 0.1},
            "api": {"timeout": 15},
        },
        openrouter_section={"timeout": 45},
    )

    monkeypatch.setenv("OPENROUTER_API_KEY_TEST", "or_test_key")

    recorded = {}

    class DummyResponse:
        def __init__(self, content: str):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": self._content,
                        }
                    }
                ]
            }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            recorded["client_args"] = (args, kwargs)

        def post(self, endpoint, json):
            recorded["endpoint"] = endpoint
            recorded["payload"] = json
            return DummyResponse("Yes")

    monkeypatch.setattr(qc.httpx, "Client", DummyClient)

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="openai/gpt-oss-20b:free",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    result = classifier.classify_question(
        "Sample scientific context text.",
        "Q_BINARY",
        {"type": "binary", "text": "Is there an intervention suggested?"},
    )

    assert result["answer"] is True
    assert recorded["endpoint"] == "/chat/completions"
    assert recorded["payload"]["model"] == "openai/gpt-oss-20b:free"
    assert recorded["payload"]["max_tokens"] == 7
    assert recorded["payload"]["temperature"] == pytest.approx(0.1, rel=1e-6)
    assert recorded["payload"]["messages"], "OpenRouter payload should include messages"


def test_openrouter_retry_after_header(monkeypatch, tmp_path):
    import questions_classifier_v2 as qc
    from types import SimpleNamespace

    qc.QuestionsClassifierV2.reset_openrouter_state()

    config_path = _write_config(
        tmp_path,
        "google/gemini-2.0-flash-exp:free",
        require_token=False,
        provider="openrouter",
    )

    monkeypatch.setenv("OPENROUTER_API_KEY_TEST", "or_test_token")

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="google/gemini-2.0-flash-exp:free",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    future = datetime.now(timezone.utc) + timedelta(seconds=90)
    header_value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    dummy_response = SimpleNamespace(headers={"Retry-After": header_value})

    delay = classifier._compute_openrouter_backoff(dummy_response, attempt=3)
    assert delay == pytest.approx(90, rel=0.15)

    numeric_header = SimpleNamespace(headers={"Retry-After": "45"})
    delay_numeric = classifier._compute_openrouter_backoff(numeric_header, attempt=2)
    assert delay_numeric == pytest.approx(45, rel=0.15)


def test_llm_prompt_includes_configured_guidelines(tmp_path):
    approaches = {
        "llm_generation": {
            "system_prompt": "System Guideline",
            "general_guidelines": ["Use context only."],
            "question_guidelines": {"Q9": ["Mention calorie restriction mechanism."]},
            "answer_format_binary": "Start with Yes/No.",
            "prompt_template_binary": (
                "{system_prompt}\nGeneral:\n{general_guidelines}\n"
                "Question rubric ({question_id}):\n{question_guidelines}\n"
                "Context:{paper_text}\nTask:{question_text}\n{answer_format_binary}"
            ),
        }
    }

    config_path = _write_config(
        tmp_path,
        "dummy-model",
        require_token=False,
        provider="huggingface",
        approaches_section=approaches,
    )

    classifier = QuestionsClassifierV2(
        config_path=config_path,
        model_name="dummy-model",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    prompt = classifier._build_llm_prompt(
        question_type="binary",
        question_text="Does calorie restriction explain the lifespan change?",
        text_truncated="Calorie restriction extends vertebrate lifespan by reducing insulin signaling.",
        question_id="Q9",
    )

    assert prompt.startswith("System Guideline")
    assert "- Use context only." in prompt
    assert "- Mention calorie restriction mechanism." in prompt
    assert "Start with Yes/No." in prompt


def test_google_genai_classification(monkeypatch, tmp_path):
    import questions_classifier_v2 as qc
    import google.genai as genai

    qc.QuestionsClassifierV2.reset_google_genai_state()

    config_path = _write_config(
        tmp_path,
        "gemini-2.5-flash",
        require_token=False,
        provider="google_genai",
        classifier_overrides={
            "api": {"rate_limit_per_minute": 60, "throttle_seconds": 0.0},
            "generation_config": {"max_output_tokens": 12, "temperature": 0.0, "max_context_length": 128},
        },
        google_section={"rate_limit_per_minute": 60},
    )

    monkeypatch.setenv("GOOGLE_GENAI_API_KEY_TEST", "or_test_google_key")

    recorded: dict[str, Any] = {}

    class DummyModels:
        def generate_content(self, **kwargs):
            recorded["payload"] = kwargs
            return types.SimpleNamespace(text="Yes")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            recorded["client_kwargs"] = kwargs
            self.models = DummyModels()

    monkeypatch.setattr(genai, "Client", DummyClient)

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="gemini-2.5-flash",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    result = classifier.classify_question(
        "Sample scientific passage about longevity interventions.",
        "Q_BINARY",
        {"type": "binary", "text": "Does it propose an intervention?"},
    )

    assert result["answer"] is True
    assert recorded["client_kwargs"]["api_key"] == "or_test_google_key"
    assert recorded["payload"]["model"] == "gemini-2.5-flash"
    assert isinstance(recorded["payload"]["contents"], str)
    assert recorded["payload"]["config"]["max_output_tokens"] == 12
    assert classifier._google_genai_rate_limit_per_minute == 60


def test_google_rate_limit_adjustment_from_headers(tmp_path):
    import questions_classifier_v2 as qc

    qc.QuestionsClassifierV2.reset_google_genai_state()

    config_path = _write_config(
        tmp_path,
        "gemini-2.5-flash",
        require_token=False,
        provider="google_genai",
    )

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="gemini-2.5-flash",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    response = types.SimpleNamespace(headers={"X-RateLimit-Limit-Minute": "12"})
    classifier._google_adjust_rate_limit_on_429(response)

    assert classifier._google_genai_rate_limit_per_minute == 10


def test_google_rate_limit_adjustment_fallback(tmp_path):
    import questions_classifier_v2 as qc

    qc.QuestionsClassifierV2.reset_google_genai_state()

    config_path = _write_config(
        tmp_path,
        "gemini-2.5-flash",
        require_token=False,
        provider="google_genai",
    )

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="gemini-2.5-flash",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    classifier._google_genai_rate_limit_per_minute = 9
    classifier._google_adjust_rate_limit_on_429(None)

    assert classifier._google_genai_rate_limit_per_minute == 4


def test_prepare_context_uses_keywords(tmp_path):
    import questions_classifier_v2 as qc

    approaches = {
        "llm_generation": {
            "max_context_length": 400,
            "question_keywords": {"Q6": ["naked mole rat"]},
        }
    }

    config_path = _write_config(
        tmp_path,
        "dummy-model",
        require_token=False,
        provider="huggingface",
        approaches_section=approaches,
    )

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="dummy-model",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    long_text = (
        "Intro text about aging and longevity. " * 30
        + "The naked mole rat exhibits remarkable longevity due to enhanced DNA repair and proteostasis. "
        + "Additional commentary on experimental evidence. "
    )

    context = classifier._prepare_llm_context(long_text, "Q6")
    assert "naked mole rat" in context
    assert len(context) <= classifier._get_max_context_length()


def test_prepare_context_without_keywords_falls_back(tmp_path):
    import questions_classifier_v2 as qc

    approaches = {
        "llm_generation": {
            "max_context_length": 120,
            "question_keywords": {"Q6": ["naked mole rat"]},
        }
    }

    config_path = _write_config(
        tmp_path,
        "dummy-model",
        require_token=False,
        provider="huggingface",
        approaches_section=approaches,
    )

    classifier = qc.QuestionsClassifierV2(
        config_path=config_path,
        model_name="dummy-model",
        approach="llm_generation",
        use_gpu=False,
        quantize=False,
    )

    long_text = "Aging mechanisms are multifactorial. " * 20
    context = classifier._prepare_llm_context(long_text, "Q2")
    expected = long_text[: classifier._get_max_context_length()]
    assert context == expected


def test_random_classifier_deterministic(tmp_path):
    config_path = _write_config(tmp_path, "random_baseline", require_token=False)

    classifier_a = QuestionsClassifierV2(
        config_path=config_path,
        model_name="random_baseline",
        approach="random",
        use_gpu=False,
        quantize=False,
        random_seed=2025,
    )

    classifier_b = QuestionsClassifierV2(
        config_path=config_path,
        model_name="random_baseline",
        approach="random",
        use_gpu=False,
        quantize=False,
        random_seed=2025,
    )

    question_binary = {"type": "binary"}
    question_multi = {"type": "multiclass", "options": ["Yes, quantitatively shown", "Yes, but not shown", "No"]}

    result_a_binary = classifier_a.classify_question("Sample text", "Q2", question_binary)
    result_b_binary = classifier_b.classify_question("Sample text", "Q2", question_binary)

    result_a_multi = classifier_a.classify_question("Sample text", "Q1", question_multi)
    result_b_multi = classifier_b.classify_question("Sample text", "Q1", question_multi)

    assert result_a_binary == result_b_binary
    assert result_a_multi == result_b_multi
