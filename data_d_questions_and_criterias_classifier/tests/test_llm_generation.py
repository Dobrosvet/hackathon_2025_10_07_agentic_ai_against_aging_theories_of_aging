import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from questions_classifier_v2 import QuestionsClassifierV2  # noqa: E402


def _write_config(tmp_path, model_name: str, require_token: bool = True) -> str:
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
            "require_token_for_models": [model_name] if require_token else [],
        },
    }
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
