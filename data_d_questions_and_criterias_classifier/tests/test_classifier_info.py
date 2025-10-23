import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from questions_classifier_v2 import QuestionsClassifierV2  # noqa: E402


def test_get_model_info_without_initialization(monkeypatch):
    calls = []

    def init_stub(self):
        calls.append("init")
        raise AssertionError("Initialization should not be triggered during get_model_info")

    monkeypatch.setattr(QuestionsClassifierV2, "_initialize", init_stub, raising=False)

    classifier = QuestionsClassifierV2(
        config_path=str(ROOT / "config.yaml"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        approach="sentence_bert",
        use_gpu=False,
        quantize=False,
    )

    info = classifier.get_model_info()

    assert info["initialized"] is False
    assert info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert calls == []


def test_get_model_info_force_initialization_failure(monkeypatch):
    def init_stub(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(QuestionsClassifierV2, "_initialize", init_stub, raising=False)

    classifier = QuestionsClassifierV2(
        config_path=str(ROOT / "config.yaml"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        approach="sentence_bert",
        use_gpu=False,
    )

    info = classifier.get_model_info(force_initialize=True)

    assert info["initialized"] is False
    assert info.get("device") is None
