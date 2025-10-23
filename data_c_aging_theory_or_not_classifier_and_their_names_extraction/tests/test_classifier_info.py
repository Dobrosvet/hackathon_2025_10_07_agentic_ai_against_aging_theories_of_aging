import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aging_theory_classifier import AgingTheoryClassifier
from pubmedbert_classifier import PubmedBertClassifier


def test_get_model_info_no_eager_initialization(monkeypatch):
    calls = []

    def _initialize_stub(self):
        calls.append("called")
        raise AssertionError("PubMedBERT initialization should not run during get_model_info")

    monkeypatch.setattr(PubmedBertClassifier, "_initialize", _initialize_stub, raising=False)

    classifier = AgingTheoryClassifier(mode="embedding", use_gpu=False)
    info = classifier.get_model_info()

    assert info["mode"] == "embedding"
    assert "embedding_model" in info
    assert info["embedding_model"]["initialized"] is False
    assert calls == []


def test_initialize_sets_initialized_before_precompute(monkeypatch):
    # Prepare lightweight torch and sentence_transformers stubs
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda _: "cpu",
            get_device_properties=lambda _: types.SimpleNamespace(total_memory=0),
        ),
        device=lambda name: name,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    dummy_model = types.SimpleNamespace(
        max_seq_length=None,
        encode=lambda texts, **_: [[0.0] * 1 for _ in texts],
    )

    sentence_transformers_stub = types.SimpleNamespace(
        SentenceTransformer=lambda *args, **kwargs: dummy_model
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_stub)

    flag = {"checked": False}

    def fake_precompute(self):
        assert self._initialized is True, "expected initialized flag before precompute"
        flag["checked"] = True

    monkeypatch.setattr(PubmedBertClassifier, "_precompute_theory_embeddings", fake_precompute, raising=False)

    classifier = PubmedBertClassifier(use_gpu=False)
    classifier._initialize()

    assert flag["checked"] is True
