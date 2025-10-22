# Aging Theory Classifier & NER Service

## Overview
This microservice classifies full-text PMC articles that already contain aging-theory markers and writes the results back to the shared `pubmed_papers` Qdrant collection. It can operate with four detection modes (`keyword`, `embedding`, `hybrid`, `gpt4o`) and exposes REST plus WebSocket APIs for automation and human-in-the-loop review. Manual reviewers can label papers and attach rationale fragments; classification progress and logs stream in real time.

Default port: `8004` (overridable in `config.yaml`).

## Code structure
| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI service, background classifier, manual review endpoints, WebSocket updates |
| `aging_theory_classifier.py` | High-level orchestrator that wraps the available inference modes |
| `pubmedbert_classifier.py` | SentenceTransformer wrapper (`pritamdeka/S-PubMedBert-MS-MARCO`) with batching utilities |
| `batch_processor.py` | Helper for batched embedding inference (used when `supports_batch_embeddings` is true) |
| `qdrant_storage.py` | Reads and updates `pubmed_papers` payloads, tracks manual labels, skip flags, statistics |
| `config.yaml` | Runtime configuration and model variant catalogue |
| `pyproject.toml` | Poetry dependencies (`sentence-transformers`, `openai`, `fastapi`, `qdrant-client`, …) |

Logs land in `../data/logs`. Exports or ancillary data are not generated here.

## Classification modes
Mode selection happens in `config.yaml` under `classifier.model_variant` and may be overridden at launch with `AGING_CLASSIFIER_MODEL_VARIANT`.

| Mode | Description |
| --- | --- |
| `keyword` | Pure phrase matching against the curated theory lexicon |
| `embedding` | PubMedBERT sentence embeddings + cosine similarity (default) |
| `hybrid` | Keyword pre-filter + PubMedBERT verification |
| `gpt4o` | Delegates decisions to `gpt-4o-mini` via OpenAI’s chat completions API |

`gpt4o` requires `OPENAI_API_KEY` in the environment. The embedding modes perform all computation locally and respect the `use_gpu` flag in the classifier config.

## Configuration highlights
`config.yaml` stores:
- `classifier.model_variants` — ready-to-use profiles (`pubmedbert_sbert`, `gpt4o_mini`).
- `classifier.embedding_model` and `classifier.llm_model` — resolved model names after variant selection.
- API binding (`api.host`, `api.port`, etc.).
- Qdrant connection string (`qdrant.host`, `qdrant.port`).

Set `AGING_CLASSIFIER_MODEL_VARIANT=gpt4o_mini` to switch to GPT-4o-mini without editing the YAML.

## API surface
### REST
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service health probe |
| `GET` | `/api/status` | Classification job state (`status`, counters, preview text, totals) |
| `POST` | `/api/start` | Launch classifier in the background |
| `POST` | `/api/stop` | Request graceful stop |
| `GET` | `/api/paper/{pmc_id}` | Retrieve payload snapshot (full text, classifications, annotations) |
| `GET` | `/api/papers/unreviewed` | List papers with full text but no `manual_label` (supports `limit`, `offset`) |
| `POST` | `/api/papers/{pmc_id}/label` | Persist manual label (`{"label": bool, "comment": ""}`) |
| `GET` | `/api/papers/labeled` | List already reviewed items (with pagination) |
| `POST` | `/api/papers/{pmc_id}/skip` | Mark a paper as skipped |

### WebSocket
`/ws` emits:
- `state` — full `service_state` snapshot (status, counters, highlighted spans, etc.).
- `log` — individual log entries.
- `logs` — last 100 log entries on connect.

## Data written to Qdrant
Classification batches call `qdrant_storage.update_papers_batch`, which stores:
- `questions_classification` — answers for Q1–Q9 (`{"Q1": {"answer": "...", "confidence": ...}, ...}`).
- `criteria_classification` — answers for C1–C4 in the same format.
- `questions_timestamp` — ISO timestamp of the last run.
- `manual_annotations` — reviewer-added rationale fragments (`{"type": "Q1", "answer": ..., "fragment": {...}}`).
- `manual_label`, `manual_label_timestamp`, `user_comment`, and `review_status` for human decisions.
- `skip_timestamp` when reviewers skip a paper.

Statistics endpoints rely on these payload fields to count progress.

## Running locally
```powershell
cd data_c_aging_theory_or_not_classifier_and_their_names_extraction
poetry install
# add OPENAI_API_KEY to the environment if you plan to use gpt4o_mini
poetry run uvicorn main:app --host 0.0.0.0 --port 8004
```

Ensure Qdrant is reachable at the location defined in `config.yaml` and that services A and B have already populated `pubmed_papers` with metadata and full texts.

### Example: start a classification run
```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8004/api/start
```

### Example: label a paper
```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8004/api/papers/123456/label `
  -Body (@{label=$true; comment="Clear mitochondrial theory."} | ConvertTo-Json) `
  -ContentType "application/json"
```

## Dependencies
Key packages (see `pyproject.toml` for versions):
- `fastapi`, `uvicorn`, `websockets` — service layer.
- `qdrant-client` — persistence.
- `sentence-transformers`, `torch` — embedding mode.
- `openai` — GPT-4o-mini integration.
- `pandas`, `numpy` are pulled indirectly by `sentence-transformers`.

Install everything with `poetry install`; do not run scripts outside the Poetry environment.

## Operational notes
- `/api/start` rejects concurrent runs; `/api/stop` flips the state to `stopped`, which the worker loop respects.
- Classification batches flush every 10 papers to limit write amplification. Remaining items are saved when the loop finishes.
- Preview text in `service_state["current_text_preview"]` is capped at 2 000 characters to keep WebSocket payloads manageable.
- Manual skips and labels update `review_status` so reviewers can revisit their decisions later.
- `config.yaml` is UTF-8 without BOM (strictly enforced by the repository tooling).
