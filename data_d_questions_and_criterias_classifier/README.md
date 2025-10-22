# Questions & Criteria Classifier Service

## Overview
This microservice classifies ageing-related papers that already passed the theory detector (service C). It produces answers for nine research questions (Q1–Q9) and four evaluation criteria (C1–C4), stores the results in the `pubmed_papers` Qdrant collection, and exposes REST/WebSocket APIs for automation and manual review. Classification is handled by `QuestionsClassifierV2`, a pluggable wrapper that supports sentence embeddings, cross-encoders, and remote LLM providers.

Default port: `8005` (overridable via `config.yaml`).

## Code structure
| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI app, background classification loop, manual review endpoints, validation utilities |
| `questions_classifier_v2.py` | Core classifier with sliding-window context extraction, provider integrations, and multiple approaches |
| `models_config_v2.yaml` | Model catalogue used by the benchmarking script |
| `config.yaml` | Service settings, model variants, and provider throttling |
| `qdrant_storage.py` | Reads/writes classification payloads, manual annotations, validation sets |
| `tests/test_llm_generation.py` | Unit tests covering authentication and OpenAI/Google/Anthropic call flows |

Logs are written to `../data/logs`. The service expects Qdrant at `http://localhost:6333` with papers that already contain `full_text` and `is_aging_theory=True`.

## Model selection
`config.yaml` defines two ready-to-use variants under `classifier.model_variants`:
- `pubmedbert_sbert` (default): sentence-transformer inference via Hugging Face.
- `gpt4o_mini`: OpenAI GPT-4o-mini generation.

Set the environment variable `QUESTIONS_CLASSIFIER_MODEL_VARIANT` before start-up to switch profiles without editing the file:
```powershell
$env:QUESTIONS_CLASSIFIER_MODEL_VARIANT = "gpt4o_mini"
```

Depending on the chosen approach, the classifier may require:
- `HF_TOKEN` (Hugging Face auth for gated models).
- `OPENROUTER_API_KEY` (OpenRouter providers in `models_config_v2.yaml`).
- `GOOGLE_GENAI_API_KEY` (Gemini 2.5 Flash).
- `ANTHROPIC_API_KEY`.
- `OPENAI_API_KEY`.

`questions_classifier_v2.py` loads these keys from `.env` automatically (`_load_env_from_file`).

## API surface
### REST
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Health probe |
| `GET` | `/api/status` | Worker state (progress, counts, preview text, current classification) |
| `POST` | `/api/start` | Launch the classifier in the background |
| `POST` | `/api/stop` | Request graceful stop |
| `GET` | `/api/paper/{pmc_id}` | Retrieve a single paper with stored classifications and annotations |
| `POST` | `/api/paper/{pmc_id}/annotate` | Append manual rationale (`{"annotation_type": "Q1", "answer": ..., "fragment": {...}}`) |
| `POST` | `/api/paper/{pmc_id}/delete_annotation` | Remove a manual annotation by index |
| `GET` | `/api/validation/papers` | Return validation set entries (`is_validation_data=True`) |
| `GET` | `/api/validation/metrics` | Aggregate metrics for validation papers |
| `POST` | `/api/validation/classify` | Run the classifier over all validation papers |
| `GET` | `/api/validation/comparison/{paper_url}` | Compare model vs. manual answers for one validation paper |

### WebSocket
`/ws` emits:
- `state` — `service_state` snapshot (status, counters, previews, totals).
- `log` — individual log entries.
- `logs` — last 100 log entries on connect.

Use the feed to stream progress into dashboards without polling REST endpoints.

## Data written to Qdrant
Successful batches call `qdrant_storage.update_papers_batch`, which stores:
- `questions_classification` (answers for Q1–Q9, including `answer`, `confidence`, and LLM free text where applicable).
- `criteria_classification` (answers for C1–C4).
- `questions_timestamp` (ISO timestamp).
- `manual_annotations` (reviewer-provided fragments tied to question IDs).
- `manual_label`, `manual_label_timestamp`, `user_comment`, `review_status` for manual QA.
- Validation helpers (`is_validation_data`, `validation_questions`, etc.) remain untouched unless validation routes are used.

The service never modifies vectors; it only enriches payloads.

## Running locally
```powershell
cd data_d_questions_and_criterias_classifier
poetry install
poetry run uvicorn main:app --host 0.0.0.0 --port 8005
```

Prerequisites:
1. Qdrant running at the configured address.
2. Services A, B, and C have populated `pubmed_papers` with full texts and `is_aging_theory=True`.
3. Relevant API keys exported (see “Model selection”).

### Example requests
```powershell
# Start a classification run
Invoke-RestMethod -Method Post http://127.0.0.1:8005/api/start

# Inspect status
Invoke-RestMethod http://127.0.0.1:8005/api/status | ConvertTo-Json -Depth 5

# Add a manual annotation
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8005/api/paper/123456/annotate `
  -Body (@{annotation_type="Q3"; answer="Yes"; fragment=@{text="..."; start_position=120; end_position=160}} | ConvertTo-Json) `
  -ContentType "application/json"
```

## Testing
Run the existing suite (covers provider authentication and error handling):
```powershell
poetry run pytest
```

## Operational notes
- `/api/start` rejects concurrent runs; `/api/stop` switches `service_state["status"]` to `"stopped"`, which the worker loop honours before writing new batches.
- Batches flush every 10 papers; remaining items are saved when the loop finishes.
- `QuestionsClassifierV2` extracts context using question-specific keywords and sliding windows to stay within provider limits (`max_context_length` in `config.yaml`).
- `models_config_v2.yaml` is used by `model_benchmark_v2.py` to evaluate multiple models; it does not affect the runtime service unless you explicitly swap `classifier.model_name`.
- All configuration files are UTF-8 without BOM; avoid editing with editors that inject BOM markers.
