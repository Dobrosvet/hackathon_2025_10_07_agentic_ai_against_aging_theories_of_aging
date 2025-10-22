# Full Project Description

## Purpose and Top-Level Overview
The repository implements an end-to-end pipeline for analysing ageing-related scientific literature. A sequence of Python microservices collects open-access PubMed Central (PMC) papers, fetches their full text, classifies them with large language models, and stores all artefacts in a shared Qdrant vector database. A React dashboard connects to each service through WebSocket feeds to display live status, manual-review tools, and export utilities. Benchmark scripts and validation loaders support experimentation with alternative language models.

Project orchestration targets Windows and is automated by `run.ps1`, which launches or stops every service, starts Qdrant (bundled in the `qdrant` directory together with the static Web UI files in `dist-qdrant`), and stores process identifiers in `.running_services.json`. Common environment variables (Hugging Face token, OpenRouter key, Google Gemini key, Anthropic key, OpenAI key) are declared in `.env.example` and loaded at runtime. Outputs land under `data/` (`db/` for Qdrant storage, `logs/` for service logs, `exports/` for generated tables, `s3/` as reserved storage). The repository bundles a Windows Qdrant binary, an optional HTTP proxy (`qdrant_proxy.py`) that serves the packaged Web UI, and multiple utility scripts.

## Microservice A вЂ“ `data_a_get_urls_list_papers`
* **Goal:** Discover PMC article metadata that matches the ageing query `(aging) AND (theory OR paradigm) OR Aging[MeSh] AND open_access[Filter]`.
* **Implementation:** `main.py` hosts a FastAPI service with REST endpoints (`/`, `/api/status`, `/api/start`, `/api/stop`) and a WebSocket feed (`/ws`). Starting a job launches `PubMedFetcher`, which uses the NCBI E-utilities `esearch.fcgi` and `esummary.fcgi` endpoints via `httpx` to gather every matching PMC ID and its metadata in batches.
* **Persistence:** `qdrant_storage.py` creates deterministic 384-dimensional hash embeddings from article titles and writes payloads (`pmc_id`, `title`, `authors`, `source`, `pubdate`, `doi`, `pmid`, `url`, `pdf_url`) to the `pubmed_papers` collection.
* **Dependencies:** Declared in `pyproject.toml` (FastAPI, Uvicorn, `httpx`, `qdrant-client`, `websockets`, Biopython, `python-multipart`). Execution should occur inside the Poetry environment. Logs are redirected to `../data/logs`.

## Microservice B вЂ“ `data_b_get_full_texts`
* **Goal:** Fetch and store the full article body for records written by microservice A.
* **Implementation:** `main.py` is another FastAPI service (same REST/WebSocket pattern, port `8003`). It retrieves PMC IDs without `full_text` using `qdrant_storage.get_papers_without_full_text`, calls `PMCTextFetcher` to download XML via `efetch.fcgi`, converts it to plain text, and updates Qdrant (`full_text`, `full_text_timestamp`).
* **Error Handling:** A 0.34вЂЇs delay guards against PMC rate limits; HTTP 429 triggers one retry after a two-second pause.
* **Dependencies:** Poetry project includes FastAPI, Uvicorn, `httpx`, `qdrant-client`, `websockets`, `python-multipart`. Outputs are appended to the shared collection and logs directory.

## Microservice C вЂ“ `data_c_aging_theory_or_not_classifier_and_their_names_extraction`
* **Goal:** Determine whether each full-text article argues for an ageing theory, extract grounded mentions, and support manual review.
* **Classifier:** `aging_theory_classifier.py` exposes modes `keyword`, `embedding` (PubMedBERT sentence transformer), `hybrid` (keyword prefilter + PubMedBERT verification), and `gpt4o` (OpenAI GPT-4o-mini). Mode selection is configured in `config.yaml` and can be overridden with `AGING_CLASSIFIER_MODEL_VARIANT`.
* **Service interface:** `main.py` runs on port `8004` and offers:
  - `/api/start`, `/api/stop`, `/api/status`, `/ws` for batch classification progress.
  - Manual review endpoints (`/api/papers/unreviewed`, `/api/papers/{pmc_id}/label`, `/api/papers/labeled`, `/api/papers/{pmc_id}/skip`).
* **Storage:** `qdrant_storage.py` augments payloads with `is_aging_theory`, confidence, extracted theory spans, manual labels (`manual_label`, `manual_label_timestamp`, `user_comment`), skip markers, and summary statistics. `pubmedbert_classifier.py` implements batched embedding inference; a thin shim keeps the legacy `BioformerClassifier` name for compatibility.
* **Dependencies:** Poetry stack adds `sentence-transformers`, PyTorch, `openai`, FastAPI, `qdrant-client`, `websockets`. Mode `gpt4o` requires `OPENAI_API_KEY`. Logs and highlighted spans stream over WebSocket for the dashboard.

## Microservice D вЂ“ `data_d_questions_and_criterias_classifier`
* **Goal:** Answer nine science questions (Q1вЂ“Q9) and four criteria (C1вЂ“C4) for papers already labelled as ageing theories.
* **Classifier core:** `questions_classifier_v2.py` supports multiple approaches (sentence embeddings, NLI cross-encoders, OpenRouter, Google Gemini, Anthropic, OpenAI). Default configuration in `config.yaml` selects the PubMedBERT sentence-transformer variant; `QUESTIONS_CLASSIFIER_MODEL_VARIANT` switches to GPT-4o-mini.
* **Service interface:** `main.py` (port `8005`) mirrors the control endpoints above and adds:
  - `/api/paper/{pmc_id}` to inspect stored classifications.
  - `/api/paper/{pmc_id}/annotate` and `/api/paper/{pmc_id}/delete_annotation` for manual rationales.
  - Validation helpers (`/api/validation/papers`, `/api/validation/metrics`, `/api/validation/classify`, `/api/validation/comparison/{paper_url}`).
* **Storage:** `qdrant_storage.py` writes `questions_classification`, `criteria_classification`, `questions_timestamp`, manual annotation arrays, and manual label metadata. The module also exposes aggregate statistics for validation and manual review dashboards.
* **Benchmarking:** `model_benchmark_v2.py` runs standalone evaluations driven by `models_config_v2.yaml`; desired models are enabled via the `enabled` flag before execution.
* **Dependencies:** Poetry project includes FastAPI, PyTorch 2.3, `transformers` 4.45, `accelerate`, `sentence-transformers`, `bitsandbytes`, `google-genai`, `anthropic`, `openai`, `tiktoken`, `qdrant-client`, and `websockets`. Unit tests (pytest) cover LLM integrations.

## Microservice X вЂ“ `data_x_view_db_and_export_to_tables`
* **Goal:** Aggregate results stored in Qdrant into publishable tables and statistics, and export them to CSV or Excel.
* **Implementation:** `main.py` (port `8006`) exposes:
  - `/api/statistics` for high-level counts (papers, theories, coverage).
  - `/api/tables/preview/{table_number}` and `/api/tables/full/{table_number}` with pagination.
  - `/api/export/{table_number}` (CSV or XLSX) and `/api/export/all` (Excel with multiple sheets).
  - `/api/refresh` and `/ws` for live dashboards.
* **Table logic:** `qdrant_storage.py` reconstructs three tables: theory counts, paper listings, and a detailed matrix with Q1вЂ“Q9 and C1вЂ“C4 answers. `export_service.py` writes timestamped CSV/Excel files to `../data/exports`, auto-adjusting column widths.
* **Dependencies:** Poetry project requires FastAPI, Uvicorn, `qdrant-client`, pandas, `openpyxl`, `websockets`, `python-multipart`.

## React Dashboard вЂ“ `client/`
* **Stack:** Vite, React, TypeScript, and WebSockets (specified in the root README).
* **Components:**
  - `ServiceMonitor` widgets connect to each microserviceвЂ™s `/ws` endpoint, display progress and logs, and expose Start/Stop buttons that call the REST API.
  - `TheoryHighlightViewer` shows real-time output from the ageing-theory classifier, including highlighted text spans and preview context.
  - `QuestionsClassifierViewer` renders question and criteria answers, manual annotations, and paper metadata.
  - `ValidationMetricsViewer` queries service DвЂ™s validation endpoints and displays per-question metrics and confusion matrices.
  - `DatabaseViewer` consumes the export service, providing previews and triggers for CSV/XLSX downloads.
* **Entry point:** `App.tsx` arranges all widgets in a вЂњMicroservices DashboardвЂќ page. Global styles live in `App.css` and `index.css`.
* **Development:** Managed via `bun` (as per root README). The client relies exclusively on the microservice APIs and WebSockets, so backend URLs are hardcoded to localhost ports.

## Validation Data Loader вЂ“ `upload_validation_data/`
* Contains `upload_validation_data.py`, which populates Qdrant with ground-truth labels from `validate_data_fulltext.csv` and associated Markdown files (PMC and PubMed records stored in the same directory). It maps `paper_url` entries to Markdown files, attaches manual answers for Q1вЂ“Q9, and sets `is_validation_data=True`.
* Sample Markdown artefacts (e.g., `PM36096982.md`) and the CSV manifest are included for reference.

## Benchmark & Ancillary Assets
* `dist-qdrant/` bundles the compiled Qdrant Web UI. `qdrant/config/config.yaml` points the static content path to this directory.
* `qdrant/` holds the Windows Qdrant executable, storage directory, and snapshot folder. The config enables HTTP (`6333`) and gRPC (`6334`).
* `qdrant_proxy.py` is an optional HTTP proxy that serves the packaged Web UI and forwards API calls to the Qdrant backend.
* `minio/`, `personal_folder/`, and other directories are placeholders supplied with the repository but not touched by the core pipeline.

## Environment & Execution Notes
* All Python services are Poetry projects targeting Python 3.10. Running them manually requires `poetry install` followed by `poetry run python main.py` in each microservice directory.
* `run.ps1` automates the full stack: it stops previously recorded processes, frees listening ports, launches Qdrant, services AвЂ“D and X, starts the React client via Bun, and records new PIDs. The script expects elevated PowerShell (Administrator).
* `.env.example` documents required tokens (`HF_TOKEN`, `OPENROUTER_API_KEY`, `GOOGLE_GENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). The `.env` file is consumed automatically by the Python services.
* Logs for all services accumulate under `data/logs`. Exports appear under `data/exports`. Qdrant data is stored under `data/db` when the embedded server is used.
* The system targets Qdrant as the sole database; no additional storage backends are configured. All services communicate via HTTP/WebSocket and share state exclusively through Qdrant.

## High-Level Workflow
1. **Collection:** Service A pulls PMC metadata into Qdrant.  
2. **Full text ingestion:** Service B adds `full_text` payloads.  
3. **Theory detection:** Service C classifies papers as ageing theories, extracts theory spans, and supports manual labels.  
4. **Question answering:** Service D attaches answers for Q1вЂ“Q9 and C1вЂ“C4, plus validation metrics and manual annotations.  
5. **Inspection & export:** Service X aggregates the dataset into three tables and exports CSV/XLSX archives.  
6. **Dashboard:** The React client monitors and controls each service and displays live outputs.  
7. **Validation & benchmarking:** Optional scripts (`upload_validation_data.py`, `model_benchmark_v2.py`) provide ground-truth ingestion and model evaluation.

No code is reproduced in this document; all details originate from the repository structure and configuration files.

