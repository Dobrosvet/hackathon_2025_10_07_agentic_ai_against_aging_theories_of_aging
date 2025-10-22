## Presentation Narrative

### 1. Set the Stage – Why Ageing Research Needs Better Tooling
- Open with the challenge: ageing research produces vast volumes of biomedical literature; insights are scattered across thousands of PMC and PubMed records.
- Highlight the bottleneck: manual curation of theories, biomarkers, and interventions is slow and inconsistent, delaying consensus and new hypotheses.
- Position the goal: deliver an automated, end-to-end pipeline that gathers full texts, classifies them against key scientific questions, and exposes results for experts to validate and extend.

### 2. System Overview – The Orchestrated Pipeline
- Describe the five Python microservices (A, B, C, D, X) chained together:
  - **Service A** retrieves open-access article metadata via the PubMed Central E-utilities API.
  - **Service B** downloads full texts and attaches them to the collected records.
  - **Service C** determines whether each paper proposes an ageing theory and extracts highlighted passages.
  - **Service D** answers nine research questions (Q1–Q9) and four evaluation criteria (C1–C4) using configurable LLM approaches.
  - **Service X** aggregates all results into exportable tables and analytics.
- Emphasize the shared Qdrant vector database: every microservice reads and writes to a common “pubmed_papers” collection, preserving a unified dataset.
- Mention orchestration: a Windows PowerShell script (`run.ps1`) starts/stops all components, launches Qdrant, and boots the dashboard.

### 3. Intelligence Inside – Model Flexibility and Benchmarking
- Explain the classification core (`QuestionsClassifierV2`) with switchable model variants:
  - Default PubMedBERT sentence embeddings for reliable, local inference.
  - Optional GPT-4o-mini, Gemini 2.5, Anthropic, OpenRouter models for deeper language understanding.
- Share that benchmarking tools (`model_benchmark_v2.py`, `models_config_v2.yaml`) allow rapid comparison and scoring of new models.
- Mention validation data loader (`upload_validation_data.py`) that seeds ground-truth answers for Q1–Q9, supporting accuracy tracking and manual review workflows.

### 4. Human in the Loop – Real-Time Dashboard and Review
- Introduce the React dashboard built with Vite, React, TypeScript, and WebSockets:
  - Service monitors show progress, errors, and controls (start/stop) for each microservice.
  - Theory highlight viewer displays extracted spans for quick triage.
  - Questions & Criteria viewer presents model answers, manual annotations, and confidence scores.
  - Validation metrics viewer visualizes per-question accuracy, precision, recall, and confusion matrices.
  - Database viewer previews tables and triggers CSV/XLSX exports from the export service.
- Emphasize manual QA support: reviewers can label papers, add rationales, or skip ambiguous cases with immediate persistence to Qdrant.

### 5. Technical Stack – Transparency and Reproducibility
- Summarize technology choices:
  - **Backend:** Python 3.10, FastAPI, Uvicorn, PyTorch, Transformers, sentence-transformers, httpx, qdrant-client.
  - **LLM providers:** Hugging Face, OpenAI, Google Gemini, Anthropic, OpenRouter (all keys managed via `.env`).
  - **Database:** Qdrant with bundled Windows binary and optional proxy for the Web UI.
  - **Frontend:** React, Vite, TypeScript, WebSocket-based live updates.
  - **Environment management:** Poetry for dependencies; scripts adopt UTF-8 without BOM to ensure portability.
- Reinforce operational details: Windows-first deployment, single PowerShell entry point, logs in `data/logs`, exports in `data/exports`.

### 6. Impact – What Audience Should Take Away
- Stress the value for researchers:
  - Rapid triage of literature into core ageing theories.
  - Consistent, repeatable question answering across papers, enabling longitudinal analysis.
  - Ability to benchmark and swap models quickly as LLM capabilities evolve.
  - Clear audit trail through manual annotations and validation metrics.
- Tie back to broader outcomes: building a curated dataset that accelerates understanding of ageing mechanics, supports hypothesis generation, and aids consensus-building around theories.

### 7. Close with a Call to Action
- Invite collaborators to:
  - Review the exported tables and validation metrics.
  - Plug in additional LLMs via the existing configuration system.
  - Extend the dashboard or analytics modules to meet their lab’s needs.
- Encourage contributions via the microservice structure: each component is isolated, well-documented, and testable.
