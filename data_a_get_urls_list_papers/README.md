# PubMed Central Data Fetcher

## Overview
This microservice discovers open-access PubMed Central (PMC) articles that match the baked-in aging query and stores their metadata in a Qdrant collection. The service exposes a FastAPI application with REST endpoints for lifecycle control and a WebSocket feed for live progress updates. It relies on the NCBI E-utilities API (`esearch.fcgi` and `esummary.fcgi`) and persists results in the shared `pubmed_papers` collection.

Default search query (editable in `service_state["search_query"]`):  
`(aging) AND (theory OR paradigm) OR Aging[MeSh] AND open_access[Filter]`

## Code structure
| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI service, background harvesting loop, WebSocket broadcasting, Qdrant orchestration |
| `pubmed_fetcher.py` | Async client for E-utilities: full export of PMC IDs and batched metadata fetch |
| `qdrant_storage.py` | Inserts metadata into `pubmed_papers`, generates deterministic 384-d hash embeddings, deduplicates existing IDs |
| `pyproject.toml` | Poetry project definition and runtime dependencies |

Logs are written to `../data/logs` and the Qdrant client targets `http://localhost:6333` by default.

## API surface
### REST
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service status probe |
| `GET` | `/api/status` | Current job state (`status`, `progress`, totals, errors, search query, etc.) |
| `POST` | `/api/start` | Launch metadata harvesting in a background task |
| `POST` | `/api/stop` | Request graceful stop of the running job |

All responses are JSON; `/api/status` mirrors the fields defined in `service_state`.

### WebSocket
`/ws` streams JSON objects:
- `{"type": "state", "data": {...}}` – current progress snapshot.
- `{"type": "log", "data": {...}}` – individual log lines.
- `{"type": "logs", "data": [...]}` – last 100 log entries on connect.

Use the feed to power dashboards or CLI monitors without polling the REST endpoint.

## Data written to Qdrant
Each paper is stored as:
```json
{
  "pmc_id": "123456",
  "title": "...",
  "authors": ["John Doe", "Jane Smith"],
  "source": "Journal Name",
  "pubdate": "2024-01-15",
  "doi": "10.1234/example",
  "pmid": "98765432",
  "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/",
  "pdf_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"
}
```

The vector is a 384-d normalised hash of the title (or PMC ID fallback). `qdrant_storage.get_existing_ids()` protects against re-ingesting already processed PMC IDs.

## Running locally
```powershell
cd data_a_get_urls_list_papers
poetry install
poetry run python main.py
```

The service listens on `http://127.0.0.1:8002`. Make sure a Qdrant instance is up (`docker run qdrant/qdrant` or equivalent). Stop the process with `Ctrl+C`; the background task also honours `POST /api/stop`.

### Health check
```powershell
Invoke-RestMethod http://127.0.0.1:8002/api/status | ConvertTo-Json -Depth 5
```

### Sample WebSocket consumer (PowerShell)
```powershell
poetry run python - <<'PY'
import asyncio, websockets, json
async def watch():
    async with websockets.connect("ws://127.0.0.1:8002/ws") as ws:
        while True:
            print(json.loads(await ws.recv()))
asyncio.run(watch())
PY
```

## Dependencies
Declared in `pyproject.toml`:
- FastAPI 0.119 for the HTTP surface.
- Uvicorn (with `standard` extras) as the ASGI runner.
- `httpx` for async E-utilities access.
- `qdrant-client` for persistence.
- `websockets`, `python-multipart`, and Biopython (the latter is available for future sequence utilities, not currently imported).

Install them via `poetry install`; do not run the service outside the Poetry environment.

## Rate limiting & API etiquette
- `PubMedFetcher` adds a 0.5s pause between `esearch` batches and a 0.34s pause before every `esummary` call to respect the documented 3 requests/second limit.
- A single retry is performed on HTTP 429 responses. Subsequent failures are logged and the batch is skipped.
- All metadata requests are JSON (`retmode=json`); full text is not downloaded here (see microservice B).

## Operational notes
- The search query is stored in `service_state` and included in `/api/status` responses.
- Counters (`saved`, `errors`, `skipped`, `db_count`) track Qdrant writes and existing records.
- Logs and states are capped (1000 log entries) to avoid unbounded memory.
- Calling `/api/start` while a job is active returns an error; `/api/stop` flips the status to `stopped`, which the background loop checks before writing new batches.
