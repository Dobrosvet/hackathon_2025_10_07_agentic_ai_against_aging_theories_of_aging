# PMC Full Text Downloader

## Overview
This microservice enriches records in the shared `pubmed_papers` Qdrant collection with article bodies downloaded from PubMed Central. It reuses the PMC IDs discovered by **data_a_get_urls_list_papers**, pulls XML articles through the E-utilities `efetch.fcgi` endpoint, extracts plain text, and saves it under the `full_text` payload field. The service publishes its progress over REST and WebSocket interfaces.

## Code structure
| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI service, background downloader, WebSocket broadcasting, Qdrant integration |
| `pmc_text_fetcher.py` | Async XML fetcher with rate limiting and retry-on-429 logic |
| `qdrant_storage.py` | Reads and updates Qdrant payloads (`full_text`, review metadata) |
| `pyproject.toml` | Poetry project definition and dependencies |

Logs are written to `../data/logs`. The default Qdrant endpoint is `http://localhost:6333`; the service expects the `pubmed_papers` collection to exist and contain PMC IDs.

## API surface
### REST
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service status probe |
| `GET` | `/api/status` | Current downloader state (`status`, counters, active paper) |
| `POST` | `/api/start` | Begin fetching full texts in a background task |
| `POST` | `/api/stop` | Ask the running job to halt gracefully |

### WebSocket
`/ws` emits the same message envelope as service A:
- `state` — progress snapshot.
- `log` — individual log lines.
- `logs` — last 100 log entries on connect.

## Processing pipeline
1. `qdrant_storage.get_papers_without_full_text()` scans the collection and returns PMC IDs missing `full_text`.
2. The downloader streams papers (default batch size 10) and calls `PMCTextFetcher.fetch_full_text`.
3. XML content is converted to plain text by walking `<body>` or `<abstract>` nodes.
4. `qdrant_storage.update_papers_batch` writes the `full_text` payload back to Qdrant.
5. Counters (`saved`, `errors`, `skipped`, `db_count`) are updated and broadcast to clients.

Rate limiting: each PMC request is delayed by 0.34 seconds; HTTP 429 triggers a single retry after 2 seconds.

## Running locally
```powershell
cd data_b_get_full_texts
poetry install
poetry run python main.py
```

The service listens on `http://127.0.0.1:8003`. Ensure Qdrant is running and already populated with metadata (via microservice A) before starting. Stop the process with `Ctrl+C` or call `POST /api/stop`.

## Dependencies
Declared in `pyproject.toml`:
- FastAPI 0.119 and Uvicorn (standard extras).
- `httpx` for async HTTP requests.
- `qdrant-client` to mutate the shared collection.
- `websockets` and `python-multipart` (for parity with other services).

## Data written to Qdrant
For every successful download the service updates:
```json
{
  "pmc_id": "123456",
  "full_text": "Title: ...\\n\\n<article body ...>",
  "full_text_timestamp": "<set in qdrant_storage during update>"
}
```
No vectors are modified; existing embeddings remain untouched.

## Operational notes
- `/api/status` avoids recounting full texts on every call for performance reasons; the counter is updated inside the worker loop.
- Skipped/failed downloads are logged but the service continues processing the remaining queue.
- `pmc_text_fetcher` exposes a `.close()` coroutine that is invoked when the background task finishes or when the event loop shuts down.
- All async work happens inside `asyncio.create_task(fetch_and_store_full_texts())`; subsequent `POST /api/start` requests are rejected while a job is running.
