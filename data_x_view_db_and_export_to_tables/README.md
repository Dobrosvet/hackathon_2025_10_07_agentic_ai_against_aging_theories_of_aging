# Database Viewer & Export Service

## Overview
This microservice aggregates classification results stored in the `pubmed_papers` Qdrant collection and publishes them as three tabular views. It offers REST endpoints for previews, pagination, statistics, and CSV/XLSX exports, plus a WebSocket channel for real-time updates. Exports are written to `../data/exports`.

Default port: `8006` (configurable via `config.yaml`).

## Code structure
| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI app, statistics refresh, table preview/full endpoints, export orchestration, WebSocket updates |
| `qdrant_storage.py` | Reads Qdrant payloads and assembles the three derived tables and summary statistics |
| `export_service.py` | Writes tables to CSV/Excel using pandas/openpyxl |
| `config.yaml` | API binding, Qdrant host, preview limits |
| `pyproject.toml` | Poetry project definition and dependencies |

## Derived tables
All tables are recalculated on demand from Qdrant payloads.

| Table | Contents |
| --- | --- |
| 1 — `Theories` | One row per theory (`theory_id`, `theory_name`, `number_of_collected_papers`) |
| 2 — `Papers` | Paper-level rows keyed by the first assigned theory (`theory_id`, `paper_url`, `paper_name`, `paper_year`) |
| 3 — `Analysis` | Joined view containing `theory_id`, paper metadata, and answers for Q1–Q9 and C1–C4 |

`theory_id` values are deterministic hashes of the theory name (`T0000`…`T9999`).

## API surface
### REST
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service health probe |
| `GET` | `/api/statistics` | Aggregated counts (papers, theories, coverage of questions/criteria) |
| `GET` | `/api/tables/preview/{table_number}` | First N rows of table 1/2/3 (preview size configurable; default 10) |
| `GET` | `/api/tables/full/{table_number}` | Paginated table data (`page`, `page_size` query params) |
| `POST` | `/api/export/{table_number}` | Export one table to CSV or XLSX (`format=csv|xlsx`) |
| `POST` | `/api/export/all` | Export all three tables into a single Excel workbook (one sheet per table) |
| `POST` | `/api/refresh` | Recompute statistics and broadcast the new state |

### WebSocket
`/ws` emits:
- `state` — latest statistics, message, last refresh timestamp.
- `log` / `logs` — incremental log feed (mirrors other microservices).

## Running locally
```powershell
cd data_x_view_db_and_export_to_tables
poetry install
poetry run uvicorn main:app --host 0.0.0.0 --port 8006
```

Ensure the `pubmed_papers` collection is populated by services A–D before calling the endpoints; otherwise the generated tables will be empty.

### Sample usage
```powershell
# Preview the theories table
Invoke-RestMethod http://127.0.0.1:8006/api/tables/preview/1 | ConvertTo-Json -Depth 5

# Export the analysis table to Excel
Invoke-RestMethod -Method Post http://127.0.0.1:8006/api/export/3?format=xlsx `
  -OutFile analysis_export_response.json
```

Exports land in `../data/exports` with timestamped filenames.

## Dependencies
From `pyproject.toml`:
- `fastapi`, `uvicorn`, `websockets`
- `qdrant-client`
- `pandas`, `openpyxl` (via `export_service.py`)
- `python-multipart`

Install them with `poetry install`. The service assumes UTF-8 files without BOM (enforced throughout the repository).

## Operational notes
- Table generation walks the full collection; expect increased latency on large datasets. Pagination (`page_size` ≤ 1000) helps limit payloads.
- Statistics (`/api/statistics`) are cached in `service_state` and reused for WebSocket snapshots.
- Exports apply basic column auto-sizing and include UTF-8 BOM for CSV files to preserve non-ASCII characters.
- WebSocket consumers can subscribe once and store the latest `state` snapshot for dashboards.
