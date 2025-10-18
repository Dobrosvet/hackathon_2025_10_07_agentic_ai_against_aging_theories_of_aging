# Full Text Downloader Microservice

This microservice downloads full texts of articles from PubMed Central and stores them in Qdrant database.

## Features

- Reads URLs from Qdrant database (populated by data_a service)
- Downloads full text content from PMC via E-utilities API
- Updates existing records in Qdrant with `full_text` field
- Batch processing with rate limiting
- Resume support (skips articles that already have full text)
- Real-time progress tracking via WebSocket
- RESTful API for control

## API Endpoints

- `GET /` - Health check
- `GET /api/status` - Get current service status
- `POST /api/start` - Start downloading full texts
- `POST /api/stop` - Stop downloading
- `WS /ws` - WebSocket for real-time updates

## Installation

```bash
cd data_b_get_full_texts
poetry install
```

## Usage

```bash
poetry run python main.py
```

The service will run on `http://127.0.0.1:8003`

## Requirements

- Python 3.10+
- Running Qdrant instance on localhost:6333
- Populated database from data_a_get_urls_list_papers service
