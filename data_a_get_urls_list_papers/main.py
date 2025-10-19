import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pubmed_fetcher import PubMedFetcher
from qdrant_storage import QdrantStorage

# Setup paths
DATA_DIR = Path(__file__).parent.parent / "data"
DB_DIR = DATA_DIR / "db"
LOGS_DIR = DATA_DIR / "logs"

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure root logger only if not already configured
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

logger = logging.getLogger(__name__)
# Prevent propagation to avoid duplicate logs
logger.propagate = False
if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# FastAPI app
app = FastAPI(title="PubMed Central Data Fetcher")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
service_state = {
    "status": "idle",  # idle, running, stopped, error, completed
    "progress": 0,
    "total": 0,
    "current_paper": "",
    "errors": 0,
    "saved": 0,
    "skipped": 0,
    "db_count": 0,
    "logs": [],
    "start_time": None,
    "search_query": "(aging) AND (theory OR paradigm) OR Aging[MeSh] AND (ffrft[Filter])",
    "message": ""
}

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")

manager = ConnectionManager()

# Initialize components
pubmed_fetcher = PubMedFetcher()
qdrant_storage = QdrantStorage(str(DB_DIR))


async def log_and_broadcast(message: str, level: str = "INFO"):
    """Add log message and broadcast to all clients"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    service_state["logs"].append(log_entry)

    # Keep only last 1000 logs in memory
    if len(service_state["logs"]) > 1000:
        service_state["logs"] = service_state["logs"][-1000:]

    # Broadcast update
    await manager.broadcast({
        "type": "log",
        "data": log_entry
    })


async def broadcast_state():
    """Broadcast current state to all clients"""
    await manager.broadcast({
        "type": "state",
        "data": {
            "status": service_state["status"],
            "progress": service_state["progress"],
            "total": service_state["total"],
            "current_paper": service_state["current_paper"],
            "errors": service_state["errors"],
            "saved": service_state["saved"],
            "skipped": service_state["skipped"],
            "db_count": service_state["db_count"],
            "start_time": service_state["start_time"],
            "search_query": service_state["search_query"],
            "message": service_state["message"]
        }
    })


async def fetch_and_store_papers():
    """Main worker function with batch processing and resume support"""
    try:
        service_state["status"] = "running"
        service_state["start_time"] = datetime.now().isoformat()
        service_state["progress"] = 0
        service_state["errors"] = 0
        service_state["saved"] = 0
        service_state["skipped"] = 0
        service_state["message"] = ""

        await log_and_broadcast("Starting PubMed Central data fetch...")
        await broadcast_state()

        # Get current database count (Qdrant is already initialized during startup)
        db_info = qdrant_storage.get_collection_info()
        service_state["db_count"] = db_info.get("count", 0)
        await log_and_broadcast(f"Current database contains {service_state['db_count']} papers")

        # Get existing IDs for duplicate detection
        await log_and_broadcast("Loading existing papers from database...")
        existing_ids = qdrant_storage.get_existing_ids()
        await log_and_broadcast(f"Loaded {len(existing_ids)} existing paper IDs")

        # Search PubMed for ALL papers
        query = service_state["search_query"]
        await log_and_broadcast(f"Searching PubMed with query: {query}")

        paper_ids = await pubmed_fetcher.search(query)
        service_state["total"] = len(paper_ids)

        await log_and_broadcast(f"Found {len(paper_ids)} total papers")

        # Filter out existing papers
        new_paper_ids = [pid for pid in paper_ids if pid not in existing_ids]

        if not new_paper_ids:
            service_state["status"] = "completed"
            service_state["message"] = "All papers already downloaded! Database is up to date."
            await log_and_broadcast("✓ All papers already in database! No new papers to download.")
            await broadcast_state()
            return

        await log_and_broadcast(f"New papers to download: {len(new_paper_ids)}")
        await log_and_broadcast(f"Already in database: {len(existing_ids)}")
        service_state["skipped"] = len(existing_ids)
        await broadcast_state()

        # Process papers in batches - fetch 200 IDs at once, store 50 at once
        FETCH_BATCH_SIZE = 200  # Fetch 200 papers metadata in one API call
        STORE_BATCH_SIZE = 50   # Store 50 papers at once to database

        total_processed = 0
        for batch_start in range(0, len(new_paper_ids), FETCH_BATCH_SIZE):
            if service_state["status"] != "running":
                await log_and_broadcast("Fetch process stopped by user", "WARNING")
                break

            # Get batch of IDs to fetch
            batch_end = min(batch_start + FETCH_BATCH_SIZE, len(new_paper_ids))
            batch_ids = new_paper_ids[batch_start:batch_end]

            service_state["current_paper"] = f"Batch {batch_start//FETCH_BATCH_SIZE + 1}: fetching {len(batch_ids)} papers"

            try:
                # Fetch entire batch with single API call
                fetched_papers = await pubmed_fetcher.fetch_papers_batch(batch_ids)

                if fetched_papers:
                    # Store papers in smaller sub-batches to database
                    for store_start in range(0, len(fetched_papers), STORE_BATCH_SIZE):
                        store_end = min(store_start + STORE_BATCH_SIZE, len(fetched_papers))
                        papers_to_store = fetched_papers[store_start:store_end]

                        saved_count = await qdrant_storage.store_papers_batch(papers_to_store)
                        service_state["saved"] += saved_count
                        service_state["db_count"] += saved_count

                    await log_and_broadcast(f"✓ Batch saved: {len(fetched_papers)} papers (Total: {service_state['saved']}/{len(new_paper_ids)})")

                # Count errors (papers that failed to fetch)
                errors_in_batch = len(batch_ids) - len(fetched_papers)
                if errors_in_batch > 0:
                    service_state["errors"] += errors_in_batch
                    await log_and_broadcast(f"⚠ {errors_in_batch} papers failed in this batch", "WARNING")

            except Exception as e:
                service_state["errors"] += len(batch_ids)
                await log_and_broadcast(f"Error processing batch: {str(e)}", "ERROR")

            total_processed += len(batch_ids)
            service_state["progress"] = total_processed + len(existing_ids)
            await broadcast_state()

        service_state["status"] = "completed"
        service_state["message"] = f"✓ Successfully downloaded all {service_state['saved']} new papers!"
        await log_and_broadcast(f"✓ Fetch complete! New papers: {service_state['saved']}, Errors: {service_state['errors']}, Skipped: {service_state['skipped']}")
        await log_and_broadcast(f"✓ Total in database: {service_state['db_count']}")
        await broadcast_state()

    except Exception as e:
        service_state["status"] = "error"
        service_state["message"] = f"Error: {str(e)}"
        await log_and_broadcast(f"Fatal error: {str(e)}", "ERROR")
        await broadcast_state()
        logger.exception(e)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        await qdrant_storage.initialize()
        db_info = qdrant_storage.get_collection_info()
        service_state["db_count"] = db_info.get("count", 0)
        logger.info(f"Database initialized with {service_state['db_count']} papers")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


@app.get("/")
async def root():
    return {"service": "PubMed Central Data Fetcher", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current service status including database count"""
    db_info = qdrant_storage.get_collection_info()
    service_state["db_count"] = db_info.get("count", 0)

    return {
        "status": service_state["status"],
        "progress": service_state["progress"],
        "total": service_state["total"],
        "current_paper": service_state["current_paper"],
        "errors": service_state["errors"],
        "saved": service_state["saved"],
        "skipped": service_state["skipped"],
        "db_count": service_state["db_count"],
        "start_time": service_state["start_time"],
        "search_query": service_state["search_query"],
        "message": service_state["message"]
    }


@app.post("/api/start")
async def start_fetch():
    """Start fetching papers"""
    if service_state["status"] == "running":
        return {"error": "Service already running"}

    # Start background task
    asyncio.create_task(fetch_and_store_papers())

    return {"message": "Fetch started"}


@app.post("/api/stop")
async def stop_fetch():
    """Stop fetching papers"""
    if service_state["status"] != "running":
        return {"error": "Service not running"}

    service_state["status"] = "stopped"
    await log_and_broadcast("Fetch stopped by user request", "WARNING")

    return {"message": "Fetch stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Get latest database count
        db_info = qdrant_storage.get_collection_info()
        service_state["db_count"] = db_info.get("count", 0)

        # Send current state immediately
        await websocket.send_json({
            "type": "state",
            "data": {
                "status": service_state["status"],
                "progress": service_state["progress"],
                "total": service_state["total"],
                "current_paper": service_state["current_paper"],
                "errors": service_state["errors"],
                "saved": service_state["saved"],
                "skipped": service_state["skipped"],
                "db_count": service_state["db_count"],
                "start_time": service_state["start_time"],
                "search_query": service_state["search_query"],
                "message": service_state["message"]
            }
        })

        # Send recent logs
        await websocket.send_json({
            "type": "logs",
            "data": service_state["logs"][-100:]  # Send last 100 logs
        })

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
