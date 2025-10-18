import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pmc_text_fetcher import PMCTextFetcher
from qdrant_storage import QdrantStorage

# Setup paths
DATA_DIR = Path("./data")
DB_DIR = DATA_DIR / "db"
LOGS_DIR = DATA_DIR / "logs"

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"full_text_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
app = FastAPI(title="PMC Full Text Downloader")

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
pmc_fetcher = PMCTextFetcher()
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
            "message": service_state["message"]
        }
    })


def extract_pmc_id_from_url(url: str) -> str:
    """Extract PMC ID from URL (without PMC prefix)"""
    # URL format: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{id}/
    match = re.search(r'PMC(\d+)', url)
    if match:
        return match.group(1)
    return ""


async def fetch_and_store_full_texts():
    """Main worker function for downloading full texts"""
    try:
        service_state["status"] = "running"
        service_state["start_time"] = datetime.now().isoformat()
        service_state["progress"] = 0
        service_state["errors"] = 0
        service_state["saved"] = 0
        service_state["skipped"] = 0
        service_state["message"] = ""

        await log_and_broadcast("Starting full text download from PubMed Central...")
        await broadcast_state()

        # Get collection info
        db_info = qdrant_storage.get_collection_info()
        total_papers = db_info.get("count", 0)
        # Don't count papers with text here - too slow, will update during processing
        await log_and_broadcast(f"Database contains {total_papers} papers")

        # Get papers without full text
        await log_and_broadcast("Loading papers without full text...")
        papers_to_process = qdrant_storage.get_papers_without_full_text()

        service_state["total"] = total_papers
        service_state["skipped"] = total_papers - len(papers_to_process)
        service_state["db_count"] = service_state["skipped"]

        if not papers_to_process:
            service_state["status"] = "completed"
            service_state["message"] = "All papers already have full text! Database is up to date."
            await log_and_broadcast("✓ All papers already have full text!")
            await broadcast_state()
            return

        await log_and_broadcast(f"Found {len(papers_to_process)} papers to download")
        await log_and_broadcast(f"Already have full text: {service_state['skipped']}")
        await broadcast_state()

        # Process papers one by one (with rate limiting in fetcher)
        BATCH_SIZE = 10  # Update database every 10 papers

        batch_updates = []

        for idx, paper in enumerate(papers_to_process):
            if service_state["status"] != "running":
                await log_and_broadcast("Download stopped by user", "WARNING")
                break

            pmc_id = paper["pmc_id"]
            point_id = paper["id"]
            title = paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"]

            service_state["current_paper"] = f"PMC{pmc_id}: {title}"
            service_state["progress"] = service_state["skipped"] + idx + 1
            await broadcast_state()

            try:
                await log_and_broadcast(f"Downloading PMC{pmc_id}...")

                # Fetch full text
                full_text = await pmc_fetcher.fetch_full_text(pmc_id)

                if full_text:
                    # Add to batch
                    batch_updates.append({
                        "point_id": point_id,
                        "pmc_id": pmc_id,
                        "full_text": full_text
                    })

                    # If batch is full, update database
                    if len(batch_updates) >= BATCH_SIZE:
                        updated_count = await qdrant_storage.update_papers_batch(batch_updates)
                        service_state["saved"] += updated_count
                        service_state["db_count"] += updated_count

                        await log_and_broadcast(f"✓ Saved batch: {updated_count} papers (Total: {service_state['saved']}/{len(papers_to_process)})")

                        batch_updates = []
                        await broadcast_state()
                else:
                    service_state["errors"] += 1
                    await log_and_broadcast(f"⚠ Failed to get full text for PMC{pmc_id}", "WARNING")

            except Exception as e:
                service_state["errors"] += 1
                await log_and_broadcast(f"Error processing PMC{pmc_id}: {str(e)}", "ERROR")

            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.1)

        # Save remaining batch
        if batch_updates:
            updated_count = await qdrant_storage.update_papers_batch(batch_updates)
            service_state["saved"] += updated_count
            service_state["db_count"] += updated_count

            await log_and_broadcast(f"✓ Saved final batch: {updated_count} papers")

        service_state["status"] = "completed"
        service_state["message"] = f"✓ Successfully downloaded {service_state['saved']} full texts!"
        await log_and_broadcast(f"✓ Download complete! New texts: {service_state['saved']}, Errors: {service_state['errors']}")
        await log_and_broadcast(f"✓ Total papers with full text: {service_state['db_count']}")
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
        # Don't count full texts on startup - it's too slow
        # Will be counted when needed during fetch process
        service_state["db_count"] = 0
        logger.info(f"Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


@app.get("/")
async def root():
    return {"service": "PMC Full Text Downloader", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current service status including database count"""
    # Don't recalculate db_count on every status request - it's too slow
    # It will be updated during the fetch process
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
        "message": service_state["message"]
    }


@app.post("/api/start")
async def start_fetch():
    """Start downloading full texts"""
    if service_state["status"] == "running":
        return {"error": "Service already running"}

    # Start background task
    asyncio.create_task(fetch_and_store_full_texts())

    return {"message": "Full text download started"}


@app.post("/api/stop")
async def stop_fetch():
    """Stop downloading full texts"""
    if service_state["status"] != "running":
        return {"error": "Service not running"}

    service_state["status"] = "stopped"
    await log_and_broadcast("Download stopped by user request", "WARNING")

    return {"message": "Download stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send current state immediately (don't recalculate db_count - too slow)
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
    uvicorn.run(app, host="127.0.0.1", port=8003)
