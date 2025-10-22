import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import yaml

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from aging_theory_classifier import AgingTheoryClassifier
from qdrant_storage import QdrantStorage

# Pydantic models for request validation
class LabelRequest(BaseModel):
    label: bool
    comment: str = ""

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "classifier": {"mode": "keyword", "use_gpu": False, "batch_size": 32},
            "api": {"host": "127.0.0.1", "port": 8004},
        }

    classifier_cfg = config.setdefault("classifier", {})
    variant_from_env = os.getenv("AGING_CLASSIFIER_MODEL_VARIANT")
    if variant_from_env:
        classifier_cfg["model_variant"] = variant_from_env

    variants = classifier_cfg.get("model_variants", {})
    active_variant = classifier_cfg.get("model_variant")
    if active_variant and active_variant in variants:
        variant_cfg = variants[active_variant] or {}
        for key, value in variant_cfg.items():
            if key == "generation_config":
                continue
            classifier_cfg[key] = value

    if "mode" not in classifier_cfg:
        classifier_cfg["mode"] = "embedding"
    if "embedding_model_name" not in classifier_cfg and "embedding_model" not in classifier_cfg:
        classifier_cfg["embedding_model"] = "pritamdeka/S-PubMedBert-MS-MARCO"
    if "llm_model" not in classifier_cfg:
        classifier_cfg["llm_model"] = "gpt-4o-mini"

    return config

config = load_config()

# Setup paths - абсолютный путь к корневой data
# Получить корневую директорию проекта (2 уровня вверх от текущего файла)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
LOGS_DIR = DATA_DIR / "logs"

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"classifier_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
app = FastAPI(title="Aging Theory Classifier & NER Service")

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
    "classified_as_theory": 0,
    "classified_as_not_theory": 0,
    "total_theories_found": 0,
    "errors": 0,
    "saved": 0,
    "skipped": 0,
    "db_count": 0,
    "logs": [],
    "start_time": None,
    "message": "",
    # For real-time UI preview
    "current_text_preview": "",
    "current_highlighted_spans": []
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
classifier_config = config.get("classifier", {})
classifier = AgingTheoryClassifier(
    mode=classifier_config.get("mode", "keyword"),
    use_gpu=classifier_config.get("use_gpu", False),
    bioformer_threshold=classifier_config.get("bioformer_threshold", classifier_config.get("embedding_threshold", 0.6)),
    quantize=classifier_config.get("quantize", False),
    embedding_model_name=classifier_config.get("embedding_model") or classifier_config.get("embedding_model_name", "pritamdeka/S-PubMedBert-MS-MARCO"),
    llm_model_name=classifier_config.get("llm_model", "gpt-4o-mini"),
    llm_temperature=classifier_config.get("llm_temperature", 0.0),
    llm_max_tokens=classifier_config.get("llm_max_tokens", 256),
    openai_api_key=classifier_config.get("openai_api_key")
)
qdrant_storage = QdrantStorage(str(DB_DIR))

# Log classifier info
logger.info(
    "Classifier configured: mode=%s variant=%s",
    classifier_config.get("mode", "embedding"),
    classifier_config.get("model_variant", "pubmedbert_sbert"),
)
logger.info(f"Classifier info: {classifier.get_model_info()}")


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
            "classified_as_theory": service_state["classified_as_theory"],
            "classified_as_not_theory": service_state["classified_as_not_theory"],
            "total_theories_found": service_state["total_theories_found"],
            "errors": service_state["errors"],
            "saved": service_state["saved"],
            "skipped": service_state["skipped"],
            "db_count": service_state["db_count"],
            "start_time": service_state["start_time"],
            "message": service_state["message"],
            "current_text_preview": service_state["current_text_preview"],
            "current_highlighted_spans": service_state["current_highlighted_spans"]
        }
    })


async def classify_papers():
    """Main worker function for classifying papers"""
    try:
        service_state["status"] = "running"
        service_state["start_time"] = datetime.now().isoformat()
        service_state["progress"] = 0
        service_state["errors"] = 0
        service_state["saved"] = 0
        service_state["skipped"] = 0
        service_state["classified_as_theory"] = 0
        service_state["classified_as_not_theory"] = 0
        service_state["total_theories_found"] = 0
        service_state["message"] = ""

        await log_and_broadcast("Starting aging theory classification...")
        await broadcast_state()

        # Get collection info
        db_info = qdrant_storage.get_collection_info()
        total_papers = db_info.get("count", 0)
        await log_and_broadcast(f"Database contains {total_papers} papers")

        # Get papers without classification
        await log_and_broadcast("Loading papers without classification...")
        papers_to_process = qdrant_storage.get_papers_without_classification()

        service_state["total"] = total_papers
        service_state["skipped"] = total_papers - len(papers_to_process)
        service_state["db_count"] = service_state["skipped"]

        if not papers_to_process:
            service_state["status"] = "completed"
            service_state["message"] = "All papers already classified! Database is up to date."
            await log_and_broadcast("✓ All papers already classified!")
            await broadcast_state()
            return

        await log_and_broadcast(f"Found {len(papers_to_process)} papers to classify")
        await log_and_broadcast(f"Already classified: {service_state['skipped']}")
        await broadcast_state()

        # Process papers one by one
        BATCH_SIZE = 10  # Update database every 10 papers

        batch_updates = []

        for idx, paper in enumerate(papers_to_process):
            if service_state["status"] != "running":
                await log_and_broadcast("Classification stopped by user", "WARNING")
                break

            pmc_id = paper["pmc_id"]
            point_id = paper["id"]
            title = paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"]
            full_text = paper["full_text"]

            service_state["current_paper"] = f"PMC{pmc_id}: {title}"
            service_state["progress"] = service_state["skipped"] + idx + 1

            # Preview text (first 2000 characters for UI)
            service_state["current_text_preview"] = full_text[:2000] if len(full_text) > 2000 else full_text
            service_state["current_highlighted_spans"] = []  # Will be updated after classification

            await broadcast_state()

            try:
                await log_and_broadcast(f"Classifying PMC{pmc_id}...")

                # Classify and extract theories
                classification_result = classifier.process_paper(full_text)

                # Update UI with highlighted spans
                if classification_result["aging_theories"]:
                    # Only show first 10 theories in preview (for performance)
                    service_state["current_highlighted_spans"] = classification_result["aging_theories"][:10]

                # Add to batch
                batch_updates.append({
                    "point_id": point_id,
                    "pmc_id": pmc_id,
                    "classification_result": classification_result
                })

                # Update counters
                if classification_result["is_aging_theory"]:
                    service_state["classified_as_theory"] += 1
                else:
                    service_state["classified_as_not_theory"] += 1

                service_state["total_theories_found"] += len(classification_result["aging_theories"])

                # If batch is full, update database
                if len(batch_updates) >= BATCH_SIZE:
                    updated_count = await qdrant_storage.update_papers_batch(batch_updates)
                    service_state["saved"] += updated_count
                    service_state["db_count"] += updated_count

                    await log_and_broadcast(
                        f"✓ Saved batch: {updated_count} papers "
                        f"(Total: {service_state['saved']}/{len(papers_to_process)}, "
                        f"Theory: {service_state['classified_as_theory']}, "
                        f"Not Theory: {service_state['classified_as_not_theory']})"
                    )

                    batch_updates = []
                    await broadcast_state()

            except Exception as e:
                service_state["errors"] += 1
                await log_and_broadcast(f"Error processing PMC{pmc_id}: {str(e)}", "ERROR")

            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

        # Save remaining batch
        if batch_updates:
            updated_count = await qdrant_storage.update_papers_batch(batch_updates)
            service_state["saved"] += updated_count
            service_state["db_count"] += updated_count

            await log_and_broadcast(f"✓ Saved final batch: {updated_count} papers")

        service_state["status"] = "completed"
        service_state["message"] = (
            f"✓ Classification complete! "
            f"Theory papers: {service_state['classified_as_theory']}, "
            f"Non-theory: {service_state['classified_as_not_theory']}, "
            f"Total theories found: {service_state['total_theories_found']}"
        )
        await log_and_broadcast(
            f"✓ Classification complete! Classified: {service_state['saved']}, "
            f"Theory: {service_state['classified_as_theory']}, "
            f"Not Theory: {service_state['classified_as_not_theory']}, "
            f"Errors: {service_state['errors']}"
        )
        await log_and_broadcast(f"✓ Total theories found: {service_state['total_theories_found']}")
        await log_and_broadcast(f"✓ Total classified papers in DB: {service_state['db_count']}")
        await broadcast_state()

    except Exception as e:
        service_state["status"] = "error"
        service_state["message"] = f"Error: {str(e)}"
        await log_and_broadcast(f"Fatal error: {str(e)}", "ERROR")
        await broadcast_state()
        logger.exception(e)


async def load_classified_count_async():
    """Загрузить количество классифицированных статей асинхронно"""
    try:
        await asyncio.sleep(2)  # Подождать немного после старта
        classified_count = qdrant_storage.get_classified_papers_count()
        service_state["db_count"] = classified_count
        service_state["skipped"] = classified_count
        logger.info(f"Loaded classified papers count: {classified_count}")
        await broadcast_state()
    except Exception as e:
        logger.error(f"Error loading classified count: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        await qdrant_storage.initialize()
        # Загружаем счетчик асинхронно, чтобы не блокировать старт
        asyncio.create_task(load_classified_count_async())
        service_state["db_count"] = 0  # Временно 0, обновится асинхронно
        logger.info(f"Database connection initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


@app.get("/")
async def root():
    return {"service": "Aging Theory Classifier & NER Service", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current service status"""
    return {
        "status": service_state["status"],
        "progress": service_state["progress"],
        "total": service_state["total"],
        "current_paper": service_state["current_paper"],
        "classified_as_theory": service_state["classified_as_theory"],
        "classified_as_not_theory": service_state["classified_as_not_theory"],
        "total_theories_found": service_state["total_theories_found"],
        "errors": service_state["errors"],
        "saved": service_state["saved"],
        "skipped": service_state["skipped"],
        "db_count": service_state["db_count"],
        "start_time": service_state["start_time"],
        "message": service_state["message"],
        "current_text_preview": service_state["current_text_preview"],
        "current_highlighted_spans": service_state["current_highlighted_spans"]
    }


@app.post("/api/start")
async def start_classification():
    """Start classifying papers"""
    if service_state["status"] == "running":
        return {"error": "Service already running"}

    # Start background task
    asyncio.create_task(classify_papers())

    return {"message": "Classification started"}


@app.post("/api/stop")
async def stop_classification():
    """Stop classifying papers"""
    if service_state["status"] != "running":
        return {"error": "Service not running"}

    service_state["status"] = "stopped"
    await log_and_broadcast("Classification stopped by user request", "WARNING")

    return {"message": "Classification stopped"}


@app.get("/api/paper/{pmc_id}")
async def get_paper_classification(pmc_id: str):
    """Get classification results for a specific paper"""
    try:
        paper = qdrant_storage.get_paper_by_pmc_id(pmc_id)

        if not paper:
            return {"error": f"Paper PMC{pmc_id} not found"}

        return {
            "pmc_id": paper.get("pmc_id"),
            "title": paper.get("title"),
            "is_aging_theory": paper.get("is_aging_theory"),
            "classification_confidence": paper.get("classification_confidence"),
            "aging_theories": paper.get("aging_theories", []),
            "classification_timestamp": paper.get("classification_timestamp"),
            "full_text": paper.get("full_text", "")
        }

    except Exception as e:
        logger.error(f"Error getting paper {pmc_id}: {e}")
        return {"error": str(e)}


# ============ MANUAL REVIEW ENDPOINTS ============

@app.get("/api/papers/unreviewed")
async def get_unreviewed_papers(limit: int = 50, offset: int = 0):
    """
    Get papers that need manual review (have full text but no manual label)

    Query params:
        limit: Maximum number of papers to return (default: 50, max: 100)
        offset: Number of papers to skip (default: 0)
    """
    try:
        # Validate parameters
        limit = min(max(1, limit), 100)  # Between 1 and 100
        offset = max(0, offset)

        papers = qdrant_storage.get_unreviewed_papers(limit=limit, offset=offset)

        return {
            "papers": papers,
            "count": len(papers),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error getting unreviewed papers: {e}")
        return {"error": str(e), "papers": []}


@app.post("/api/papers/{pmc_id}/label")
async def label_paper(pmc_id: str, request: LabelRequest):
    """
    Save manual label for a paper

    Body params:
        label: True if aging theory, False if not
        comment: Optional user comment
    """
    try:
        success = await qdrant_storage.save_manual_label(pmc_id, request.label, request.comment)

        if success:
            await log_and_broadcast(
                f"Manual label saved for PMC{pmc_id}: {'theory' if request.label else 'not theory'}"
            )
            return {
                "success": True,
                "message": f"Label saved for PMC{pmc_id}",
                "pmc_id": pmc_id,
                "label": request.label
            }
        else:
            return {
                "success": False,
                "error": f"Failed to save label for PMC{pmc_id}"
            }

    except Exception as e:
        logger.error(f"Error labeling paper {pmc_id}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/papers/labeled")
async def get_labeling_statistics():
    """Get statistics about manual labeling"""
    try:
        stats = qdrant_storage.get_labeling_stats()
        return {
            "success": True,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error getting labeling stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "statistics": {
                "total_labeled": 0,
                "labeled_as_theory": 0,
                "labeled_as_not_theory": 0,
                "skipped": 0,
                "unreviewed": 0
            }
        }


@app.post("/api/papers/{pmc_id}/skip")
async def skip_paper_review(pmc_id: str):
    """Mark paper as skipped for manual review"""
    try:
        success = await qdrant_storage.skip_paper(pmc_id)

        if success:
            await log_and_broadcast(f"Paper PMC{pmc_id} marked as skipped")
            return {
                "success": True,
                "message": f"Paper PMC{pmc_id} skipped",
                "pmc_id": pmc_id
            }
        else:
            return {
                "success": False,
                "error": f"Failed to skip PMC{pmc_id}"
            }

    except Exception as e:
        logger.error(f"Error skipping paper {pmc_id}: {e}")
        return {"success": False, "error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send current state immediately
        await websocket.send_json({
            "type": "state",
            "data": {
                "status": service_state["status"],
                "progress": service_state["progress"],
                "total": service_state["total"],
                "current_paper": service_state["current_paper"],
                "classified_as_theory": service_state["classified_as_theory"],
                "classified_as_not_theory": service_state["classified_as_not_theory"],
                "total_theories_found": service_state["total_theories_found"],
                "errors": service_state["errors"],
                "saved": service_state["saved"],
                "skipped": service_state["skipped"],
                "db_count": service_state["db_count"],
                "start_time": service_state["start_time"],
                "message": service_state["message"],
                "current_text_preview": service_state["current_text_preview"],
                "current_highlighted_spans": service_state["current_highlighted_spans"]
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
    api_config = config.get("api", {})
    uvicorn.run(
        app,
        host=api_config.get("host", "127.0.0.1"),
        port=api_config.get("port", 8004),
        reload=api_config.get("reload", False),
        workers=api_config.get("workers", 1)
    )
