import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from questions_classifier_v2 import QuestionsClassifierV2
from qdrant_storage import QdrantStorage
from validation_metrics import ValidationMetrics

# Pydantic models
class AnnotationRequest(BaseModel):
    annotation_type: str  # Q1, Q2, ..., C1, C2, ...
    answer: Any  # True/False или строка для Q1
    fragment: Dict[str, Any]  # {text, start_position, end_position}

class DeleteAnnotationRequest(BaseModel):
    annotation_index: int

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "classifier": {"use_gpu": False, "batch_size": 32},
            "api": {"host": "127.0.0.1", "port": 8005}
        }

    classifier_cfg = config.setdefault("classifier", {})
    variant_from_env = os.getenv("QUESTIONS_CLASSIFIER_MODEL_VARIANT")
    if variant_from_env:
        classifier_cfg["model_variant"] = variant_from_env

    variants = classifier_cfg.get("model_variants", {})
    active_variant = classifier_cfg.get("model_variant")
    if active_variant and active_variant in variants:
        variant_cfg = variants[active_variant] or {}

        for key in ("model_name", "provider", "approach", "use_gpu", "quantize"):
            if key in variant_cfg:
                classifier_cfg[key] = variant_cfg[key]

        if "generation_config" in variant_cfg and variant_cfg["generation_config"] is not None:
            classifier_cfg["generation_config"] = variant_cfg["generation_config"]
        elif "generation_config" in classifier_cfg and classifier_cfg.get("approach") != "llm_generation":
            classifier_cfg.pop("generation_config", None)

    if "model_name" not in classifier_cfg:
        classifier_cfg["model_name"] = "pritamdeka/S-PubMedBert-MS-MARCO"
    if "provider" not in classifier_cfg:
        classifier_cfg["provider"] = "huggingface"
    if "approach" not in classifier_cfg:
        classifier_cfg["approach"] = "sentence_bert"

    return config

config = load_config()

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
LOGS_DIR = DATA_DIR / "logs"

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"questions_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
logger.propagate = False
if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# FastAPI app
app = FastAPI(title="Questions and Criterias Classifier Service")

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
    "status": "idle",
    "progress": 0,
    "total": 0,
    "current_paper": "",
    "classified_count": 0,
    "errors": 0,
    "saved": 0,
    "skipped": 0,
    "db_count": 0,
    "logs": [],
    "start_time": None,
    "message": "",
    "current_text_preview": "",
    "current_classification": None
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
classifier = QuestionsClassifierV2(
    config_path=str(Path(__file__).parent / "config.yaml"),
    model_name=classifier_config.get("model_name"),
    approach=classifier_config.get("approach", "auto"),
    use_gpu=classifier_config.get("use_gpu", False),
    quantize=classifier_config.get("quantize", False),
    requires_auth=classifier_config.get("requires_auth"),
    random_seed=classifier_config.get("random_seed")
)
qdrant_storage = QdrantStorage(str(DB_DIR))
validation_metrics = ValidationMetrics()

logger.info(
    "Classifier initialized with variant '%s' (%s)",
    classifier_config.get("model_variant", "pubmedbert_sbert"),
    classifier_config.get("model_name"),
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

    if len(service_state["logs"]) > 1000:
        service_state["logs"] = service_state["logs"][-1000:]

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
            "classified_count": service_state["classified_count"],
            "errors": service_state["errors"],
            "saved": service_state["saved"],
            "skipped": service_state["skipped"],
            "db_count": service_state["db_count"],
            "start_time": service_state["start_time"],
            "message": service_state["message"],
            "current_text_preview": service_state["current_text_preview"],
            "current_classification": service_state["current_classification"]
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
        service_state["classified_count"] = 0
        service_state["message"] = ""

        await log_and_broadcast("Starting questions classification...")
        await broadcast_state()

        # Get papers with aging theories
        all_papers = qdrant_storage.get_papers_with_aging_theories()
        total_papers = len(all_papers)
        await log_and_broadcast(f"Found {total_papers} papers with aging theories")

        # Get papers without classification
        papers_to_process = qdrant_storage.get_papers_without_questions_classification()

        service_state["total"] = total_papers
        service_state["skipped"] = total_papers - len(papers_to_process)
        service_state["db_count"] = service_state["skipped"]

        if not papers_to_process:
            service_state["status"] = "completed"
            service_state["message"] = "All papers already classified!"
            await log_and_broadcast("All papers already classified!")
            await broadcast_state()
            return

        await log_and_broadcast(f"Found {len(papers_to_process)} papers to classify")
        await broadcast_state()

        # Process papers
        BATCH_SIZE = 10

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

            # Preview text
            service_state["current_text_preview"] = full_text[:2000] if len(full_text) > 2000 else full_text

            await broadcast_state()

            try:
                await log_and_broadcast(f"Classifying PMC{pmc_id}...")

                # Classify
                classification_result = classifier.classify_paper(full_text)

                # Update UI
                service_state["current_classification"] = classification_result

                # Add to batch
                batch_updates.append({
                    "point_id": point_id,
                    "pmc_id": pmc_id,
                    "classification_result": classification_result
                })

                service_state["classified_count"] += 1

                # Save batch
                if len(batch_updates) >= BATCH_SIZE:
                    updated_count = await qdrant_storage.update_papers_batch(batch_updates)
                    service_state["saved"] += updated_count
                    service_state["db_count"] += updated_count

                    await log_and_broadcast(
                        f"Saved batch: {updated_count} papers "
                        f"(Total: {service_state['saved']}/{len(papers_to_process)})"
                    )

                    batch_updates = []
                    await broadcast_state()

            except Exception as e:
                service_state["errors"] += 1
                await log_and_broadcast(f"Error processing PMC{pmc_id}: {str(e)}", "ERROR")

            await asyncio.sleep(0.1)

        # Save remaining
        if batch_updates:
            updated_count = await qdrant_storage.update_papers_batch(batch_updates)
            service_state["saved"] += updated_count
            service_state["db_count"] += updated_count
            await log_and_broadcast(f"Saved final batch: {updated_count} papers")

        service_state["status"] = "completed"
        service_state["message"] = f"Classification complete! Classified: {service_state['saved']}"
        await log_and_broadcast(f"Classification complete! Papers classified: {service_state['saved']}")
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

        # Загрузить текущее количество классифицированных статей
        classified_count = qdrant_storage.get_classified_papers_count()
        service_state["db_count"] = classified_count
        service_state["saved"] = classified_count

        logger.info(f"Database connection initialized. Found {classified_count} classified papers")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


@app.get("/")
async def root():
    return {"service": "Questions and Criterias Classifier Service", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current service status"""
    return service_state


@app.post("/api/start")
async def start_classification():
    """Start classifying papers"""
    if service_state["status"] == "running":
        return {"error": "Service already running"}

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
async def get_paper(pmc_id: str):
    """Get paper with classification results"""
    try:
        paper = qdrant_storage.get_paper_by_pmc_id(pmc_id)

        if not paper:
            return {"error": f"Paper PMC{pmc_id} not found"}

        return {
            "pmc_id": paper.get("pmc_id"),
            "title": paper.get("title"),
            "full_text": paper.get("full_text", ""),
            "is_aging_theory": paper.get("is_aging_theory"),
            "questions_classification": paper.get("questions_classification"),
            "criteria_classification": paper.get("criteria_classification"),
            "manual_annotations": paper.get("manual_annotations", [])
        }

    except Exception as e:
        logger.error(f"Error getting paper {pmc_id}: {e}")
        return {"error": str(e)}


@app.post("/api/paper/{pmc_id}/annotate")
async def add_annotation(pmc_id: str, request: AnnotationRequest):
    """Add manual annotation to paper"""
    try:
        annotation = {
            "type": request.annotation_type,
            "answer": request.answer,
            "fragment": request.fragment
        }

        success = await qdrant_storage.save_manual_annotation(pmc_id, annotation)

        if success:
            await log_and_broadcast(f"Manual annotation saved for PMC{pmc_id}: {request.annotation_type}")
            return {
                "success": True,
                "message": f"Annotation saved for PMC{pmc_id}"
            }
        else:
            return {"success": False, "error": "Failed to save annotation"}

    except Exception as e:
        logger.error(f"Error saving annotation for {pmc_id}: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/paper/{pmc_id}/delete_annotation")
async def delete_annotation(pmc_id: str, request: DeleteAnnotationRequest):
    """Delete manual annotation"""
    try:
        success = await qdrant_storage.delete_manual_annotation(pmc_id, request.annotation_index)

        if success:
            await log_and_broadcast(f"Annotation deleted for PMC{pmc_id}")
            return {"success": True, "message": "Annotation deleted"}
        else:
            return {"success": False, "error": "Failed to delete annotation"}

    except Exception as e:
        logger.error(f"Error deleting annotation: {e}")
        return {"success": False, "error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send current state
        await websocket.send_json({
            "type": "state",
            "data": service_state
        })

        # Send recent logs
        await websocket.send_json({
            "type": "logs",
            "data": service_state["logs"][-100:]
        })

        # Keep connection alive
        while True:
            data = await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# Validation Endpoints
# ============================================================================

@app.get("/api/validation/papers")
async def get_validation_papers():
    """Get all validation papers with manual annotations"""
    try:
        papers = qdrant_storage.get_validation_papers()

        # Подготовить данные для отображения
        result = []
        for paper in papers:
            result.append({
                "paper_url": paper.get("paper_url"),
                "title": paper.get("validation_paper_name") or paper.get("title"),
                "year": paper.get("validation_paper_year") or paper.get("year"),
                "is_manually_annotated": paper.get("is_manually_annotated", False),
                "has_predictions": paper.get("questions_classification") is not None,
                "validation_timestamp": paper.get("validation_timestamp"),
                "questions_timestamp": paper.get("questions_timestamp")
            })

        return {
            "papers": result,
            "count": len(result)
        }

    except Exception as e:
        logger.error(f"Error getting validation papers: {e}")
        return {"error": str(e), "papers": [], "count": 0}


@app.get("/api/validation/metrics")
async def get_validation_metrics():
    """Calculate and return validation metrics"""
    try:
        # Получить валидационные статьи
        validation_papers = qdrant_storage.get_validation_papers()

        if not validation_papers:
            return {
                "error": "No validation papers found",
                "overall": {},
                "per_question": {},
                "paper_comparisons": [],
                "num_papers": 0
            }

        # Рассчитать метрики
        metrics = validation_metrics.calculate_validation_metrics(validation_papers)

        # Форматировать для UI
        formatted_metrics = validation_metrics.format_metrics_for_display(metrics)

        logger.info(f"Calculated validation metrics for {len(validation_papers)} papers")

        return formatted_metrics

    except Exception as e:
        logger.error(f"Error calculating validation metrics: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/validation/classify")
async def classify_validation_papers():
    """Classify all validation papers and update predictions"""
    try:
        # Получить валидационные статьи
        validation_papers = qdrant_storage.get_validation_papers()

        if not validation_papers:
            return {"error": "No validation papers found"}

        await log_and_broadcast(f"Starting classification of {len(validation_papers)} validation papers...")

        classified_count = 0
        errors = 0

        for paper in validation_papers:
            try:
                paper_url = paper.get("paper_url", "")
                full_text = paper.get("full_text", "")
                point_id = paper.get("id")

                if not full_text:
                    logger.warning(f"Paper {paper_url} has no full text")
                    continue

                await log_and_broadcast(f"Classifying {paper_url}...")

                # Классифицировать
                classification_result = classifier.classify_paper(full_text)

                # Обновить в БД
                success = await qdrant_storage.update_paper_with_questions_classification(
                    point_id=point_id,
                    pmc_id=paper.get("paper_url", ""),  # Используем URL как ID
                    classification_result=classification_result
                )

                if success:
                    classified_count += 1
                    await log_and_broadcast(f"✓ Classified {paper_url}")
                else:
                    errors += 1
                    await log_and_broadcast(f"✗ Failed to save {paper_url}", "ERROR")

            except Exception as e:
                errors += 1
                logger.error(f"Error classifying paper {paper.get('paper_url')}: {e}")
                await log_and_broadcast(f"Error: {str(e)}", "ERROR")

        await log_and_broadcast(
            f"Classification complete! Classified: {classified_count}, Errors: {errors}"
        )

        return {
            "success": True,
            "classified_count": classified_count,
            "errors": errors,
            "total": len(validation_papers)
        }

    except Exception as e:
        logger.error(f"Error in validation classification: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/validation/comparison/{paper_url:path}")
async def get_validation_comparison(paper_url: str):
    """Get detailed comparison for a specific validation paper"""
    try:
        paper = qdrant_storage.get_validation_paper_by_url(paper_url)

        if not paper:
            return {"error": f"Paper not found: {paper_url}"}

        # Подготовить детальное сравнение
        comparison = {
            "paper_url": paper.get("paper_url"),
            "title": paper.get("validation_paper_name") or paper.get("title"),
            "year": paper.get("validation_paper_year") or paper.get("year"),
            "questions": {}
        }

        for i in range(1, 10):
            question_id = f"Q{i}"

            # Ручная разметка
            manual = paper.get("validation_questions", {}).get(question_id)

            # Предсказание модели
            model = None
            questions_classification = paper.get("questions_classification", {})
            if isinstance(questions_classification, dict):
                question_result = questions_classification.get(question_id)
                if question_result and isinstance(question_result, dict):
                    model = question_result.get("answer")

            # Нормализовать
            manual_normalized = validation_metrics.normalize_answer(manual) if manual is not None else None
            model_normalized = validation_metrics.normalize_answer(model) if model is not None else None

            comparison["questions"][question_id] = {
                "manual": manual_normalized,
                "predicted": model_normalized,
                "match": manual_normalized == model_normalized if (manual_normalized and model_normalized) else None
            }

        return comparison

    except Exception as e:
        logger.error(f"Error getting comparison for {paper_url}: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    api_config = config.get("api", {})
    uvicorn.run(
        app,
        host=api_config.get("host", "127.0.0.1"),
        port=api_config.get("port", 8005),
        reload=api_config.get("reload", False),
        workers=api_config.get("workers", 1)
    )
