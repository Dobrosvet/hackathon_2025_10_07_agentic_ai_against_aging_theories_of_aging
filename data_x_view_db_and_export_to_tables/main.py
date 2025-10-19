import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from qdrant_storage import QdrantStorage
from export_service import ExportService

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        "api": {"host": "127.0.0.1", "port": 8006},
        "qdrant": {"host": "localhost", "port": 6333}
    }

config = load_config()

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
LOGS_DIR = DATA_DIR / "logs"
EXPORTS_DIR = DATA_DIR / "exports"

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"db_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Initialize components
qdrant_url = f"http://{config.get('qdrant', {}).get('host', 'localhost')}:{config.get('qdrant', {}).get('port', 6333)}"
qdrant_storage = QdrantStorage(qdrant_url=qdrant_url)
export_service = ExportService(output_dir=str(EXPORTS_DIR))

logger.info(f"Database Viewer Service initialized")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    try:
        await qdrant_storage.initialize()
        logger.info(f"Qdrant connection initialized")

        # Load initial statistics
        service_state["statistics"] = qdrant_storage.get_statistics()
        service_state["last_update"] = datetime.now().isoformat()
        service_state["status"] = "ready"
        service_state["message"] = "Service ready"

    except Exception as e:
        logger.error(f"Error initializing service: {e}")
        service_state["status"] = "error"
        service_state["message"] = f"Initialization error: {str(e)}"

    yield

    # Shutdown
    logger.info("Service shutting down")

# FastAPI app with lifespan
app = FastAPI(title="Database Viewer and Export Service", lifespan=lifespan)

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
    "message": "",
    "logs": [],
    "statistics": None,
    "last_update": None
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
        "data": service_state
    })


@app.get("/")
async def root():
    return {
        "service": "Database Viewer and Export Service",
        "status": service_state["status"],
        "version": "1.0"
    }


@app.get("/api/status")
async def get_status():
    """Get current service status"""
    return service_state


@app.get("/api/statistics")
async def get_statistics():
    """Get statistics about the data"""
    try:
        stats = qdrant_storage.get_statistics()
        service_state["statistics"] = stats
        service_state["last_update"] = datetime.now().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tables/preview/{table_number}")
async def get_table_preview(table_number: int):
    """
    Get preview of a table (first 10 rows)

    Args:
        table_number: Table number (1, 2, or 3)
    """
    try:
        preview_limit = config.get("preview", {}).get("rows_limit", 10)

        if table_number == 1:
            data = qdrant_storage.generate_table1_theories()
            table_name = "Theories"
        elif table_number == 2:
            data = qdrant_storage.generate_table2_papers()
            table_name = "Papers"
        elif table_number == 3:
            data = qdrant_storage.generate_table3_analysis()
            table_name = "Analysis"
        else:
            raise HTTPException(status_code=400, detail="Invalid table number. Use 1, 2, or 3.")

        # Return preview
        preview_data = data[:preview_limit]

        return {
            "table_number": table_number,
            "table_name": table_name,
            "preview": preview_data,
            "total_rows": len(data),
            "showing_rows": len(preview_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table {table_number} preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tables/full/{table_number}")
async def get_table_full(
    table_number: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000)
):
    """
    Get full table with pagination

    Args:
        table_number: Table number (1, 2, or 3)
        page: Page number (1-based)
        page_size: Number of rows per page
    """
    try:
        if table_number == 1:
            data = qdrant_storage.generate_table1_theories()
            table_name = "Theories"
        elif table_number == 2:
            data = qdrant_storage.generate_table2_papers()
            table_name = "Papers"
        elif table_number == 3:
            data = qdrant_storage.generate_table3_analysis()
            table_name = "Analysis"
        else:
            raise HTTPException(status_code=400, detail="Invalid table number. Use 1, 2, or 3.")

        # Pagination
        total_rows = len(data)
        total_pages = (total_rows + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        page_data = data[start_idx:end_idx]

        return {
            "table_number": table_number,
            "table_name": table_name,
            "data": page_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table {table_number} full data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/{table_number}")
async def export_table(table_number: int, format: str = Query("csv", pattern="^(csv|xlsx)$")):
    """
    Export table to file

    Args:
        table_number: Table number (1, 2, or 3)
        format: Export format (csv or xlsx)
    """
    try:
        # Get data
        if table_number == 1:
            data = qdrant_storage.generate_table1_theories()
            table_name = "aging_theories_table1_theories"
        elif table_number == 2:
            data = qdrant_storage.generate_table2_papers()
            table_name = "aging_theories_table2_papers"
        elif table_number == 3:
            data = qdrant_storage.generate_table3_analysis()
            table_name = "aging_theories_table3_analysis"
        else:
            raise HTTPException(status_code=400, detail="Invalid table number. Use 1, 2, or 3.")

        if not data:
            raise HTTPException(status_code=404, detail="No data available for export")

        # Export
        if format == "csv":
            file_path = export_service.export_to_csv(data, table_name, f"Table {table_number}")
        else:  # xlsx
            file_path = export_service.export_to_excel(data, table_name, f"Table {table_number}")

        await log_and_broadcast(f"Exported Table {table_number} to {format.upper()}: {file_path}")

        # Return file
        return FileResponse(
            path=file_path,
            filename=Path(file_path).name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting table {table_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/all")
async def export_all_tables():
    """
    Export all 3 tables to a single Excel file with multiple sheets
    """
    try:
        # Get all tables
        table1_data = qdrant_storage.generate_table1_theories()
        table2_data = qdrant_storage.generate_table2_papers()
        table3_data = qdrant_storage.generate_table3_analysis()

        if not table1_data and not table2_data and not table3_data:
            raise HTTPException(status_code=404, detail="No data available for export")

        # Export to single Excel file
        file_path = export_service.export_all_tables_to_excel(
            table1_data,
            table2_data,
            table3_data
        )

        await log_and_broadcast(f"Exported all tables to Excel: {file_path}")

        # Return file
        return FileResponse(
            path=file_path,
            filename=Path(file_path).name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting all tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
async def refresh_data():
    """Refresh statistics and data from database"""
    try:
        service_state["statistics"] = qdrant_storage.get_statistics()
        service_state["last_update"] = datetime.now().isoformat()
        await log_and_broadcast("Data refreshed")
        await broadcast_state()
        return {"message": "Data refreshed successfully", "statistics": service_state["statistics"]}
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    api_config = config.get("api", {})
    uvicorn.run(
        app,
        host=api_config.get("host", "127.0.0.1"),
        port=api_config.get("port", 8006),
        reload=api_config.get("reload", False),
        workers=api_config.get("workers", 1)
    )
