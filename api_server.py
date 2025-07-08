"""
Spatial.AI Production API Server
Real-time GIS Workflow Generation System
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from spatial_ai import SpatialAISystem
from config import PROJECT_ROOT, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Spatial.AI - GIS Workflow Generation API",
    description="Real-time LLM-powered GIS workflow generation and reasoning system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web interface integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
web_interface_path = Path(__file__).parent / "web_interface"
if web_interface_path.exists():
    app.mount("/static", StaticFiles(directory=str(web_interface_path)), name="static")
    
    # Serve CSS files with correct MIME type
    @app.get("/styles.css")
    async def serve_css():
        css_file = web_interface_path / "styles.css"
        if css_file.exists():
            return FileResponse(css_file, media_type="text/css")
        else:
            raise HTTPException(status_code=404, detail="CSS file not found")
    
    # Serve JS files with correct MIME type
    @app.get("/app.js")
    async def serve_js():
        js_file = web_interface_path / "app.js"
        if js_file.exists():
            return FileResponse(js_file, media_type="application/javascript")
        else:
            raise HTTPException(status_code=404, detail="JS file not found")

@app.get("/web", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface HTML file"""
    web_file = web_interface_path / "index.html"
    if web_file.exists():
        return FileResponse(web_file)
    else:
        raise HTTPException(status_code=404, detail="Web interface not found")


# Global system instance
spatial_ai_system: Optional[SpatialAISystem] = None


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language spatial analysis query")
    context: Optional[str] = Field(None, description="Additional context for the query")
    execute_workflow: bool = Field(False, description="Execute the generated workflow")
    include_reasoning: bool = Field(True, description="Include detailed reasoning in response")
    include_workflow: bool = Field(True, description="Include workflow steps in response")


class QueryResponse(BaseModel):
    query: str
    status: str
    timestamp: str
    processing_time: Optional[float] = None
    chain_of_thought: Optional[List[Dict[str, Any]]] = None
    workflow_visualization: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    download_links: Optional[Dict[str, str]] = None
    execution_results: Optional[Dict[str, Any]] = None
    parsed_query: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    workflow: Optional[Dict[str, Any]] = None
    rag_context: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    system_ready: bool
    components: Dict[str, str]


class SystemStats(BaseModel):
    uptime: str
    total_queries_processed: int
    average_processing_time: float
    system_memory_usage: float
    vector_db_size: int


# Utility function to convert Pydantic objects to dictionaries
def _convert_to_dict(obj):
    """Convert Pydantic objects to dictionaries for JSON serialization"""
    if hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        # Convert object attributes to dict
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if hasattr(value, 'dict'):
                    result[key] = value.dict()
                elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                    result[key] = _convert_to_dict(value)
                else:
                    result[key] = value
        return result
    elif isinstance(obj, list):
        return [_convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_to_dict(value) for key, value in obj.items()}
    else:
        return str(obj) if obj is not None else None


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize the Spatial.AI system on startup"""
    global spatial_ai_system
    try:
        logger.info("Initializing Spatial.AI system...")
        spatial_ai_system = SpatialAISystem()
        logger.info("Spatial.AI system ready for real-time processing")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Spatial.AI system shutting down...")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Spatial.AI - GIS Workflow Generation API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "web_interface": "/web"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    try:
        if spatial_ai_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Get system status
        system_status = spatial_ai_system.get_system_status()
        
        # Convert component statuses to strings for Pydantic validation
        component_strings = {}
        if "components" in system_status:
            for component_name, component_data in system_status["components"].items():
                if isinstance(component_data, dict):
                    if "ready" in component_data:
                        component_strings[component_name] = "operational" if component_data["ready"] else "degraded"
                    elif "available" in component_data:
                        component_strings[component_name] = "operational" if component_data["available"] else "unavailable"
                    else:
                        component_strings[component_name] = "operational"
                else:
                    component_strings[component_name] = str(component_data)
        
        return HealthResponse(
            status="healthy" if system_status.get("ready", False) else "degraded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            system_ready=system_status.get("ready", False),
            components=component_strings
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    """Get detailed system status and component information"""
    try:
        if spatial_ai_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Get comprehensive system status
        system_status = spatial_ai_system.get_system_status()
        
        # Add additional status information
        status_response = {
            "system": {
                "status": system_status.get("status", "unknown"),
                "ready": system_status.get("ready", False),
                "integration": system_status.get("integration", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "components": system_status.get("components", {}),
            "api": {
                "status": "operational",
                "endpoints": {
                    "query": "operational",
                    "health": "operational",
                    "status": "operational",
                    "stats": "operational"
                }
            }
        }
        
        # Add error information if present
        if "error" in system_status:
            status_response["system"]["error"] = system_status["error"]
        
        return status_response
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_spatial_query(request: QueryRequest):
    """Process a spatial analysis query with enhanced Chain-of-Thought reasoning"""
    start_time = datetime.now()
    
    try:
        if spatial_ai_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process the query with enhanced features
        result = spatial_ai_system.process_query(
            query=request.query,
            context=request.context,
            execute_workflow=request.execute_workflow
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare enhanced response
        response_data = {
            "query": request.query,
            "status": result.get("status", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time
        }
        
        # Add Chain-of-Thought reasoning if available
        if "chain_of_thought" in result:
            response_data["chain_of_thought"] = result["chain_of_thought"]
        
        # Add workflow visualization if available
        if "workflow_visualization" in result:
            response_data["workflow_visualization"] = result["workflow_visualization"]
        
        # Add performance metrics if available
        if "performance_metrics" in result:
            response_data["performance_metrics"] = result["performance_metrics"]
        
        # Add download links if available
        if "download_links" in result:
            response_data["download_links"] = result["download_links"]
        
        # Add execution results if available
        if "execution_results" in result:
            response_data["execution_results"] = result["execution_results"]
        
        # Include traditional fields for backward compatibility
        if request.include_reasoning and "reasoning" in result:
            response_data["reasoning"] = _convert_to_dict(result["reasoning"])
        
        if request.include_workflow and "workflow" in result:
            response_data["workflow"] = _convert_to_dict(result["workflow"])
        
        if "parsed_query" in result:
            response_data["parsed_query"] = _convert_to_dict(result["parsed_query"])
        
        if "rag_context" in result:
            response_data["rag_context"] = result["rag_context"]
        
        if "error" in result:
            response_data["error"] = result["error"]
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        return QueryResponse(**response_data)
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error processing query: {e}")
        
        error_response = {
            "query": request.query,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "error": str(e),
            "chain_of_thought": [
                {
                    "step": 1,
                    "title": "Error Occurred",
                    "description": f"System error while processing query: {str(e)}",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        return QueryResponse(**error_response)


@app.get("/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    try:
        if spatial_ai_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Get basic system status
        system_status = spatial_ai_system.get_system_status()
        
        return {
            "uptime": "operational",
            "total_queries_processed": 0,  # TODO: implement tracking
            "average_processing_time": 0.0,
            "system_memory_usage": 0.0,
            "vector_db_size": 0,
            "components": system_status.get("components", {})
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/export")
async def export_workflow(workflow_data: Dict[str, Any]):
    """Export workflow to various formats (JSON, YAML)"""
    try:
        # Implementation for workflow export
        return {"message": "Workflow export functionality - to be implemented"}
    except Exception as e:
        logger.error(f"Error exporting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def api_process_query(request: dict):
    """API endpoint for processing queries from web interface"""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    try:
        if spatial_ai_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        logger.info(f"API processing query: {query[:100]}...")
        
        # Process the query with timeout using thread pool
        def sync_process():
            return spatial_ai_system.process_query(
                query=query,
                context=request.get("context"),
                execute_workflow=request.get("execute_workflow", False)
            )
        
        try:
            # 30 second timeout
            with ThreadPoolExecutor() as executor:
                future = executor.submit(sync_process)
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future), 
                    timeout=30.0
                )
                
        except asyncio.TimeoutError:
            logger.error("Query processing timed out after 30 seconds")
            return {
                "status": "error",
                "error": "Processing timeout",
                "message": "Query processing took too long"
            }
        
        logger.info("API query processing completed successfully")
        return {
            "status": "success",
            "data": result,
            "message": "Query processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API query processing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Processing failed"
        }

@app.get("/api/status")
async def api_get_status():
    """API endpoint for system status"""
    try:
        if spatial_ai_system:
            status = spatial_ai_system.get_system_status()
            return {
                "status": "healthy",
                "system_ready": True,
                "components": {
                    "spatial_ai": True,
                    "web_interface": True
                },
                "system_status": status
            }
        else:
            return {
                "status": "error",
                "system_ready": False,
                "components": {
                    "spatial_ai": False,
                    "web_interface": True
                }
            }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "system_ready": False
        }


if __name__ == "__main__":
    # Production server configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"Starting Spatial.AI API Server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,  # Set to False for production
        log_level=LOG_LEVEL.lower()
    )
