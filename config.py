"""
Spatial.AI Production Configuration
Core LLM & Reasoning Engine System
"""

import os
from pathlib import Path

# System paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
OUTPUTS_DIR = DATA_DIR / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, VECTOR_DB_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data Source Configuration
BHOONIDHI_API_URL = os.getenv("BHOONIDHI_API_URL", "https://bhoonidhi.nrsc.gov.in/bhoonidhi/")
OSM_API_URL = os.getenv("OSM_API_URL", "https://overpass-api.de/api/interpreter")

# GIS Processing Paths
QGIS_PATH = os.getenv("QGIS_PATH")
GDAL_DATA = os.getenv("GDAL_DATA")
PROJ_LIB = os.getenv("PROJ_LIB")

# LLM Configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# RAG Configuration
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# System configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")
