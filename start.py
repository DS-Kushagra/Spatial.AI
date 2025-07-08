"""
Spatial.AI - Complete System Startup
====================================

Single script to run the complete Spatial.AI system:
- LLM & Reasoning Engine (Member 1 deliverable)
- GIS Engine & Web Interface (Member 2 deliverable)
- Chain-of-Thought reasoning
- Natural language to GIS workflow conversion
- Interactive web dashboard

Usage: python start.py
Access: http://localhost:8000
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Configure logging without emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import groq
        import fastapi
        import uvicorn
        import geopandas
        import rasterio
        logger.info("All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment variables"""
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        logger.error(".env file not found")
        logger.error("Please create a .env file with your API keys")
        return False
    
    # Check GROQ_API_KEY
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("GROQ_API_KEY environment variable not set")
        logger.error("Please check your .env file and ensure GROQ_API_KEY is properly set")
        logger.error("Current .env file exists but may have syntax issues")
        return False
    
    if groq_key.startswith("gsk_"):
        logger.info(f"GROQ_API_KEY found (starts with: {groq_key[:8]}...)")
    else:
        logger.warning("GROQ_API_KEY found but doesn't start with expected 'gsk_' prefix")
    
    logger.info("Environment variables are properly configured")
    return True

def start_system():
    """Start the complete Spatial.AI system"""
    logger.info("=" * 60)
    logger.info("Starting Spatial.AI - Complete GIS Workflow Generation System")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_requirements():
        return False
    
    if not check_environment():
        return False
    
    logger.info("All checks passed - starting system...")
    logger.info("")
    logger.info("Features included:")
    logger.info("- Chain-of-Thought LLM reasoning")
    logger.info("- Natural language to GIS workflow conversion")
    logger.info("- Interactive web dashboard")
    logger.info("- Real-time query processing")
    logger.info("- JSON/YAML workflow export")
    logger.info("")
    logger.info("Access the system at: http://localhost:8000")
    logger.info("API documentation at: http://localhost:8000/docs")
    logger.info("")
    logger.info("Press Ctrl+C to stop the system")
    logger.info("=" * 60)
    
    try:
        # Start the API server
        import uvicorn
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System failed to start: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = start_system()
    if not success:
        sys.exit(1)
