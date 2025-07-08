"""
Spatial.AI Web Interface Server
Simple HTTP server for the web dashboard
"""

import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebInterfaceHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving the web interface"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Add cache-busting headers for development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def start_web_server():
    """Start the web interface server"""
    port = int(os.getenv("WEB_PORT", "8501"))
    host = os.getenv("WEB_HOST", "localhost")
    
    logger.info(f"Starting Spatial.AI Web Interface on http://{host}:{port}")
    logger.info("Make sure the API server is running on http://localhost:8000")
    
    try:
        server = HTTPServer((host, port), WebInterfaceHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Web interface server stopped")
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_web_server()
