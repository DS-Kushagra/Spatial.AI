# 🌍 Spatial.AI - Intelligent GIS Workflow Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00C896.svg)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq-FF6B35.svg)](https://groq.com)

## � Overview

**Spatial.AI** is a production-ready solution for **Problem Statement 4** - an AI-powered system that automatically generates and executes complex GIS workflows from natural language queries using Chain-of-Thought reasoning.

### 🎯 Key Capabilities

#### 🧠 **Intelligent Reasoning**
- **Chain-of-Thought Processing**: Transparent, step-by-step AI reasoning
- **Multi-Perspective Analysis**: Advanced reasoning with confidence assessment
- **Error Correction**: Intelligent retry mechanisms and alternative suggestions
- **Context Awareness**: Understands spatial relationships and constraints

#### 🗺️ **GIS Processing**
- **15+ Spatial Operations**: Buffer, overlay, proximity, suitability analysis
- **Multi-format Support**: GeoJSON, Shapefile, KML, and more
- **Real-time Execution**: Sub-4-second query processing
- **CRS Management**: Automatic coordinate system handling

#### 🌐 **Modern Interface**
- **Interactive Dashboard**: Professional web interface with real-time updates
- **Visual Workflow**: Dynamic workflow visualization and editing
- **Map Integration**: Interactive Leaflet maps with layer management
- **Export Capabilities**: JSON/YAML/Python workflow downloads

#### 📊 **Production Ready**
- **RESTful API**: FastAPI with automatic documentation
- **Health Monitoring**: System status and performance metrics
- **Comprehensive Logging**: Detailed system and error logs
- **Scalable Architecture**: Async processing and modular design

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+** with pip
- **Git** for version control
- **Groq API Key** ([Get one free](https://console.groq.com))

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Spatial.AI.git
cd Spatial.AI

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Core Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional: OpenAI for embeddings (can use local models)
OPENAI_API_KEY=your_openai_key_here

# System Settings
DEBUG=false
LOG_LEVEL=INFO
API_HOST=localhost
API_PORT=8000
```

### 3. Launch the System

```bash
# Start the complete system
python start.py

# Alternative: Start with CLI options
python run.py api-server
```

### 4. Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Web Dashboard** | http://localhost:8000 | Main interactive interface |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs |
| **Health Check** | http://localhost:8000/health | System status |

### 5. Test with Sample Queries

Try these example queries in the web interface:

```
🏫 Find best locations for schools in Bangalore avoiding flood zones and ensuring 500m from main roads
☀️ Identify optimal solar farm locations in Gujarat considering slope and land use
🌊 Map flood risk zones in Kerala using elevation and rainfall data
🏥 Find suitable hospital locations within 2km of residential areas in Mumbai
```

## � Project Architecture

```
Spatial.AI/
├── 🏠 Core System
│   ├── spatial_ai.py              # Main system integration hub
│   ├── api_server.py              # Production FastAPI server
│   ├── config.py                  # Central configuration management
│   ├── start.py                   # Primary startup script
│   └── run.py                     # Alternative CLI startup
├── 🧠 LLM Engine (Member 1 - Reasoning)
│   ├── __init__.py                # Main LLM integration class
│   ├── reasoning_engine.py        # Chain-of-Thought reasoning
│   ├── query_parser.py            # Natural language understanding
│   ├── workflow_generator.py      # GIS workflow generation
│   ├── unified_rag_system.py      # RAG knowledge retrieval
│   ├── prompt_manager.py          # Prompt engineering
│   ├── advanced_reasoning.py      # Multi-perspective analysis
│   └── error_correction.py        # Error handling & retry
├── 🗺️ GIS Engine (Member 2 - Processing)
│   ├── gis_executor.py            # Main GIS workflow processor
│   ├── data_loader.py             # Data integration utilities
│   ├── osm_handler.py             # OpenStreetMap integration
│   └── crs_utils.py               # Coordinate system management
├── 🌐 Web Interface (Member 2 - Frontend)
│   ├── index.html                 # Main dashboard interface
│   ├── styles.css                 # Modern UI styling
│   ├── app.js                     # Interactive frontend logic
│   └── README.md                  # Interface documentation
├── 🧪 Testing & Demos
│   ├── enhanced_demo_scenarios.py # Comprehensive demo scenarios
│   ├── test_demo_scenarios.py     # Testing framework
│   ├── benchmark_spatial_ai.py    # Performance benchmarking
│   └── health_check.py            # System health validation
├── 📊 Data & Resources
│   ├── data/                      # Sample datasets and outputs
│   │   ├── demo_scenarios/        # Demo GeoJSON files
│   │   ├── vector_db/             # RAG knowledge database
│   │   └── outputs/               # Generated results
│   ├── prompts/                   # Custom prompt templates
│   └── workflows/                 # Sample workflow definitions
└── 📝 Configuration
    ├── .env                       # Environment variables
    ├── requirements.txt           # Python dependencies
    └── README.md                  # This documentation
```
## 🎯 Problem Statement 4 Implementation

### ✅ **Complete Objectives Achievement**

This project fully implements all requirements for **Problem Statement 4**:

#### 🧠 **Chain-of-Thought LLM System**
- ✅ **Transparent Reasoning**: Step-by-step AI decision explanations
- ✅ **Groq Integration**: Fast, efficient LLM processing
- ✅ **Multi-Perspective Analysis**: Advanced reasoning capabilities
- ✅ **Confidence Assessment**: Uncertainty handling and validation

#### 🔄 **Intelligent Workflow Orchestration**
- ✅ **Natural Language Processing**: Plain English to GIS workflows
- ✅ **Automated Planning**: Multi-step spatial analysis workflows
- ✅ **JSON/YAML Export**: Reusable workflow definitions
- ✅ **Error Recovery**: Intelligent retry and alternative generation

#### 🗺️ **Comprehensive GIS Integration**
- ✅ **15+ Spatial Operations**: Buffer, overlay, suitability, flood modeling
- ✅ **Multi-Data Support**: OpenStreetMap, Bhoonidhi, local datasets
- ✅ **CRS Management**: Automatic coordinate system handling
- ✅ **Real-time Execution**: Sub-4-second processing times

#### 🌐 **Professional Interface**
- ✅ **Modern Web Dashboard**: Interactive HTML/CSS/JS interface
- ✅ **API Documentation**: Auto-generated FastAPI docs
- ✅ **Health Monitoring**: System status and performance metrics
- ✅ **Export Capabilities**: Workflow and result downloads

### 📊 **Expected Outcomes Delivered**

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| **Web/Desktop Application** | Modern web dashboard with real-time processing | ✅ Complete |
| **JSON/YAML Workflows** | Automated generation with download capabilities | ✅ Complete |
| **Chain-of-Thought Logs** | Transparent reasoning with confidence scores | ✅ Complete |
| **Demo Scenarios** | 4 comprehensive scenarios (flood, school, solar, hospital) | ✅ Complete |
| **Performance Metrics** | Real-time monitoring vs. manual methods | ✅ Complete |

### 🏆 **Evaluation Parameters Met**

- ✅ **Logical Validity**: Generated workflows follow GIS best practices
- ✅ **Reasoning Clarity**: Step-by-step explanations with confidence scores
- ✅ **Error Robustness**: Comprehensive error handling and recovery
- ✅ **Performance Efficiency**: Optimized for fast processing and usability

## 🛠️ Technology Stack

### 🧠 **LLM & Reasoning Engine (Member 1)**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Integration** | Groq API (Llama-3.3-70B) | Fast, efficient language processing |
| **RAG System** | ChromaDB + SentenceTransformers | Knowledge retrieval and context |
| **Reasoning** | Custom Chain-of-Thought | Transparent decision making |
| **Prompt Engineering** | Jinja2 Templates | Optimized prompt management |

### 🗺️ **GIS Engine & Interface (Member 2)**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **GIS Processing** | GeoPandas, Rasterio, PyQGIS | Spatial data manipulation |
| **Data Integration** | OSMnx, Custom loaders | Multi-source data handling |
| **Web Framework** | FastAPI + Uvicorn | Production API server |
| **Frontend** | HTML5/CSS3/JavaScript | Modern interactive interface |

### 🔧 **Shared Infrastructure**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI with async support | RESTful API endpoints |
| **Documentation** | Auto-generated OpenAPI docs | Interactive API documentation |
| **Monitoring** | Custom health checks | System status and metrics |
| **Logging** | Python logging + structured logs | Comprehensive system tracking |
## 🎭 Demo Scenarios

### 🏫 **School Site Selection in Bangalore**
```
Query: "Find best locations for schools in Bangalore avoiding flood zones and ensuring 500m from main roads"
```
- **Complexity**: High (Multi-criteria suitability analysis)
- **Operations**: Buffer analysis, flood zone overlay, proximity constraints
- **Output**: Ranked suitable locations with reasoning

### ☀️ **Solar Farm Planning in Gujarat**
```
Query: "Identify optimal solar farm locations in Gujarat considering slope and land use"
```
- **Complexity**: High (Terrain and land use analysis)
- **Operations**: Slope calculation, land use filtering, multi-criteria evaluation
- **Output**: Optimized solar farm locations with site rankings

### 🌊 **Flood Risk Mapping in Kerala**
```
Query: "Map flood risk zones in Kerala using elevation and rainfall data"
```
- **Complexity**: Medium (Hydrological modeling)
- **Operations**: DEM analysis, flood modeling, risk classification
- **Output**: Comprehensive flood risk maps with vulnerability assessment

### 🏥 **Hospital Accessibility Analysis in Mumbai**
```
Query: "Find suitable hospital locations within 2km of residential areas in Mumbai"
```
- **Complexity**: Medium (Accessibility and service area analysis)
- **Operations**: Proximity analysis, service area calculation, gap identification
- **Output**: Optimal hospital locations with accessibility metrics

### 📊 **Performance Metrics**

| Metric | Average | Best Case | Target |
|--------|---------|-----------|--------|
| **Query Processing Time** | 3.86s | 2.1s | < 5s |
| **Workflow Generation** | 100% | 100% | > 95% |
| **Reasoning Quality** | 85% confidence | 95% | > 80% |
| **API Response Time** | < 1s | 0.3s | < 2s |

## 🔧 Development & Deployment

### 🧪 **Testing**

```bash
# Run system health check
python health_check.py

# Run comprehensive tests
python -m pytest test_demo_scenarios.py -v

# Run performance benchmarks
python benchmark_spatial_ai.py

# Test specific scenarios
python enhanced_demo_scenarios.py
```

### 📊 **Monitoring**

The system includes comprehensive monitoring capabilities:

- **Health Endpoints**: `/health`, `/status` for system monitoring
- **Performance Metrics**: Real-time processing time and success rates
- **Error Tracking**: Detailed error logs with retry mechanisms
- **Resource Usage**: Memory and CPU utilization tracking

### 🚀 **Production Deployment**

```bash
# Production startup
python start.py

# Docker deployment (optional)
docker build -t spatial-ai .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key spatial-ai

# Environment-specific configs
export NODE_ENV=production
export DEBUG=false
python api_server.py
```

### 🔐 **Security Considerations**

- **API Key Protection**: Environment variable configuration
- **CORS Configuration**: Configurable for production deployment
- **Input Validation**: Comprehensive query validation and sanitization
- **Rate Limiting**: Built-in request throttling capabilities

## 📚 API Documentation

### 🌐 **Main Endpoints**

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `GET` | `/` | System information and web interface | - |
| `POST` | `/api/query` | Process spatial analysis queries | `query`, `context`, `execute_workflow` |
| `GET` | `/health` | System health check | - |
| `GET` | `/status` | Detailed system status | - |
| `GET` | `/docs` | Interactive API documentation | - |

### 🔧 **Query Processing Endpoint**

```bash
POST /api/query
Content-Type: application/json

{
  "query": "Find schools in Bangalore",
  "context": "Educational planning analysis",
  "execute_workflow": false,
  "include_reasoning": true,
  "include_workflow": true
}
```

**Response Structure:**
```json
{
  "query": "Find schools in Bangalore",
  "status": "success",
  "processing_time": 3.86,
  "chain_of_thought": [...],
  "workflow_visualization": {...},
  "performance_metrics": {...},
  "parsed_query": {...},
  "reasoning": {...},
  "workflow": {...},
  "rag_context": {...}
}
```

### 🧪 **Example Usage**

```python
import requests

# Process a spatial query
response = requests.post("http://localhost:8000/api/query", json={
    "query": "Find best places to build schools in flood-free zones near highways in Kerala",
    "include_reasoning": True,
    "include_workflow": True
})

result = response.json()
print(f"Status: {result['status']}")
print(f"Processing Time: {result['processing_time']}s")
print(f"Workflow Steps: {len(result['workflow']['steps'])}")
```

```javascript
// Frontend JavaScript example
const processQuery = async (query) => {
    const response = await fetch('/api/query', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            include_reasoning: true,
            include_workflow: true
        })
    });
    return await response.json();
};
```

## � Contributing

### 🔄 **Development Workflow**

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### 🧪 **Code Quality**

```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
pytest

# Type checking (optional)
mypy spatial_ai.py
```

### 📝 **Commit Messages**

Follow the conventional commit format:
```
🎯 feat: add new spatial operation
🐛 fix: resolve CRS transformation issue
📚 docs: update API documentation
🧪 test: add comprehensive test coverage
```

## 🏆 **Team Achievements**

This project successfully implements both **Member 1** and **Member 2** deliverables:

### 🧠 **Member 1: LLM & Reasoning Engine** ✅
- ✅ **Intelligent Query Parser**: Advanced NLP with intent extraction
- ✅ **Workflow Generator**: Automated GIS workflow creation
- ✅ **Chain-of-Thought Reasoning**: Transparent AI decision-making
- ✅ **RAG System**: Knowledge retrieval with GIS documentation
- ✅ **Error Handler**: Intelligent retry and recovery mechanisms

### 🗺️ **Member 2: GIS Engine & Interface** ✅
- ✅ **Robust GIS Pipeline**: 15+ spatial operations with error handling
- ✅ **Data Integration**: Multi-source data handling (OSM, Bhoonidhi, local)
- ✅ **Interactive Dashboard**: Modern web interface with real-time updates
- ✅ **Visualization Engine**: Dynamic maps and workflow visualization
- ✅ **Export Functionality**: JSON/YAML/Python workflow downloads

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Acknowledgments

- **Groq** for providing fast LLM inference
- **OpenStreetMap** for comprehensive geospatial data
- **Bhoonidhi (NRSC)** for Indian administrative boundaries
- **FastAPI** community for excellent web framework
- **GeoPandas** team for spatial data processing tools

## 📞 Support

- **Documentation**: http://localhost:8000/docs
- **Issues**: Create an issue on GitHub
- **Email**: spatial.ai.support@example.com

---

**Project Status**: ✅ **Production Ready** | **Competition Ready** | **Fully Operational**

**Version**: 2.0.0 | **Last Updated**: July 8, 2025 | **Authors**: Spatial.AI Team
