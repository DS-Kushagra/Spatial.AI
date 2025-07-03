# Spatial.AI - Chain-of-Thought GIS Workflow Orchestration

An intelligent geospatial analysis system that uses Large Language Models to automatically generate and execute complex GIS workflows from natural language queries.

## 🎯 System Architecture

```
User Query → LLM Brain (Member A) → GIS Engine (Member B) → Results → UI (Both)
```

## 📋 Project Overview

This system combines the reasoning capabilities of LLMs with powerful geoprocessing tools to automatically plan and execute geospatial workflows step-by-step, similar to how a human GIS expert would approach complex spatial analysis tasks.

### Key Features
- **Natural Language to GIS Workflow**: Convert user queries into executable spatial analysis workflows
- **Chain-of-Thought Reasoning**: Transparent, step-by-step reasoning for workflow generation
- **Intelligent Tool Selection**: Automatic selection of appropriate GIS operations and parameters
- **Error Recovery**: Robust error handling and iterative improvement
- **Multi-format Support**: Handle various spatial data formats and coordinate systems

## 🛠️ Technology Stack

### LLM & Intelligence (Member A)
- **LLM**: Mistral-7B-Instruct / LLaMA-3-8B
- **Framework**: LangChain / Transformers
- **RAG**: Vector embeddings, document retrieval
- **Orchestration**: Custom workflow engine

### GIS & Execution (Member B)
- **Core Tools**: PyQGIS, GeoPandas, Rasterio, GDAL
- **Processing**: WhiteboxTools, headless QGIS
- **Data Sources**: Bhoonidhi, OpenStreetMap, public datasets
- **Formats**: Vector, raster, various spatial formats

### Shared Infrastructure
- **UI**: Streamlit-based web interface
- **API**: RESTful communication between components
- **Storage**: Workflow patterns, knowledge base
- **Validation**: Quality checks and error handling

## 📁 Project Structure

```
Spatial.AI/
├── core/                    # Core system components
│   ├── llm/                # LLM integration (Member A)
│   ├── gis/                # GIS operations (Member B)
│   ├── orchestrator.py     # Main workflow orchestrator
│   └── api.py              # API interfaces
├── rag/                    # Retrieval-Augmented Generation
│   ├── knowledge_base/     # GIS documentation and examples
│   ├── embeddings/         # Vector embeddings
│   └── retrieval.py        # RAG system
├── prompts/                # Chain-of-Thought prompts (Member A)
│   ├── templates/          # Prompt templates
│   ├── examples/           # Few-shot examples
│   └── reasoning.py        # Reasoning patterns
├── workflows/              # Workflow definitions and patterns
│   ├── schemas/            # JSON/YAML schemas
│   ├── patterns/           # Common workflow patterns
│   └── executor.py         # Workflow execution engine
├── ui/                     # Streamlit interface
│   ├── app.py              # Main application
│   ├── components/         # UI components
│   └── visualization/      # Map and result display
├── data/                   # Sample datasets and outputs
├── tests/                  # Test cases and benchmarks
├── docs/                   # Documentation
└── examples/               # Demo use cases
```

## 🎯 Target Use Cases

1. **Flood Risk Assessment**: Analyze elevation, drainage, and precipitation data
2. **Site Suitability Analysis**: Multi-criteria spatial analysis for optimal locations
3. **Land Cover Monitoring**: Change detection and classification workflows

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- QGIS 3.x (for PyQGIS)
- GPU support (recommended for LLM inference)

### Installation
```bash
git clone <repository>
cd Spatial.AI
pip install -r requirements.txt
```

### Quick Start
```bash
streamlit run ui/app.py
```

## 📊 Evaluation Metrics

- **Workflow Validity**: Logical and syntactic correctness
- **Reasoning Clarity**: Chain-of-Thought transparency
- **Error Robustness**: Handling of spatial data issues
- **Performance**: Runtime efficiency and resource usage
- **Accuracy**: Comparison with manual expert workflows

## 🤝 Team Division

- **Member A (LLM & Intelligence)**: LLM integration, reasoning engine, RAG system, prompt engineering
- **Member B (GIS & Execution)**: Spatial operations, workflow execution, data management, validation
- **Shared**: System integration, UI development, testing, demo preparation

## 📝 License

MIT License - See LICENSE file for details
