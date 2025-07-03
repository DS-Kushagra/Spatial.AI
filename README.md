# Spatial.AI - Your AI GIS Expert 🗺️🤖

**One-Line Pitch**: "Describe what you want in plain English, and our AI will automatically plan and execute the entire GIS workflow for you."

## 🎯 Example Magic

- **Input**: "Find best places to build schools in flood-free zones near highways in Kerala"
- **Output**: Complete GIS workflow with maps, analysis, and reasoning steps

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  LLM Reasoning  │───▶│  GIS Execution  │
│  (Natural Lang) │    │    Engine       │    │    Pipeline     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Chain-of-Thought│    │   Streamlit     │
                       │   Reasoning     │    │      UI         │
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### LLM & Reasoning (Member 1)

- **LLM**: Groq API (Mixtral-8x7B or Llama3-70B)
- **Frameworks**: LangChain, Custom Prompt Engineering
- **RAG**: Chroma Vector DB + GIS Documentation
- **Reasoning**: Chain-of-Thought prompt templates

### GIS Engine (Member 2)

- **Processing**: PyQGIS, GeoPandas, Rasterio, WhiteboxTools
- **Data Sources**: Bhoonidhi, OpenStreetMap
- **UI**: Streamlit + Folium/Plotly
- **Backend**: FastAPI

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your Groq API key to .env

# Run the application
streamlit run app.py
```

## 📁 Project Structure

```
spatial-ai/
├── llm_engine/          # LLM & Reasoning Engine (Member 1)
├── gis_engine/          # GIS Processing Pipeline (Member 2)
├── ui/                  # Streamlit Interface
├── data/                # Sample datasets
├── workflows/           # Generated workflow templates
└── tests/               # Unit and integration tests
```

## 🎯 Demo Scenarios

1. **School Site Selection**: Find optimal school locations avoiding flood zones
2. **Solar Farm Planning**: Identify best solar farm areas considering multiple factors
3. **Flood Risk Mapping**: Map flood risk zones using elevation and rainfall data

## 🏆 Key Features

- **Transparent Reasoning**: See every step of AI thinking
- **Error Recovery**: Intelligent failure handling and alternatives
- **Workflow Templates**: Reusable patterns for similar problems
- **Real-time Visualization**: Interactive maps and charts
