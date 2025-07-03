"""
RAG System - Retrieval-Augmented Generation for GIS Knowledge
============================================================

This module implements a RAG (Retrieval-Augmented Generation) system that
provides GIS domain knowledge to enhance LLM reasoning. It maintains a
vector database of GIS documentation, operation guides, and best practices.

Key Features:
- Vector database of GIS operations and documentation
- Semantic search for relevant GIS knowledge
- Context-aware prompt augmentation
- Dynamic knowledge retrieval for reasoning
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Document processing
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class GISKnowledgeItem:
    """Individual piece of GIS knowledge in the database"""
    
    item_id: str
    title: str
    content: str
    category: str  # operation, concept, dataset, best_practice
    subcategory: str  # buffer, intersect, qgis, gdal, etc.
    source: str  # qgis_docs, grass_docs, gdal_docs, custom
    tags: List[str] = None
    confidence: float = 1.0
    last_updated: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval"""
    
    query: str
    retrieved_items: List[GISKnowledgeItem]
    relevance_scores: List[float]
    total_items: int
    retrieval_time: float
    context_summary: str = ""


class RAGSystem:
    """
    Retrieval-Augmented Generation System for GIS Knowledge
    
    This class manages a vector database of GIS knowledge and provides
    semantic retrieval capabilities to enhance LLM reasoning with
    domain-specific information.
    """
    
    def __init__(self, 
                 db_path: str = "./data/vector_db",
                 embeddings_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "gis_knowledge"):
        """
        Initialize the RAG system
        
        Args:
            db_path: Path to store the vector database
            embeddings_model: Sentence transformer model for embeddings
            collection_name: Name of the ChromaDB collection
        """
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_model_name = embeddings_model
        self.collection_name = collection_name
        
        # Initialize embeddings model
        logger.info(f"Loading embeddings model: {embeddings_model}")
        self.embeddings_model = SentenceTransformer(embeddings_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            logger.info(f"Creating new collection: {collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Check if database is empty and populate if needed
        if self.collection.count() == 0:
            logger.info("Vector database is empty, populating with initial knowledge")
            self._populate_initial_knowledge()
        else:
            logger.info(f"Vector database loaded with {self.collection.count()} items")
    
    def add_knowledge(self, items: List[GISKnowledgeItem]) -> bool:
        """
        Add knowledge items to the vector database
        
        Args:
            items: List of GIS knowledge items to add
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Adding {len(items)} knowledge items to database")
            
            # Prepare data for ChromaDB
            documents = [item.content for item in items]
            metadatas = [
                {
                    "title": item.title,
                    "category": item.category,
                    "subcategory": item.subcategory,
                    "source": item.source,
                    "tags": ",".join(item.tags),
                    "confidence": item.confidence
                }
                for item in items
            ]
            ids = [item.item_id for item in items]
            
            # Generate embeddings
            embeddings = self.embeddings_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully added {len(items)} items to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge items: {str(e)}")
            return False
    
    def retrieve_knowledge(self, 
                          query: str, 
                          max_results: int = 10,
                          category_filter: Optional[str] = None,
                          min_relevance: float = 0.3) -> RetrievalResult:
        """
        Retrieve relevant knowledge for a query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            category_filter: Filter by knowledge category
            min_relevance: Minimum relevance score threshold
            
        Returns:
            RetrievalResult: Retrieved knowledge items with relevance scores
        """
        import time
        start_time = time.time()
        
        logger.info(f"Retrieving knowledge for query: {query[:100]}...")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query]).tolist()[0]
            
            # Build filter if specified
            where_filter = {}
            if category_filter:
                where_filter["category"] = category_filter
            
            # Perform semantic search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            retrieved_items = []
            relevance_scores = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Convert distance to relevance score (cosine similarity)
                relevance = 1.0 - distance
                
                if relevance >= min_relevance:
                    item = GISKnowledgeItem(
                        item_id=f"retrieved_{i}",
                        title=metadata["title"],
                        content=doc,
                        category=metadata["category"],
                        subcategory=metadata["subcategory"],
                        source=metadata["source"],
                        tags=metadata["tags"].split(",") if metadata["tags"] else [],
                        confidence=metadata["confidence"]
                    )
                    
                    retrieved_items.append(item)
                    relevance_scores.append(relevance)
            
            retrieval_time = time.time() - start_time
            
            # Generate context summary
            context_summary = self._generate_context_summary(retrieved_items, query)
            
            result = RetrievalResult(
                query=query,
                retrieved_items=retrieved_items,
                relevance_scores=relevance_scores,
                total_items=len(retrieved_items),
                retrieval_time=retrieval_time,
                context_summary=context_summary
            )
            
            logger.info(f"Retrieved {len(retrieved_items)} relevant items in {retrieval_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return RetrievalResult(
                query=query,
                retrieved_items=[],
                relevance_scores=[],
                total_items=0,
                retrieval_time=time.time() - start_time,
                context_summary="Error retrieving knowledge"
            )
    
    def augment_prompt(self, base_prompt: str, query: str, max_context_items: int = 5) -> str:
        """
        Augment a prompt with relevant GIS knowledge
        
        Args:
            base_prompt: Original prompt text
            query: Query for knowledge retrieval
            max_context_items: Maximum knowledge items to include
            
        Returns:
            str: Augmented prompt with relevant context
        """
        # Retrieve relevant knowledge
        knowledge = self.retrieve_knowledge(query, max_results=max_context_items)
        
        if not knowledge.retrieved_items:
            return base_prompt
        
        # Build context section
        context_section = "\n\n=== RELEVANT GIS KNOWLEDGE ===\n"
        
        for i, (item, score) in enumerate(zip(knowledge.retrieved_items, knowledge.relevance_scores)):
            context_section += f"\n[{i+1}] {item.title} ({item.category}/{item.subcategory})\n"
            context_section += f"Relevance: {score:.2f}\n"
            context_section += f"Content: {item.content[:300]}...\n"
            context_section += f"Source: {item.source}\n"
        
        context_section += "\n=== END KNOWLEDGE CONTEXT ===\n\n"
        
        # Insert context before the main prompt
        augmented_prompt = context_section + base_prompt
        
        logger.info(f"Augmented prompt with {len(knowledge.retrieved_items)} knowledge items")
        
        return augmented_prompt
    
    def _populate_initial_knowledge(self):
        """Populate the database with initial GIS knowledge"""
        
        logger.info("Populating initial GIS knowledge base")
        
        # Core GIS operations
        gis_operations = [
            {
                "title": "Buffer Analysis",
                "content": "Buffer analysis creates zones of specified distance around point, line, or polygon features. Common uses include proximity analysis, impact assessment, and service area definition. Parameters include buffer distance, buffer units (meters, feet, etc.), and end cap style (round, flat, square). Considerations: Choose appropriate distance based on analysis scale, ensure consistent coordinate system, handle overlapping buffers appropriately.",
                "category": "operation",
                "subcategory": "buffer",
                "source": "gis_fundamentals",
                "tags": ["proximity", "distance", "zones", "analysis"]
            },
            {
                "title": "Spatial Intersection",
                "content": "Intersection finds areas where two or more spatial datasets overlap. Creates new features containing only the overlapping portions with attributes from all input layers. Used for overlay analysis, finding common areas, and spatial filtering. Important considerations: Ensure matching coordinate systems, handle topology errors, consider attribute joining methods, performance implications for large datasets.",
                "category": "operation", 
                "subcategory": "intersect",
                "source": "gis_fundamentals",
                "tags": ["overlay", "overlap", "spatial_query", "analysis"]
            },
            {
                "title": "Suitability Analysis",
                "content": "Multi-criteria suitability analysis combines multiple spatial criteria to identify optimal locations for specific purposes. Involves: 1) Criterion identification, 2) Data standardization/normalization, 3) Weight assignment, 4) Weighted overlay calculation, 5) Result classification. Methods include Boolean overlay, weighted linear combination, and fuzzy logic. Considerations: Criterion independence, weight sensitivity, scale effects.",
                "category": "operation",
                "subcategory": "suitability",
                "source": "spatial_analysis",
                "tags": ["mcda", "criteria", "weights", "optimization", "site_selection"]
            },
            {
                "title": "Proximity Analysis",
                "content": "Proximity analysis measures spatial relationships based on distance. Types include: Euclidean distance (straight-line), Manhattan distance (grid-based), network distance (along paths), and cost-weighted distance. Applications: accessibility analysis, service area definition, travel time calculation. Tools: Near table, distance rasters, service area analysis. Considerations: Distance metric choice, barriers and impedances, computational efficiency.",
                "category": "operation",
                "subcategory": "proximity",
                "source": "spatial_analysis", 
                "tags": ["distance", "accessibility", "travel_time", "nearest"]
            }
        ]
        
        # GIS concepts
        gis_concepts = [
            {
                "title": "Coordinate Reference Systems (CRS)",
                "content": "CRS defines how spatial data relates to real-world locations. Components: Geographic coordinate system (datum, ellipsoid, prime meridian) and projected coordinate system (projection method, parameters). Common systems: WGS84 (EPSG:4326) for global data, UTM zones for regional analysis, local state plane systems. Best practices: Use appropriate CRS for analysis scale, ensure all layers use same CRS, understand distortion effects, reproject when necessary.",
                "category": "concept",
                "subcategory": "crs",
                "source": "gis_fundamentals",
                "tags": ["projection", "datum", "epsg", "coordinate_system"]
            },
            {
                "title": "Raster vs Vector Data",
                "content": "Vector data represents features as points, lines, and polygons with exact boundaries. Good for discrete features, precise analysis, smaller file sizes for simple features. Raster data uses regular grid cells with values. Good for continuous phenomena, spatial modeling, image analysis. Choose based on: data nature (discrete vs continuous), analysis type, precision requirements, data sources. Conversion between formats may cause generalization.",
                "category": "concept",
                "subcategory": "data_models",
                "source": "gis_fundamentals",
                "tags": ["vector", "raster", "data_model", "representation"]
            }
        ]
        
        # Dataset information
        dataset_info = [
            {
                "title": "Digital Elevation Models (DEM)",
                "content": "DEM represents terrain elevation as raster data. Sources: SRTM (30m global), ASTER GDEM (30m global), national DEMs (higher resolution). Applications: slope analysis, watershed delineation, viewshed analysis, flood modeling, 3D visualization. Quality considerations: horizontal/vertical accuracy, data gaps, artifacts from source (radar vs photogrammetry). Processing: void filling, smoothing, resampling.",
                "category": "dataset",
                "subcategory": "elevation",
                "source": "data_sources",
                "tags": ["elevation", "terrain", "srtm", "topography", "slope"]
            },
            {
                "title": "OpenStreetMap (OSM) Data", 
                "content": "OSM provides free, editable map data worldwide. Features: roads, buildings, points of interest, land use, administrative boundaries. Access via: Overpass API (query-based), data extracts, specialized tools. Quality varies by region and feature type. Good for: transportation networks, building footprints, POI analysis. Considerations: data completeness, update frequency, licensing (ODbL), attribute standardization.",
                "category": "dataset",
                "subcategory": "osm",
                "source": "data_sources", 
                "tags": ["openstreetmap", "roads", "buildings", "poi", "crowdsourced"]
            }
        ]
        
        # Best practices
        best_practices = [
            {
                "title": "Data Quality and Validation",
                "content": "Essential data quality checks: Completeness (missing data, gaps), positional accuracy (coordinate precision, alignment), attribute accuracy (correct values, classifications), temporal accuracy (currency, time stamps), logical consistency (topology, relationships). Validation methods: visual inspection, statistical analysis, comparison with reference data, automated checks. Document data quality for decision-making.",
                "category": "best_practice",
                "subcategory": "quality",
                "source": "gis_standards",
                "tags": ["quality", "validation", "accuracy", "completeness", "standards"]
            },
            {
                "title": "Spatial Analysis Workflow Design",
                "content": "Systematic workflow design improves results and reproducibility: 1) Problem definition and requirements, 2) Data inventory and assessment, 3) Analysis design and method selection, 4) Implementation with quality checks, 5) Results validation and interpretation, 6) Documentation and communication. Consider: scale effects, error propagation, sensitivity analysis, alternative methods, computational efficiency.",
                "category": "best_practice",
                "subcategory": "workflow",
                "source": "spatial_analysis",
                "tags": ["workflow", "methodology", "reproducibility", "design", "documentation"]
            }
        ]
        
        # Combine all knowledge
        all_knowledge = []
        
        for item_data in gis_operations + gis_concepts + dataset_info + best_practices:
            item = GISKnowledgeItem(
                item_id=self._generate_item_id(item_data["title"]),
                title=item_data["title"],
                content=item_data["content"],
                category=item_data["category"],
                subcategory=item_data["subcategory"],
                source=item_data["source"],
                tags=item_data["tags"]
            )
            all_knowledge.append(item)
        
        # Add to database
        success = self.add_knowledge(all_knowledge)
        
        if success:
            logger.info(f"Successfully populated database with {len(all_knowledge)} initial knowledge items")
        else:
            logger.error("Failed to populate initial knowledge base")
    
    def _generate_item_id(self, title: str) -> str:
        """Generate unique ID for knowledge item"""
        return hashlib.md5(title.encode()).hexdigest()[:12]
    
    def _generate_context_summary(self, items: List[GISKnowledgeItem], query: str) -> str:
        """Generate a summary of retrieved context"""
        
        if not items:
            return "No relevant knowledge retrieved"
        
        categories = {}
        for item in items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item.title)
        
        summary = f"Retrieved {len(items)} relevant knowledge items for query '{query[:50]}...':\n"
        
        for category, titles in categories.items():
            summary += f"- {category.title()}: {len(titles)} items\n"
        
        return summary
    
    def export_knowledge(self, output_path: str) -> bool:
        """Export knowledge base to JSON file"""
        
        try:
            # Get all items from collection
            results = self.collection.get(include=["documents", "metadatas"])
            
            knowledge_items = []
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                item = {
                    "title": metadata["title"],
                    "content": doc,
                    "category": metadata["category"],
                    "subcategory": metadata["subcategory"],
                    "source": metadata["source"],
                    "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                    "confidence": metadata["confidence"]
                }
                knowledge_items.append(item)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_items, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(knowledge_items)} knowledge items to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge: {str(e)}")
            return False
    
    def import_knowledge(self, input_path: str) -> bool:
        """Import knowledge base from JSON file"""
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            knowledge_items = []
            for item_data in data:
                item = GISKnowledgeItem(
                    item_id=self._generate_item_id(item_data["title"]),
                    title=item_data["title"],
                    content=item_data["content"],
                    category=item_data["category"],
                    subcategory=item_data["subcategory"],
                    source=item_data["source"],
                    tags=item_data.get("tags", []),
                    confidence=item_data.get("confidence", 1.0)
                )
                knowledge_items.append(item)
            
            success = self.add_knowledge(knowledge_items)
            
            if success:
                logger.info(f"Imported {len(knowledge_items)} knowledge items from {input_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error importing knowledge: {str(e)}")
            return False
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        
        try:
            # Get all metadata
            results = self.collection.get(include=["metadatas"])
            
            total_items = len(results["metadatas"])
            
            # Count by category
            categories = {}
            sources = {}
            
            for metadata in results["metadatas"]:
                category = metadata["category"]
                source = metadata["source"]
                
                categories[category] = categories.get(category, 0) + 1
                sources[source] = sources.get(source, 0) + 1
            
            stats = {
                "total_items": total_items,
                "categories": categories,
                "sources": sources,
                "embeddings_model": self.embeddings_model_name,
                "collection_name": self.collection_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {str(e)}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Test RAG system
    rag = RAGSystem()
    
    # Test knowledge retrieval
    test_queries = [
        "buffer analysis for proximity",
        "suitability analysis methods",
        "coordinate reference systems",
        "DEM data sources"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = rag.retrieve_knowledge(query, max_results=3)
        print(f"Retrieved {result.total_items} items in {result.retrieval_time:.3f}s")
        
        for item, score in zip(result.retrieved_items, result.relevance_scores):
            print(f"- {item.title} (relevance: {score:.3f})")
            print(f"  Category: {item.category}/{item.subcategory}")
            print(f"  Content: {item.content[:100]}...")
    
    # Print knowledge base stats
    print("\nKnowledge Base Statistics:")
    print("=" * 40)
    stats = rag.get_knowledge_stats()
    print(json.dumps(stats, indent=2))
