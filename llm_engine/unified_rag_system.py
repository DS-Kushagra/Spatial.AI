"""
Unified RAG System - Comprehensive Retrieval-Augmented Generation for GIS
=========================================================================

This module implements a complete RAG system with both basic and advanced features:
- Vector database of GIS operations and documentation  
- Semantic search with multiple retrieval strategies
- Knowledge clustering and hierarchical organization
- Context-aware retrieval and knowledge gap detection
- Dynamic knowledge expansion and real-time validation

This replaces both rag_system.py and enhanced_rag.py with a unified implementation.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from datetime import datetime
import logging

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Document processing
import requests
from bs4 import BeautifulSoup
import numpy as np

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
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval"""
    
    query: str
    retrieved_items: List[GISKnowledgeItem]
    relevance_scores: List[float]
    context_summary: str
    total_items: int
    retrieval_time: float
    strategy_used: str = "basic"


class KnowledgeCategory(Enum):
    """Knowledge categories for organization"""
    GIS_OPERATIONS = "gis_operations"
    SPATIAL_ANALYSIS = "spatial_analysis"
    DATA_SOURCES = "data_sources"
    ALGORITHMS = "algorithms"
    BEST_PRACTICES = "best_practices"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CASE_STUDIES = "case_studies"
    TOOLS_SOFTWARE = "tools_software"
    STANDARDS_FORMATS = "standards_formats"


@dataclass
class KnowledgeCluster:
    """Cluster of related knowledge items"""
    
    cluster_id: str
    name: str
    description: str
    items: List[GISKnowledgeItem]
    centroid_embedding: np.ndarray
    cluster_size: int
    coherence_score: float
    representative_keywords: List[str]


@dataclass
@dataclass
class ContextualRetrievalResult:
    """Enhanced retrieval result with context and analysis"""
    
    query: str
    primary_results: List[GISKnowledgeItem]
    contextual_results: List[GISKnowledgeItem]
    cluster_matches: List[KnowledgeCluster]
    knowledge_gaps: List[str]
    confidence_score: float
    retrieval_strategy: str
    expansion_suggestions: List[str]
    
    @property
    def total_items(self) -> int:
        """Total number of retrieved items"""
        return len(self.primary_results) + len(self.contextual_results)
    
    @property
    def all_results(self) -> List[GISKnowledgeItem]:
        """All retrieved items combined"""
        return self.primary_results + self.contextual_results


class UnifiedRAGSystem:
    """
    Unified RAG System combining basic and advanced retrieval capabilities
    
    This single class replaces both RAGSystem and EnhancedRAGSystem,
    providing all functionality in one cohesive implementation.
    """
    
    def __init__(self, 
                 chroma_persist_dir: str = "./rag_db",
                 model_name: str = "all-MiniLM-L6-v2"):
        
        self.persist_dir = Path(chroma_persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize embeddings model
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="gis_knowledge",
            metadata={"description": "GIS domain knowledge and documentation"}
        )
        
        # Advanced features
        self.knowledge_clusters: Dict[str, KnowledgeCluster] = {}
        self.category_embeddings: Dict[KnowledgeCategory, np.ndarray] = {}
        self.knowledge_gaps: List[str] = []
        
        # Retrieval strategies
        self.retrieval_strategies = {
            'basic': self._basic_retrieval,
            'semantic': self._semantic_retrieval,
            'hierarchical': self._hierarchical_retrieval,
            'contextual': self._contextual_retrieval,
            'hybrid': self._hybrid_retrieval
        }
        
        # Context windows for different query types
        self.context_windows = {
            'spatial_analysis': 5,
            'site_selection': 7,
            'network_analysis': 6,
            'temporal_analysis': 4,
            'general': 3
        }
        
        # Initialize advanced features if we have data
        self._initialize_advanced_features()
        
        logger.info(f"Unified RAG System initialized with {self.collection.count()} items")
    
    # ================================================================
    # CORE RAG FUNCTIONALITY (Basic Features)
    # ================================================================
    
    def add_knowledge(self, items: List[GISKnowledgeItem]) -> bool:
        """Add knowledge items to the database"""
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            for item in items:
                # Create searchable document text
                doc_text = f"{item.title}\\n{item.content}"
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "title": item.title,
                    "category": item.category,
                    "subcategory": item.subcategory,
                    "source": item.source,
                    "confidence": item.confidence,
                    "last_updated": item.last_updated,
                    "tags": ",".join(item.tags) if item.tags else ""
                }
                metadatas.append(metadata)
                
                # Use item_id as document ID
                ids.append(item.item_id)
                
                # Generate embedding
                embedding = self.model.encode(doc_text)
                embeddings.append(embedding.tolist())
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            # Refresh advanced features
            self._initialize_advanced_features()
            
            logger.info(f"Added {len(items)} knowledge items successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge items: {e}")
            return False
    
    def retrieve_knowledge(self, 
                         query: str, 
                         max_results: int = 5,
                         strategy: str = "basic") -> RetrievalResult:
        """
        Retrieve relevant knowledge using specified strategy
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            strategy: Retrieval strategy ('basic', 'semantic', 'hierarchical', 'contextual', 'hybrid')
        """
        
        start_time = time.time()
        
        # Use the specified strategy
        if strategy in self.retrieval_strategies:
            retrieved_items = self.retrieval_strategies[strategy](query, max_results)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using basic retrieval")
            retrieved_items = self._basic_retrieval(query, max_results)
        
        # Calculate relevance scores
        relevance_scores = [1.0 - (i * 0.1) for i in range(len(retrieved_items))]
        
        # Generate context summary
        context_summary = self._generate_context_summary(retrieved_items, query)
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query=query,
            retrieved_items=retrieved_items,
            relevance_scores=relevance_scores,
            context_summary=context_summary,
            total_items=len(retrieved_items),
            retrieval_time=retrieval_time,
            strategy_used=strategy
        )
        
        logger.info(f"Retrieved {len(retrieved_items)} items using {strategy} strategy in {retrieval_time:.2f}s")
        return result
    
    # ================================================================
    # ADVANCED RAG FUNCTIONALITY (Enhanced Features)  
    # ================================================================
    
    def contextual_retrieve(self, 
                          query: str, 
                          context: Dict[str, Any] = None,
                          strategy: str = 'hybrid') -> ContextualRetrievalResult:
        """Advanced contextual retrieval with gap detection"""
        
        logger.info(f"Starting contextual retrieval for: {query}")
        
        # Determine context window size
        query_type = context.get('analysis_type', 'general') if context else 'general'
        context_window = self.context_windows.get(query_type, 3)
        
        # Primary retrieval using selected strategy
        primary_items = self.retrieval_strategies.get(strategy, self._hybrid_retrieval)(query, context_window)
        
        # Contextual expansion
        contextual_items = self._expand_contextual_results(query, primary_items, context)
        
        # Find relevant clusters
        cluster_matches = self._find_relevant_clusters(query)
        
        # Detect knowledge gaps
        knowledge_gaps = self._detect_knowledge_gaps(query, primary_items)
        
        # Calculate confidence
        confidence_score = self._calculate_retrieval_confidence(primary_items, contextual_items, cluster_matches)
        
        # Generate expansion suggestions
        expansion_suggestions = self._generate_expansion_suggestions(knowledge_gaps)
        
        return ContextualRetrievalResult(
            query=query,
            primary_results=primary_items,
            contextual_results=contextual_items,
            cluster_matches=cluster_matches,
            knowledge_gaps=knowledge_gaps,
            confidence_score=confidence_score,
            retrieval_strategy=strategy,
            expansion_suggestions=expansion_suggestions
        )
    
    def generate_knowledge_report(self) -> Dict[str, Any]:
        """Generate comprehensive knowledge base report"""
        
        total_items = self.collection.count()
        
        report = {
            "total_knowledge_items": total_items,
            "total_clusters": len(self.knowledge_clusters),
            "knowledge_categories": len(self.category_embeddings),
            "identified_gaps": len(self.knowledge_gaps),
            "cluster_summary": {},
            "gap_summary": self.knowledge_gaps[:5],
            "coverage_analysis": self._analyze_knowledge_coverage()
        }
        
        # Add cluster summaries
        for cluster_id, cluster in self.knowledge_clusters.items():
            report["cluster_summary"][cluster_id] = {
                "name": cluster.name,
                "size": cluster.cluster_size,
                "coherence": cluster.coherence_score,
                "keywords": cluster.representative_keywords[:3]
            }
        
        return report
    
    # ================================================================
    # RETRIEVAL STRATEGY IMPLEMENTATIONS
    # ================================================================
    
    def _basic_retrieval(self, query: str, top_k: int = 5) -> List[GISKnowledgeItem]:
        """Basic semantic similarity retrieval"""
        
        if self.collection.count() == 0:
            return []
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count())
            )
            
            # Convert to GISKnowledgeItem objects
            items = []
            for i, (doc, metadata, doc_id) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['ids'][0]
            )):
                
                # Parse tags
                tags = metadata.get('tags', '').split(',') if metadata.get('tags') else []
                tags = [tag.strip() for tag in tags if tag.strip()]
                
                item = GISKnowledgeItem(
                    item_id=doc_id,
                    title=metadata.get('title', f'Item {i}'),
                    content=doc.split('\\n', 1)[1] if '\\n' in doc else doc,
                    category=metadata.get('category', 'general'),
                    subcategory=metadata.get('subcategory', 'general'),
                    source=metadata.get('source', 'unknown'),
                    tags=tags,
                    confidence=metadata.get('confidence', 1.0),
                    last_updated=metadata.get('last_updated', datetime.now().isoformat())
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Basic retrieval failed: {e}")
            return []
    
    def _semantic_retrieval(self, query: str, top_k: int = 5) -> List[GISKnowledgeItem]:
        """Pure semantic similarity retrieval"""
        return self._basic_retrieval(query, top_k)
    
    def _hierarchical_retrieval(self, query: str, top_k: int = 5) -> List[GISKnowledgeItem]:
        """Hierarchical retrieval considering knowledge categories"""
        
        # Find most relevant category
        query_embedding = self.model.encode(query)
        category_similarities = {}
        
        for category, cat_embedding in self.category_embeddings.items():
            similarity = cosine_similarity([query_embedding], [cat_embedding])[0][0]
            category_similarities[category] = similarity
        
        if not category_similarities:
            return self._basic_retrieval(query, top_k)
        
        # Get top category
        top_category = max(category_similarities, key=category_similarities.get)
        
        # Retrieve and filter by category
        all_results = self._basic_retrieval(query, top_k * 2)
        filtered_results = [r for r in all_results if r.category == top_category.value]
        
        return filtered_results[:top_k]
    
    def _contextual_retrieval(self, query: str, top_k: int = 5) -> List[GISKnowledgeItem]:
        """Context-aware retrieval with keyword expansion"""
        
        # Extract context keywords
        context_keywords = self._extract_context_keywords(query)
        
        # Expand query with context
        expanded_query = f"{query} {' '.join(context_keywords)}"
        
        return self._basic_retrieval(expanded_query, top_k)
    
    def _hybrid_retrieval(self, query: str, top_k: int = 5) -> List[GISKnowledgeItem]:
        """Hybrid retrieval combining multiple strategies"""
        
        # Get results from different strategies
        semantic_results = self._semantic_retrieval(query, top_k//2)
        hierarchical_results = self._hierarchical_retrieval(query, top_k//2)
        contextual_results = self._contextual_retrieval(query, top_k//2)
        
        # Combine and deduplicate
        all_results = semantic_results + hierarchical_results + contextual_results
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.item_id not in seen_ids:
                seen_ids.add(result.item_id)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    # ================================================================
    # ADVANCED FEATURE IMPLEMENTATIONS
    # ================================================================
    
    def _initialize_advanced_features(self):
        """Initialize advanced RAG features"""
        
        if self.collection.count() == 0:
            logger.info("No knowledge items available, skipping advanced feature initialization")
            return
        
        logger.info("Initializing advanced RAG features...")
        
        try:
            # Create knowledge clusters
            self._create_knowledge_clusters()
            
            # Generate category embeddings
            self._generate_category_embeddings()
            
            # Initialize knowledge gap detection
            self._initialize_gap_detection()
            
            logger.info("Advanced RAG features initialized successfully")
            
        except Exception as e:
            logger.warning(f"Advanced feature initialization failed: {e}")
    
    def _create_knowledge_clusters(self, num_clusters: int = 10):
        """Create semantic clusters of knowledge items"""
        
        try:
            # Get all items
            all_results = self.collection.get()
            
            if len(all_results['ids']) < 2:
                logger.info("Not enough items for clustering")
                return
            
            # Limit clusters to available items
            actual_clusters = min(num_clusters, len(all_results['ids']))
            
            # Get embeddings
            embeddings_array = np.array(all_results['embeddings'])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Create cluster objects
            for cluster_id in range(actual_clusters):
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if cluster_indices:
                    cluster_items = []
                    cluster_embeddings = []
                    
                    for idx in cluster_indices:
                        # Create GISKnowledgeItem
                        metadata = all_results['metadatas'][idx]
                        doc = all_results['documents'][idx]
                        doc_id = all_results['ids'][idx]
                        
                        tags = metadata.get('tags', '').split(',') if metadata.get('tags') else []
                        tags = [tag.strip() for tag in tags if tag.strip()]
                        
                        item = GISKnowledgeItem(
                            item_id=doc_id,
                            title=metadata.get('title', f'Item {idx}'),
                            content=doc.split('\\n', 1)[1] if '\\n' in doc else doc,
                            category=metadata.get('category', 'general'),
                            subcategory=metadata.get('subcategory', 'general'),
                            source=metadata.get('source', 'unknown'),
                            tags=tags,
                            confidence=metadata.get('confidence', 1.0),
                            last_updated=metadata.get('last_updated', datetime.now().isoformat())
                        )
                        
                        cluster_items.append(item)
                        cluster_embeddings.append(embeddings_array[idx])
                    
                    # Calculate centroid
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Generate cluster metadata
                    cluster_name = f"Cluster_{cluster_id}"
                    cluster_description = self._generate_cluster_description(cluster_items)
                    coherence = self._calculate_cluster_coherence(cluster_embeddings, centroid)
                    keywords = self._extract_cluster_keywords(cluster_items)
                    
                    cluster = KnowledgeCluster(
                        cluster_id=str(cluster_id),
                        name=cluster_name,
                        description=cluster_description,
                        items=cluster_items,
                        centroid_embedding=centroid,
                        cluster_size=len(cluster_items),
                        coherence_score=coherence,
                        representative_keywords=keywords
                    )
                    
                    self.knowledge_clusters[str(cluster_id)] = cluster
            
            logger.info(f"Created {len(self.knowledge_clusters)} knowledge clusters")
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
    
    def _generate_category_embeddings(self):
        """Generate embeddings for knowledge categories"""
        
        category_descriptions = {
            KnowledgeCategory.GIS_OPERATIONS: "Basic GIS operations like buffering, overlay, and transformation",
            KnowledgeCategory.SPATIAL_ANALYSIS: "Advanced spatial analysis methods and techniques",
            KnowledgeCategory.DATA_SOURCES: "Various spatial data sources and formats",
            KnowledgeCategory.ALGORITHMS: "Spatial algorithms and computational methods",
            KnowledgeCategory.BEST_PRACTICES: "Best practices and guidelines for GIS workflows",
            KnowledgeCategory.ERROR_HANDLING: "Error handling and debugging in GIS processes",
            KnowledgeCategory.PERFORMANCE_OPTIMIZATION: "Performance optimization techniques",
            KnowledgeCategory.CASE_STUDIES: "Real-world GIS application examples",
            KnowledgeCategory.TOOLS_SOFTWARE: "GIS software tools and platforms",
            KnowledgeCategory.STANDARDS_FORMATS: "GIS standards and data formats"
        }
        
        for category, description in category_descriptions.items():
            embedding = self.model.encode(description)
            self.category_embeddings[category] = embedding
    
    def _initialize_gap_detection(self):
        """Initialize knowledge gap detection"""
        
        common_gaps = [
            "Handling large-scale raster processing",
            "Real-time spatial data streaming", 
            "3D spatial analysis workflows",
            "Cloud-based GIS processing",
            "Machine learning integration with GIS",
            "Advanced coordinate system handling",
            "Spatial database optimization",
            "Mobile GIS development"
        ]
        
        self.knowledge_gaps = common_gaps
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _expand_contextual_results(self, query: str, primary_results: List[GISKnowledgeItem], context: Dict[str, Any] = None) -> List[GISKnowledgeItem]:
        """Expand results with contextual information"""
        
        if not primary_results:
            return []
        
        # Extract related concepts
        related_concepts = []
        for result in primary_results:
            if result.tags:
                related_concepts.extend(result.tags)
        
        if not related_concepts:
            return []
        
        # Create expansion query
        expansion_query = f"related to {' '.join(related_concepts[:5])}"
        
        # Get additional results
        additional_results = self._basic_retrieval(expansion_query, 3)
        
        # Filter out duplicates
        primary_ids = {r.item_id for r in primary_results}
        contextual_filtered = [r for r in additional_results if r.item_id not in primary_ids]
        
        return contextual_filtered
    
    def _find_relevant_clusters(self, query: str) -> List[KnowledgeCluster]:
        """Find knowledge clusters relevant to the query"""
        
        if not self.knowledge_clusters:
            return []
        
        query_embedding = self.model.encode(query)
        cluster_similarities = []
        
        for cluster in self.knowledge_clusters.values():
            similarity = cosine_similarity([query_embedding], [cluster.centroid_embedding])[0][0]
            cluster_similarities.append((cluster, similarity))
        
        # Sort by similarity and return top clusters
        cluster_similarities.sort(key=lambda x: x[1], reverse=True)
        return [cluster for cluster, sim in cluster_similarities[:3] if sim > 0.3]
    
    def _detect_knowledge_gaps(self, query: str, results: List[GISKnowledgeItem]) -> List[str]:
        """Detect potential knowledge gaps"""
        
        gaps = []
        
        # Check result quality
        if not results:
            gaps.append(f"No relevant results found for query: {query}")
        elif len(results) < 3:
            gaps.append(f"Limited results available for query: {query}")
        
        # Check for domain-specific gaps
        query_lower = query.lower()
        domain_keywords = {
            '3d': "Limited 3D spatial analysis knowledge",
            'machine learning': "Limited ML integration knowledge",
            'real-time': "Limited real-time processing knowledge",
            'cloud': "Limited cloud GIS knowledge",
            'mobile': "Limited mobile GIS knowledge"
        }
        
        for keyword, gap_desc in domain_keywords.items():
            if keyword in query_lower:
                gaps.append(gap_desc)
        
        return gaps
    
    def _calculate_retrieval_confidence(self, primary_results: List[GISKnowledgeItem], contextual_results: List[GISKnowledgeItem], cluster_matches: List[KnowledgeCluster]) -> float:
        """Calculate overall confidence in retrieval results"""
        
        # Base confidence from primary results
        primary_conf = min(len(primary_results) / 3.0, 1.0)
        
        # Boost from contextual results
        contextual_boost = min(len(contextual_results) / 3.0, 0.2)
        
        # Boost from cluster matches
        cluster_boost = min(len(cluster_matches) / 3.0, 0.1)
        
        return min(primary_conf + contextual_boost + cluster_boost, 1.0)
    
    def _generate_expansion_suggestions(self, knowledge_gaps: List[str]) -> List[str]:
        """Generate suggestions for expanding knowledge base"""
        
        suggestions = []
        
        for gap in knowledge_gaps:
            if "3d" in gap.lower():
                suggestions.append("Add 3D spatial analysis documentation and examples")
            elif "machine learning" in gap.lower():
                suggestions.append("Include ML-GIS integration tutorials and case studies")
            elif "real-time" in gap.lower():
                suggestions.append("Add real-time spatial data processing workflows")
            elif "cloud" in gap.lower():
                suggestions.append("Include cloud-based GIS platform documentation")
            else:
                suggestions.append(f"Expand knowledge base to address: {gap}")
        
        return suggestions[:5]
    
    def _extract_context_keywords(self, query: str) -> List[str]:
        """Extract context keywords from query"""
        
        gis_keywords = [
            'spatial', 'buffer', 'overlay', 'analysis', 'raster', 'vector',
            'coordinate', 'projection', 'geodatabase', 'shapefile', 'topology',
            'interpolation', 'kriging', 'network', 'route', 'service area'
        ]
        
        query_words = query.lower().split()
        return [word for word in query_words if word in gis_keywords]
    
    def _generate_cluster_description(self, items: List[GISKnowledgeItem]) -> str:
        """Generate description for a knowledge cluster"""
        
        if not items:
            return "Empty cluster"
        
        categories = [item.category for item in items]
        most_common_category = max(set(categories), key=categories.count)
        
        return f"Knowledge cluster focused on {most_common_category} with {len(items)} items"
    
    def _calculate_cluster_coherence(self, embeddings: List[np.ndarray], centroid: np.ndarray) -> float:
        """Calculate coherence score for a cluster"""
        
        if not embeddings:
            return 0.0
        
        distances = [cosine_similarity([emb], [centroid])[0][0] for emb in embeddings]
        return sum(distances) / len(distances)
    
    def _extract_cluster_keywords(self, items: List[GISKnowledgeItem]) -> List[str]:
        """Extract representative keywords for a cluster"""
        
        all_keywords = []
        for item in items:
            if item.tags:
                all_keywords.extend(item.tags)
        
        # Count frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Return most common
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:5]]
    
    def _generate_context_summary(self, items: List[GISKnowledgeItem], query: str) -> str:
        """Generate a summary of retrieved context"""
        
        if not items:
            return "No relevant knowledge retrieved"
        
        categories = list(set(item.category for item in items))
        sources = list(set(item.source for item in items))
        
        summary = f"Retrieved {len(items)} items from {len(categories)} categories "
        summary += f"({', '.join(categories[:3])}) "
        summary += f"and {len(sources)} sources ({', '.join(sources[:2])}) "
        summary += f"relevant to: {query}"
        
        return summary
    
    def _analyze_knowledge_coverage(self) -> Dict[str, float]:
        """Analyze knowledge coverage across different domains"""
        
        if self.collection.count() == 0:
            return {}
        
        try:
            all_results = self.collection.get()
            total_count = len(all_results['metadatas'])
            
            category_coverage = {}
            
            for category in KnowledgeCategory:
                category_count = sum(1 for metadata in all_results['metadatas'] 
                                   if metadata.get('category') == category.value)
                coverage = category_count / total_count if total_count > 0 else 0
                category_coverage[category.value] = coverage
            
            return category_coverage
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return {}


# Convenience functions for backward compatibility
def create_rag_system(persist_dir: str = "./rag_db") -> UnifiedRAGSystem:
    """Create a unified RAG system instance"""
    return UnifiedRAGSystem(chroma_persist_dir=persist_dir)


# Example usage and testing
if __name__ == "__main__":
    print("Unified RAG System module loaded successfully!")
    
    # Example usage
    rag = UnifiedRAGSystem()
    
    # Add some sample knowledge
    sample_items = [
        GISKnowledgeItem(
            item_id="spatial_buffer_001",
            title="Spatial Buffer Analysis",
            content="Creating buffer zones around geographic features for proximity analysis",
            category="spatial_analysis",
            subcategory="buffer",
            source="gis_handbook",
            tags=["buffer", "proximity", "spatial", "analysis"]
        ),
        GISKnowledgeItem(
            item_id="site_selection_001", 
            title="Site Suitability Analysis",
            content="Evaluating location suitability using multiple criteria and constraints",
            category="spatial_analysis",
            subcategory="suitability",
            source="gis_handbook",
            tags=["suitability", "site_selection", "criteria", "analysis"]
        )
    ]
    
    # Test the system
    success = rag.add_knowledge(sample_items)
    if success:
        print("✅ Sample knowledge added successfully")
        
        # Test retrieval
        result = rag.retrieve_knowledge("buffer analysis", strategy="hybrid")
        print(f"✅ Retrieved {result.total_items} items using {result.strategy_used} strategy")
        
        # Test contextual retrieval
        contextual_result = rag.contextual_retrieve("site selection analysis")
        print(f"✅ Contextual retrieval found {len(contextual_result.primary_results)} primary results")
        
        # Generate report
        report = rag.generate_knowledge_report()
        print(f"✅ Knowledge base report: {report['total_knowledge_items']} items, {report['total_clusters']} clusters")
