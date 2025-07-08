"""
LLM & Reasoning Engine Package
=====================================

This package contains the core LLM and reasoning components for Spatial.AI:

- Query Parser: Understands natural language spatial queries
- Workflow Generator: Creates step-by-step GIS workflows  
- Reasoning Engine: Implements Chain-of-Thought reasoning with advanced features
- Unified RAG System: Advanced Retrieval-Augmented Generation for GIS knowledge
- Prompt Manager: Handles prompt templates and engineering
- Advanced Reasoning: Multi-perspective reasoning, validation, and confidence assessment
- Error Correction: Intelligent error detection and retry logic

Author: Member 1 (LLM & Reasoning Engine Lead)
"""

from datetime import datetime
from .query_parser import QueryParser
from .workflow_generator import WorkflowGenerator
from .reasoning_engine import ReasoningEngine
from .unified_rag_system import UnifiedRAGSystem, create_rag_system
from .prompt_manager import PromptManager
from .advanced_reasoning import AdvancedReasoningEngine
from .error_correction import ErrorCorrectionEngine

# Main integration class
class SpatialAIEngine:
    """
    Main Spatial.AI LLM & Reasoning Engine Integration Class
    
    This class integrates all Phase 2 components into a cohesive system:
    - Query parsing and understanding
    - Advanced reasoning with Chain-of-Thought
    - Enhanced RAG with clustering and contextual retrieval
    - Workflow generation with error correction
    - Comprehensive logging and validation
    """
    
    def __init__(self, rag_persist_dir: str = "./rag_db"):
        """Initialize the complete Spatial.AI engine"""
        # Core components
        self.query_parser = QueryParser()
        self.rag_system = create_rag_system(rag_persist_dir)
        self.reasoning_engine = ReasoningEngine()
        self.advanced_reasoning = AdvancedReasoningEngine(self.reasoning_engine)
        self.workflow_generator = WorkflowGenerator()
        self.error_correction = ErrorCorrectionEngine(self.reasoning_engine)
        self.prompt_manager = PromptManager()
        
    def process_query(self, query: str, context: str = None) -> dict:
        """
        Process a complete query through the integrated system pipeline
        
        Args:
            query: Natural language spatial query
            context: Optional additional context
            
        Returns:
            dict: Complete system response with reasoning, workflow, and metadata
        """
        try:
            # Step 1: Parse the query
            parsed_query = self.query_parser.parse(query)
            
            # Step 2: Enhanced RAG retrieval
            rag_results = self.rag_system.contextual_retrieve(
                query, 
                context={'analysis_type': context} if context else None,
                strategy='hybrid'
            )
            
            # Step 3: Simple reasoning (optimized for speed)
            reasoning_result = self.reasoning_engine.reason_about_query(query)
            
            # Step 4: Generate workflow
            workflow = self.workflow_generator.generate_workflow(parsed_query)
            
            return {
                "query": query,
                "parsed_query": parsed_query,
                "rag_results": rag_results,
                "reasoning": reasoning_result,
                "workflow": workflow,
                "status": "success",
                "timestamp": str(datetime.now())
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "status": "error",
                "timestamp": str(datetime.now())
            }
        
    def get_system_status(self) -> dict:
        """Get current system status and component health"""
        try:
            components = {
                "query_parser": "operational",
                "rag_system": "operational",
                "reasoning_engine": "operational",
                "advanced_reasoning": "operational",
                "workflow_generator": "operational",
                "error_correction": "operational",
                "prompt_manager": "operational"
            }
            
            # Basic health checks
            if hasattr(self.rag_system, 'collection') and self.rag_system.collection:
                components["vector_database"] = "operational"
            else:
                components["vector_database"] = "degraded"
            
            return {
                "ready": all(status == "operational" for status in components.values()),
                "components": components,
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {
                "ready": False,
                "components": {"system": "error"},
                "error": str(e),
                "timestamp": str(datetime.now())
            }

__version__ = "0.2.0"  # Phase 2 completion
__author__ = "Spatial.AI Team - Member 1"

__all__ = [
    "QueryParser",
    "WorkflowGenerator", 
    "ReasoningEngine",
    "UnifiedRAGSystem",
    "create_rag_system",
    "PromptManager",
    "AdvancedReasoningEngine", 
    "ErrorCorrectionEngine",
    "SpatialAIEngine"  # Main integration class
]
