"""
LLM & Reasoning Engine Package
=====================================

This package contains the core LLM and reasoning components for Spatial.AI:

- Query Parser: Understands natural language spatial queries
- Workflow Generator: Creates step-by-step GIS workflows  
- Reasoning Engine: Implements Chain-of-Thought reasoning
- RAG System: Retrieval-Augmented Generation for GIS knowledge
- Prompt Manager: Handles prompt templates and engineering

Author: Member 1 (LLM & Reasoning Engine Lead)
"""

from .query_parser import QueryParser
from .workflow_generator import WorkflowGenerator
from .reasoning_engine import ReasoningEngine
from .rag_system import RAGSystem
from .prompt_manager import PromptManager

__version__ = "0.1.0"
__author__ = "Spatial.AI Team - Member 1"

__all__ = [
    "QueryParser",
    "WorkflowGenerator", 
    "ReasoningEngine",
    "RAGSystem",
    "PromptManager"
]
