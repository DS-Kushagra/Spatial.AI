"""
Reasoning Engine - Chain-of-Thought Implementation for Spatial Analysis
======================================================================

This module implements the Chain-of-Thought reasoning system that powers
the intelligent decision-making in Spatial.AI. It integrates with Groq LLM
to provide transparent, step-by-step reasoning for GIS workflow generation.

Key Features:
- Chain-of-Thought prompt engineering
- Reasoning trace generation and logging
- Error analysis and alternative suggestions
- Confidence assessment and uncertainty handling
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

# Groq integration
from groq import Groq
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from .query_parser import ParsedQuery, SpatialIntent
from .workflow_generator import GISWorkflow, WorkflowStep

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning operations"""
    QUERY_UNDERSTANDING = "query_understanding"
    WORKFLOW_PLANNING = "workflow_planning"
    STEP_VALIDATION = "step_validation"
    ERROR_ANALYSIS = "error_analysis"
    ALTERNATIVE_GENERATION = "alternative_generation"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"


@dataclass
class ReasoningStep:
    """Individual step in reasoning chain"""
    
    step_id: str
    reasoning_type: ReasoningType
    question: str
    reasoning: str
    conclusion: str
    confidence: float
    evidence: List[str] = None
    alternatives: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.alternatives is None:
            self.alternatives = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a spatial analysis"""
    
    trace_id: str
    original_query: str
    reasoning_steps: List[ReasoningStep]
    final_confidence: float
    total_reasoning_time: float
    llm_model: str
    created_at: str
    
    # Summary
    key_insights: List[str] = None
    assumptions: List[str] = None
    uncertainties: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.key_insights is None:
            self.key_insights = []
        if self.assumptions is None:
            self.assumptions = []
        if self.uncertainties is None:
            self.uncertainties = []
        if self.recommendations is None:
            self.recommendations = []


class ReasoningEngine:
    """
    Chain-of-Thought Reasoning Engine for Spatial Analysis
    
    This class implements sophisticated reasoning capabilities using Groq LLM
    to provide transparent, step-by-step decision-making for GIS workflows.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, model: str = None):
        """
        Initialize the reasoning engine
        
        Args:
            groq_api_key: Groq API key (if not provided, reads from environment)
            model: Groq model to use for reasoning (if not provided, reads from environment)
        """
        
        # Initialize Groq client
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Reasoning configuration
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        
        # Initialize prompt templates
        self._init_prompts()
        
        logger.info(f"Initialized ReasoningEngine with model: {model}")
    
    def _init_prompts(self):
        """Initialize Chain-of-Thought prompt templates"""
        
        # System prompt for spatial reasoning
        self.system_prompt = """You are an expert GIS analyst and spatial reasoning AI. Your role is to provide clear, logical, step-by-step reasoning for spatial analysis tasks.

REASONING PRINCIPLES:
1. Think step-by-step using Chain-of-Thought methodology
2. Explain your reasoning clearly and transparently  
3. Consider multiple perspectives and alternatives
4. Assess confidence levels and identify uncertainties
5. Ground reasoning in GIS and spatial analysis best practices

RESPONSE FORMAT:
- Start with understanding the spatial problem
- Break down complex requirements into logical steps
- Explain WHY each step is necessary
- Consider potential issues and alternatives
- Provide confidence assessment

Be precise, logical, and educational in your explanations."""

        # Query understanding prompt
        self.query_understanding_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Analyze this spatial query and explain your understanding:

Query: "{query}"

Please provide step-by-step reasoning covering:
1. What is the user trying to accomplish? (Spatial intent)
2. What are the key spatial components mentioned?
3. What constraints or requirements are specified?
4. What GIS data and operations will likely be needed?
5. What are the potential challenges or ambiguities?

Format your response as clear reasoning steps with explanations.""")
        ])
        
        # Workflow planning prompt
        self.workflow_planning_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Given this parsed spatial query, design a logical GIS workflow:

PARSED QUERY:
- Intent: {intent}
- Location: {location}
- Target Features: {features}
- Constraints: {constraints}
- Required Datasets: {datasets}

Please reason through:
1. What is the logical sequence of GIS operations needed?
2. Why is this sequence optimal?
3. What are the dependencies between steps?
4. What are potential failure points and alternatives?
5. How confident are you in this approach?

Provide detailed reasoning for each workflow step.""")
        ])
        
        # Step validation prompt
        self.step_validation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Validate this GIS workflow step and explain your reasoning:

WORKFLOW STEP:
Operation: {operation}
Description: {description}
Inputs: {inputs}
Outputs: {outputs}
Parameters: {parameters}

CONTEXT:
Previous Steps: {previous_steps}
Overall Goal: {goal}

Please reason through:
1. Is this step logically sound?
2. Are the inputs and outputs appropriate?
3. Are the parameters reasonable?
4. Does this step fit the overall workflow logic?
5. What could go wrong and how to handle it?

Provide detailed validation reasoning.""")
        ])
        
        # Error analysis prompt
        self.error_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Analyze this workflow execution error and provide reasoning:

ERROR DETAILS:
Step: {step_name}
Operation: {operation}
Error Message: {error_message}
Context: {context}

Please reason through:
1. What likely caused this error?
2. What are the possible root causes?
3. How critical is this error to the overall workflow?
4. What are viable alternatives or fixes?
5. How can similar errors be prevented?

Provide systematic error analysis reasoning.""")
        ])

    def reason_about_query(self, query: str) -> ReasoningStep:
        """
        Apply Chain-of-Thought reasoning to understand a spatial query
        
        Args:
            query: Natural language spatial query
            
        Returns:
            ReasoningStep: Detailed reasoning about query understanding
        """
        logger.info(f"Reasoning about query: {query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Enhanced prompt for better step-by-step reasoning
            enhanced_prompt = f"""You are an expert GIS analyst. Analyze this spatial query using clear step-by-step reasoning:

Query: "{query}"

Please provide detailed step-by-step reasoning in the following format:

STEP 1: Query Understanding
- What is the user trying to accomplish?
- What are the key spatial components?

STEP 2: Data Requirements  
- What datasets will be needed?
- What spatial operations are required?

STEP 3: Workflow Planning
- What is the logical sequence of operations?
- What are the dependencies between steps?

STEP 4: Constraints and Considerations
- What constraints must be considered?
- What are potential challenges?

STEP 5: Expected Outcomes
- What will the final result look like?
- How will success be measured?

Provide clear, detailed reasoning for each step."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            reasoning_text = response.choices[0].message.content
            
            # Extract confidence and other components
            confidence = self._extract_confidence(reasoning_text)
            evidence = self._extract_evidence(reasoning_text)
            conclusion = self._extract_conclusion(reasoning_text)
            alternatives = self._extract_alternatives(reasoning_text)
            
            reasoning_step = ReasoningStep(
                step_id="query_understanding_001",
                reasoning_type=ReasoningType.QUERY_UNDERSTANDING,
                question=f"How should I interpret this spatial query: '{query}'?",
                reasoning=reasoning_text,
                conclusion=conclusion,
                confidence=confidence,
                evidence=evidence,
                alternatives=alternatives
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Query reasoning completed in {elapsed_time:.2f}s, confidence: {confidence:.1%}")
            
            return reasoning_step
            
        except Exception as e:
            logger.error(f"Error in query reasoning: {str(e)}")
            return ReasoningStep(
                step_id="query_understanding_error",
                reasoning_type=ReasoningType.QUERY_UNDERSTANDING,
                question=f"How should I interpret this spatial query: '{query}'?",
                reasoning=f"Error occurred during reasoning: {str(e)}",
                conclusion="Unable to complete reasoning due to error",
                confidence=0.0,
                evidence=[],
                alternatives=["Manual analysis required", "Retry with simplified query"]
            )

    def reason_about_workflow(self, parsed_query: ParsedQuery) -> ReasoningStep:
        """
        Apply Chain-of-Thought reasoning to workflow planning
        
        Args:
            parsed_query: Structured query from QueryParser
            
        Returns:
            ReasoningStep: Detailed reasoning about workflow design
        """
        logger.info(f"Reasoning about workflow for intent: {parsed_query.intent}")
        
        start_time = time.time()
        
        try:
            # Prepare context for LLM
            messages = self.workflow_planning_prompt.format_messages(
                intent=parsed_query.intent.value,
                location=parsed_query.location or "Not specified",
                features=", ".join(parsed_query.target_features) or "Not specified",
                constraints=", ".join(parsed_query.constraints) or "None",
                datasets=", ".join(parsed_query.required_datasets) or "To be determined"
            )
            
            # Convert LangChain messages to Groq format
            groq_messages = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    role = "system" if msg.type == "system" else "user"
                else:
                    role = "user"
                groq_messages.append({"role": role, "content": msg.content})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            reasoning_text = response.choices[0].message.content
            confidence = self._extract_confidence(reasoning_text)
            
            reasoning_step = ReasoningStep(
                step_id="workflow_planning_001",
                reasoning_type=ReasoningType.WORKFLOW_PLANNING,
                question="What is the optimal GIS workflow for this spatial analysis?",
                reasoning=reasoning_text,
                conclusion=self._extract_conclusion(reasoning_text),
                confidence=confidence,
                evidence=self._extract_evidence(reasoning_text),
                alternatives=self._extract_alternatives(reasoning_text)
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Workflow reasoning completed in {elapsed_time:.2f}s")
            
            return reasoning_step
            
        except Exception as e:
            logger.error(f"Error in workflow reasoning: {str(e)}")
            return ReasoningStep(
                step_id="workflow_planning_error",
                reasoning_type=ReasoningType.WORKFLOW_PLANNING,
                question="What is the optimal GIS workflow for this spatial analysis?",
                reasoning=f"Error occurred during reasoning: {str(e)}",
                conclusion="Unable to complete workflow reasoning due to error",
                confidence=0.0
            )

    def validate_workflow_step(self, step: WorkflowStep, context: Dict[str, Any]) -> ReasoningStep:
        """
        Apply reasoning to validate a workflow step
        
        Args:
            step: Workflow step to validate
            context: Additional context about the workflow
            
        Returns:
            ReasoningStep: Validation reasoning
        """
        logger.info(f"Validating workflow step: {step.name}")
        
        try:
            messages = self.step_validation_prompt.format_messages(
                operation=step.operation.value,
                description=step.description,
                inputs=json.dumps(step.inputs, indent=2),
                outputs=json.dumps(step.outputs, indent=2),
                parameters=json.dumps(step.parameters, indent=2),
                previous_steps=context.get("previous_steps", "None"),
                goal=context.get("goal", "Not specified")
            )
            
            # Convert LangChain messages to Groq format
            groq_messages = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    role = "system" if msg.type == "system" else "user"
                else:
                    role = "user"
                groq_messages.append({"role": role, "content": msg.content})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            reasoning_text = response.choices[0].message.content
            confidence = self._extract_confidence(reasoning_text)
            
            return ReasoningStep(
                step_id=f"validation_{step.step_id}",
                reasoning_type=ReasoningType.STEP_VALIDATION,
                question=f"Is the workflow step '{step.name}' logically sound and appropriate?",
                reasoning=reasoning_text,
                conclusion=self._extract_conclusion(reasoning_text),
                confidence=confidence,
                evidence=self._extract_evidence(reasoning_text)
            )
            
        except Exception as e:
            logger.error(f"Error in step validation: {str(e)}")
            return ReasoningStep(
                step_id=f"validation_error_{step.step_id}",
                reasoning_type=ReasoningType.STEP_VALIDATION,
                question=f"Is the workflow step '{step.name}' logically sound?",
                reasoning=f"Validation error: {str(e)}",
                conclusion="Unable to validate step due to error",
                confidence=0.0
            )

    def analyze_error(self, step_name: str, operation: str, error_message: str, context: str = "") -> ReasoningStep:
        """
        Apply reasoning to analyze workflow execution errors
        
        Args:
            step_name: Name of the failed step
            operation: Operation that failed
            error_message: Error message from execution
            context: Additional context about the error
            
        Returns:
            ReasoningStep: Error analysis reasoning
        """
        logger.info(f"Analyzing error in step: {step_name}")
        
        try:
            messages = self.error_analysis_prompt.format_messages(
                step_name=step_name,
                operation=operation,
                error_message=error_message,
                context=context or "No additional context provided"
            )
            
            # Convert LangChain messages to Groq format
            groq_messages = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    role = "system" if msg.type == "system" else "user"
                else:
                    role = "user"
                groq_messages.append({"role": role, "content": msg.content})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            reasoning_text = response.choices[0].message.content
            
            return ReasoningStep(
                step_id=f"error_analysis_{int(time.time())}",
                reasoning_type=ReasoningType.ERROR_ANALYSIS,
                question=f"What caused the error in step '{step_name}' and how can it be resolved?",
                reasoning=reasoning_text,
                conclusion=self._extract_conclusion(reasoning_text),
                confidence=self._extract_confidence(reasoning_text),
                alternatives=self._extract_alternatives(reasoning_text)
            )
            
        except Exception as e:
            logger.error(f"Error in error analysis: {str(e)}")
            return ReasoningStep(
                step_id=f"error_analysis_failed_{int(time.time())}",
                reasoning_type=ReasoningType.ERROR_ANALYSIS,
                question=f"What caused the error in step '{step_name}'?",
                reasoning=f"Unable to analyze error: {str(e)}",
                conclusion="Error analysis failed",
                confidence=0.0
            )

    def generate_complete_reasoning_trace(self, query: str, parsed_query: ParsedQuery, workflow: GISWorkflow) -> ReasoningTrace:
        """
        Generate a complete reasoning trace for the entire analysis
        
        Args:
            query: Original natural language query
            parsed_query: Parsed query structure
            workflow: Generated workflow
            
        Returns:
            ReasoningTrace: Complete reasoning trace
        """
        logger.info("Generating complete reasoning trace")
        
        start_time = time.time()
        trace_id = f"trace_{int(time.time())}"
        
        reasoning_steps = []
        
        # Step 1: Query understanding
        query_reasoning = self.reason_about_query(query)
        reasoning_steps.append(query_reasoning)
        
        # Step 2: Workflow planning
        workflow_reasoning = self.reason_about_workflow(parsed_query)
        reasoning_steps.append(workflow_reasoning)
        
        # Step 3: Validate key workflow steps
        for i, step in enumerate(workflow.steps[:3]):  # Validate first 3 steps to avoid too many API calls
            context = {
                "previous_steps": [s.name for s in workflow.steps[:i]],
                "goal": workflow.description
            }
            validation = self.validate_workflow_step(step, context)
            reasoning_steps.append(validation)
        
        # Calculate overall confidence
        confidences = [step.confidence for step in reasoning_steps if step.confidence > 0]
        final_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        total_time = time.time() - start_time
        
        # Generate insights and recommendations
        insights, assumptions, uncertainties, recommendations = self._generate_trace_summary(reasoning_steps, workflow)
        
        trace = ReasoningTrace(
            trace_id=trace_id,
            original_query=query,
            reasoning_steps=reasoning_steps,
            final_confidence=final_confidence,
            total_reasoning_time=total_time,
            llm_model=self.model,
            created_at=datetime.now().isoformat(),
            key_insights=insights,
            assumptions=assumptions,
            uncertainties=uncertainties,
            recommendations=recommendations
        )
        
        logger.info(f"Generated reasoning trace with {len(reasoning_steps)} steps in {total_time:.2f}s")
        
        return trace

    def _extract_confidence(self, reasoning_text: str) -> float:
        """Extract confidence score from reasoning text"""
        
        # Look for explicit confidence mentions
        confidence_patterns = [
            r'confidence[:\s]+(\d+)%',
            r'confident[:\s]+(\d+)%',
            r'certainty[:\s]+(\d+)%',
            r'confidence[:\s]+(\d+\.\d+)',
            r'(\d+)%\s+confident'
        ]
        
        import re
        for pattern in confidence_patterns:
            match = re.search(pattern, reasoning_text.lower())
            if match:
                try:
                    confidence = float(match.group(1))
                    return confidence / 100.0 if confidence > 1.0 else confidence
                except:
                    continue
        
        # Estimate confidence from reasoning quality indicators
        quality_indicators = [
            'clear', 'obvious', 'certain', 'definitely', 'strongly',
            'logical', 'appropriate', 'optimal', 'best', 'reliable'
        ]
        
        uncertainty_indicators = [
            'uncertain', 'unclear', 'maybe', 'possibly', 'might',
            'could', 'assumptions', 'limitations', 'challenges'
        ]
        
        text_lower = reasoning_text.lower()
        quality_score = sum(1 for indicator in quality_indicators if indicator in text_lower)
        uncertainty_score = sum(1 for indicator in uncertainty_indicators if indicator in text_lower)
        
        # Simple heuristic
        base_confidence = 0.7
        quality_boost = min(quality_score * 0.05, 0.2)
        uncertainty_penalty = min(uncertainty_score * 0.1, 0.3)
        
        return max(0.1, min(1.0, base_confidence + quality_boost - uncertainty_penalty))

    def _extract_conclusion(self, reasoning_text: str) -> str:
        """Extract main conclusion from reasoning text"""
        
        # Look for conclusion indicators
        conclusion_patterns = [
            r'conclusion[:\s]+(.*?)(?:\n|$)',
            r'therefore[:\s]+(.*?)(?:\n|$)',
            r'in summary[:\s]+(.*?)(?:\n|$)',
            r'overall[:\s]+(.*?)(?:\n|$)'
        ]
        
        import re
        for pattern in conclusion_patterns:
            match = re.search(pattern, reasoning_text.lower())
            if match:
                return match.group(1).strip()
        
        # Fallback: take the last sentence
        sentences = reasoning_text.split('.')
        return sentences[-2].strip() if len(sentences) > 1 else reasoning_text[:200] + "..."

    def _extract_evidence(self, reasoning_text: str) -> List[str]:
        """Extract evidence points from reasoning text"""
        
        evidence = []
        
        # Look for evidence indicators
        evidence_patterns = [
            r'because\s+(.*?)(?:\.|,|\n)',
            r'since\s+(.*?)(?:\.|,|\n)',
            r'given that\s+(.*?)(?:\.|,|\n)',
            r'evidence[:\s]+(.*?)(?:\n|$)'
        ]
        
        import re
        for pattern in evidence_patterns:
            matches = re.findall(pattern, reasoning_text.lower())
            evidence.extend([match.strip() for match in matches])
        
        return evidence[:5]  # Limit to top 5 evidence points

    def _extract_alternatives(self, reasoning_text: str) -> List[str]:
        """Extract alternative approaches from reasoning text"""
        
        alternatives = []
        
        # Look for alternative indicators
        alternative_patterns = [
            r'alternatively[:\s]+(.*?)(?:\.|,|\n)',
            r'another option[:\s]+(.*?)(?:\.|,|\n)',
            r'could also[:\s]+(.*?)(?:\.|,|\n)',
            r'alternatives?[:\s]+(.*?)(?:\n|$)'
        ]
        
        import re
        for pattern in alternative_patterns:
            matches = re.findall(pattern, reasoning_text.lower())
            alternatives.extend([match.strip() for match in matches])
        
        return alternatives[:3]  # Limit to top 3 alternatives

    def _generate_trace_summary(self, reasoning_steps: List[ReasoningStep], workflow: GISWorkflow) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate summary insights from reasoning trace"""
        
        insights = [
            f"Query was interpreted with {reasoning_steps[0].confidence:.1%} confidence",
            f"Workflow planned with {len(workflow.steps)} steps",
            f"Analysis type: {workflow.intent.replace('_', ' ').title()}"
        ]
        
        assumptions = [
            "Data quality is consistent across all sources",
            "Coordinate reference systems are properly handled",
            "Analysis parameters are appropriate for the scale"
        ]
        
        uncertainties = []
        low_confidence_steps = [step for step in reasoning_steps if step.confidence < 0.7]
        if low_confidence_steps:
            uncertainties.append(f"{len(low_confidence_steps)} reasoning steps have below 70% confidence")
        
        recommendations = [
            "Review results for spatial accuracy and logical consistency",
            "Validate assumptions with domain experts if possible",
            "Consider sensitivity analysis for key parameters"
        ]
        
        return insights, assumptions, uncertainties, recommendations

    def to_json(self, trace: ReasoningTrace) -> str:
        """Convert reasoning trace to JSON"""
        
        # Convert dataclasses to dictionaries
        import json
        from dataclasses import asdict
        
        trace_dict = asdict(trace)
        
        # Convert enum values to strings
        for step in trace_dict['reasoning_steps']:
            step['reasoning_type'] = step['reasoning_type']
        
        return json.dumps(trace_dict, indent=2, default=str)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test reasoning engine
    reasoning_engine = ReasoningEngine()
    
    test_query = "Find best places to build schools in flood-free zones near highways in Kerala"
    
    print(f"Query: {test_query}")
    print("="*60)
    
    # Test query reasoning
    query_reasoning = reasoning_engine.reason_about_query(test_query)
    print(f"Query Reasoning Confidence: {query_reasoning.confidence:.1%}")
    print(f"Conclusion: {query_reasoning.conclusion}")
    print(f"Evidence: {query_reasoning.evidence}")
