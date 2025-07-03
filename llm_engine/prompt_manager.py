"""
Prompt Manager - Advanced Prompt Engineering for Spatial Analysis
================================================================

This module manages sophisticated prompt templates and engineering strategies
for spatial analysis tasks. It implements Chain-of-Thought prompting,
few-shot learning, and context-aware prompt generation.

Key Features:
- Chain-of-Thought prompt templates
- Few-shot learning examples  
- Dynamic prompt generation
- Context-aware prompt adaptation
- Prompt performance tracking
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts for different analysis stages"""
    QUERY_UNDERSTANDING = "query_understanding"
    WORKFLOW_PLANNING = "workflow_planning"
    STEP_REASONING = "step_reasoning"
    ERROR_ANALYSIS = "error_analysis"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"


@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    
    template_id: str
    name: str
    prompt_type: PromptType
    system_prompt: str
    user_template: str
    
    # Prompt engineering configuration
    few_shot_examples: List[Dict[str, str]] = None
    chain_of_thought_structure: List[str] = None
    context_variables: List[str] = None
    
    # Performance tracking
    usage_count: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    
    # Metadata
    created_at: str = None
    last_used: str = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.few_shot_examples is None:
            self.few_shot_examples = []
        if self.chain_of_thought_structure is None:
            self.chain_of_thought_structure = []
        if self.context_variables is None:
            self.context_variables = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class PromptContext:
    """Context information for prompt generation"""
    
    user_query: str
    analysis_type: str
    location: Optional[str] = None
    constraints: List[str] = None
    available_data: List[str] = None
    previous_reasoning: List[str] = None
    error_context: Optional[str] = None
    confidence_threshold: float = 0.7
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.available_data is None:
            self.available_data = []
        if self.previous_reasoning is None:
            self.previous_reasoning = []


class PromptManager:
    """
    Advanced Prompt Engineering Manager for Spatial Analysis
    
    This class manages sophisticated prompt templates and implements
    advanced prompting strategies including Chain-of-Thought reasoning,
    few-shot learning, and context-aware generation.
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the prompt manager
        
        Args:
            templates_path: Path to load custom prompt templates
        """
        
        self.templates: Dict[str, PromptTemplate] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize with default templates
        self._init_default_templates()
        
        # Load custom templates if provided
        if templates_path and os.path.exists(templates_path):
            self.load_templates(templates_path)
        
        logger.info(f"Initialized PromptManager with {len(self.templates)} templates")
    
    def _init_default_templates(self):
        """Initialize default prompt templates"""
        
        # Query Understanding Template
        query_understanding = PromptTemplate(
            template_id="query_understanding_v1",
            name="Spatial Query Understanding",
            prompt_type=PromptType.QUERY_UNDERSTANDING,
            system_prompt="""You are an expert GIS analyst specializing in spatial reasoning and analysis. Your role is to understand natural language spatial queries and break them down into structured components.

ANALYSIS METHODOLOGY:
1. Use Chain-of-Thought reasoning to work through the problem step-by-step
2. Identify spatial intent, geographic context, and analysis requirements
3. Consider data needs, constraints, and potential challenges
4. Assess confidence in your interpretation
5. Suggest clarifications if the query is ambiguous

RESPONSE STRUCTURE:
- Step 1: Intent Analysis
- Step 2: Geographic Context
- Step 3: Requirements Breakdown  
- Step 4: Data and Methods Assessment
- Step 5: Confidence and Clarifications

Be thorough, logical, and educational in your explanations.""",
            
            user_template="""Analyze this spatial query using Chain-of-Thought reasoning:

QUERY: "{user_query}"

Please work through this step-by-step:

1. INTENT ANALYSIS: What is the user trying to accomplish spatially?
   - Primary objective
   - Type of spatial analysis needed
   - Expected outcomes

2. GEOGRAPHIC CONTEXT: What spatial elements are involved?
   - Location/study area
   - Geographic features mentioned
   - Spatial relationships

3. REQUIREMENTS BREAKDOWN: What specific requirements can you identify?
   - Target features to analyze
   - Constraints and filters
   - Criteria for evaluation

4. DATA AND METHODS ASSESSMENT: What data and methods will be needed?
   - Required datasets
   - GIS operations
   - Analysis workflow outline

5. CONFIDENCE AND CLARIFICATIONS: How confident are you in this interpretation?
   - Confidence level (0-100%)
   - Ambiguities or assumptions
   - Recommended clarifications

Provide detailed reasoning for each step.""",
            
            chain_of_thought_structure=[
                "Intent Analysis",
                "Geographic Context", 
                "Requirements Breakdown",
                "Data and Methods Assessment",
                "Confidence and Clarifications"
            ],
            
            few_shot_examples=[
                {
                    "query": "Find best locations for new hospitals near highways in Mumbai",
                    "reasoning": """1. INTENT ANALYSIS: The user wants to perform site selection for hospital locations with accessibility constraints.
- Primary objective: Identify optimal hospital sites
- Type: Suitability analysis with proximity constraints
- Expected outcomes: Ranked list of suitable locations

2. GEOGRAPHIC CONTEXT: Focus on Mumbai city with transportation network consideration.
- Location: Mumbai metropolitan area
- Features: Healthcare facilities, road network, population centers
- Spatial relationships: Proximity to highways for accessibility

3. REQUIREMENTS BREAKDOWN: Specific criteria for hospital site selection.
- Target features: Potential hospital sites
- Constraints: Must be near highways
- Criteria: Accessibility, population served, existing healthcare gaps

4. DATA AND METHODS ASSESSMENT: Multi-layered analysis required.
- Required datasets: Administrative boundaries, road network, population data, existing hospitals
- GIS operations: Buffer analysis, suitability modeling, proximity analysis
- Workflow: Load data → Buffer highways → Exclude existing hospitals → Suitability analysis → Rank results

5. CONFIDENCE AND CLARIFICATIONS: High confidence with some clarifications needed.
- Confidence: 85%
- Assumptions: "Near highways" needs distance specification
- Clarifications: Define "near" distance, population served requirements, minimum site size"""
                }
            ],
            
            context_variables=["user_query"],
            tags=["query_understanding", "chain_of_thought", "spatial_analysis"]
        )
        
        # Workflow Planning Template
        workflow_planning = PromptTemplate(
            template_id="workflow_planning_v1",
            name="GIS Workflow Planning",
            prompt_type=PromptType.WORKFLOW_PLANNING,
            system_prompt="""You are a senior GIS analyst and workflow architect. Your expertise lies in designing efficient, logical GIS workflows that solve complex spatial problems.

WORKFLOW DESIGN PRINCIPLES:
1. Follow logical sequence and dependencies
2. Ensure data quality and validation steps
3. Consider error handling and alternatives
4. Optimize for performance and accuracy
5. Provide clear reasoning for each step

ANALYSIS APPROACH:
- Think step-by-step through the entire process
- Consider data preparation, analysis, and output phases
- Identify critical decision points and validations
- Plan for error scenarios and recovery
- Estimate complexity and runtime

Your goal is to create robust, executable workflows with transparent reasoning.""",
            
            user_template="""Design a detailed GIS workflow for this spatial analysis:

ANALYSIS REQUIREMENTS:
- Intent: {analysis_type}
- Location: {location}
- User Query: "{user_query}"
- Constraints: {constraints}
- Available Data: {available_data}

Please design the workflow using this Chain-of-Thought approach:

1. WORKFLOW OVERVIEW: What is the high-level approach?
   - Main analysis phases
   - Key decision points
   - Success criteria

2. DATA PREPARATION STEPS: How will you prepare the data?
   - Data loading and validation
   - Coordinate system handling
   - Quality checks and cleaning

3. CORE ANALYSIS STEPS: What are the main analytical operations?
   - Sequential GIS operations
   - Parameters and thresholds
   - Intermediate outputs

4. VALIDATION AND QC: How will you ensure quality results?
   - Result validation methods
   - Quality control checks
   - Error detection

5. OUTPUT AND VISUALIZATION: How will results be presented?
   - Output formats
   - Visualization strategy
   - Delivery methods

6. REASONING AND ALTERNATIVES: Why this approach?
   - Rationale for each major step
   - Alternative methods considered
   - Confidence assessment

Provide detailed step-by-step workflow with clear reasoning.""",
            
            chain_of_thought_structure=[
                "Workflow Overview",
                "Data Preparation",
                "Core Analysis",
                "Validation and QC",
                "Output and Visualization",
                "Reasoning and Alternatives"
            ],
            
            context_variables=["user_query", "analysis_type", "location", "constraints", "available_data"],
            tags=["workflow_planning", "gis_operations", "methodology"]
        )
        
        # Step Reasoning Template
        step_reasoning = PromptTemplate(
            template_id="step_reasoning_v1",
            name="GIS Step Reasoning",
            prompt_type=PromptType.STEP_REASONING,
            system_prompt="""You are a GIS operations expert who provides detailed reasoning for individual workflow steps. Your role is to explain WHY each step is necessary and HOW it contributes to the overall analysis.

REASONING APPROACH:
1. Explain the purpose and necessity of the step
2. Justify the chosen method and parameters
3. Consider the step's role in the larger workflow
4. Identify potential issues and mitigation strategies
5. Assess the step's contribution to result quality

Be precise, educational, and thorough in your explanations.""",
            
            user_template="""Provide detailed reasoning for this GIS workflow step:

STEP DETAILS:
- Operation: {operation}
- Description: {description}
- Inputs: {inputs}
- Outputs: {outputs}
- Parameters: {parameters}

WORKFLOW CONTEXT:
- Overall Goal: {goal}
- Previous Steps: {previous_steps}
- Subsequent Steps: {next_steps}

Please reason through:

1. PURPOSE: Why is this step necessary?
   - Role in achieving the overall goal
   - What problem does it solve?
   - What would happen if skipped?

2. METHOD JUSTIFICATION: Why this specific approach?
   - Why this operation type?
   - Parameter choices and rationale
   - Alternative methods considered

3. INPUT/OUTPUT LOGIC: How do inputs transform to outputs?
   - Data transformation process
   - Expected output characteristics
   - Quality implications

4. WORKFLOW INTEGRATION: How does this fit the larger workflow?
   - Dependencies on previous steps
   - Contribution to subsequent steps
   - Critical path considerations

5. RISK ASSESSMENT: What could go wrong?
   - Potential failure modes
   - Error indicators to watch for
   - Mitigation strategies

Provide comprehensive reasoning for this workflow step.""",
            
            context_variables=["operation", "description", "inputs", "outputs", "parameters", "goal", "previous_steps", "next_steps"],
            tags=["step_reasoning", "gis_operations", "validation"]
        )
        
        # Error Analysis Template
        error_analysis = PromptTemplate(
            template_id="error_analysis_v1",
            name="GIS Error Analysis",
            prompt_type=PromptType.ERROR_ANALYSIS,
            system_prompt="""You are a GIS troubleshooting expert specializing in spatial analysis errors. Your role is to diagnose problems, identify root causes, and recommend solutions.

DIAGNOSTIC APPROACH:
1. Analyze error symptoms and context
2. Identify most likely root causes
3. Consider data, parameter, and method issues
4. Recommend specific solutions and alternatives
5. Provide prevention strategies for similar errors

Be systematic, thorough, and solution-oriented in your analysis.""",
            
            user_template="""Analyze this GIS workflow error and provide solutions:

ERROR DETAILS:
- Failed Step: {step_name}
- Operation: {operation}
- Error Message: {error_message}
- Context: {error_context}

WORKFLOW CONTEXT:
- Previous Steps: {previous_steps}
- Data Inputs: {inputs}
- Parameters Used: {parameters}

Please provide systematic error analysis:

1. ERROR INTERPRETATION: What does this error mean?
   - Error type classification
   - Likely immediate cause
   - Severity assessment

2. ROOT CAUSE ANALYSIS: What are the underlying issues?
   - Data-related causes
   - Parameter-related causes
   - Method-related causes
   - Environmental causes

3. DIAGNOSTIC QUESTIONS: What should be investigated?
   - Data quality checks needed
   - Parameter validations required
   - System resource considerations

4. SOLUTION RECOMMENDATIONS: How can this be fixed?
   - Immediate fixes for this error
   - Parameter adjustments needed
   - Alternative approaches

5. PREVENTION STRATEGIES: How to avoid similar errors?
   - Validation steps to add
   - Best practices to follow
   - Monitoring recommendations

Provide actionable solutions and clear prevention guidance.""",
            
            context_variables=["step_name", "operation", "error_message", "error_context", "previous_steps", "inputs", "parameters"],
            tags=["error_analysis", "troubleshooting", "debugging"]
        )
        
        # Add templates to collection
        templates = [query_understanding, workflow_planning, step_reasoning, error_analysis]
        
        for template in templates:
            self.templates[template.template_id] = template
        
        logger.info(f"Initialized {len(templates)} default prompt templates")
    
    def generate_prompt(self, 
                       prompt_type: PromptType, 
                       context: PromptContext,
                       template_id: Optional[str] = None,
                       include_rag_context: bool = True) -> Tuple[str, str]:
        """
        Generate a complete prompt for a specific analysis task
        
        Args:
            prompt_type: Type of prompt to generate
            context: Context information for prompt generation
            template_id: Specific template to use (optional)
            include_rag_context: Whether to include RAG context
            
        Returns:
            Tuple[str, str]: (system_prompt, user_prompt)
        """
        
        # Select appropriate template
        template = self._select_template(prompt_type, template_id)
        if not template:
            raise ValueError(f"No template found for prompt type: {prompt_type}")
        
        # Build context variables
        context_vars = self._build_context_variables(context)
        
        # Generate user prompt from template
        user_prompt = template.user_template.format(**context_vars)
        
        # Add few-shot examples if available
        if template.few_shot_examples:
            user_prompt = self._add_few_shot_examples(user_prompt, template.few_shot_examples)
        
        # Update usage statistics
        template.usage_count += 1
        template.last_used = datetime.now().isoformat()
        
        logger.info(f"Generated prompt using template: {template.name}")
        
        return template.system_prompt, user_prompt
    
    def _select_template(self, prompt_type: PromptType, template_id: Optional[str]) -> Optional[PromptTemplate]:
        """Select appropriate template based on type and ID"""
        
        if template_id and template_id in self.templates:
            return self.templates[template_id]
        
        # Find best template for prompt type
        matching_templates = [
            template for template in self.templates.values()
            if template.prompt_type == prompt_type
        ]
        
        if not matching_templates:
            return None
        
        # Return template with highest success rate
        return max(matching_templates, key=lambda t: t.success_rate)
    
    def _build_context_variables(self, context: PromptContext) -> Dict[str, str]:
        """Build context variables for template formatting"""
        
        context_vars = {
            "user_query": context.user_query,
            "analysis_type": context.analysis_type,
            "location": context.location or "Not specified",
            "constraints": ", ".join(context.constraints) if context.constraints else "None specified",
            "available_data": ", ".join(context.available_data) if context.available_data else "To be determined",
            "error_context": context.error_context or "No additional context",
            "previous_reasoning": "\n".join(context.previous_reasoning) if context.previous_reasoning else "No previous reasoning available"
        }
        
        # Add placeholder values for missing variables
        placeholder_vars = {
            "operation": "Not specified",
            "description": "Not provided",
            "inputs": "{}",
            "outputs": "{}",
            "parameters": "{}",
            "goal": context.user_query,
            "previous_steps": "None",
            "next_steps": "To be determined",
            "step_name": "Unknown step",
            "error_message": "No error message provided"
        }
        
        context_vars.update(placeholder_vars)
        
        return context_vars
    
    def _add_few_shot_examples(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        """Add few-shot examples to the prompt"""
        
        if not examples:
            return prompt
        
        examples_section = "\n\n=== EXAMPLES ===\n"
        
        for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples
            examples_section += f"\nExample {i}:\n"
            examples_section += f"Query: {example.get('query', 'N/A')}\n"
            examples_section += f"Reasoning: {example.get('reasoning', 'N/A')}\n"
        
        examples_section += "\n=== END EXAMPLES ===\n\n"
        examples_section += "Now analyze the current query using similar reasoning:\n\n"
        
        return examples_section + prompt
    
    def record_prompt_performance(self, 
                                 template_id: str, 
                                 success: bool, 
                                 confidence: float,
                                 execution_time: float,
                                 feedback: Optional[str] = None):
        """Record performance metrics for prompt templates"""
        
        if template_id not in self.templates:
            logger.warning(f"Template {template_id} not found for performance recording")
            return
        
        template = self.templates[template_id]
        
        # Update template metrics
        if template.usage_count > 0:
            # Running average for success rate and confidence
            current_success_rate = template.success_rate
            current_avg_confidence = template.avg_confidence
            
            # Update success rate
            template.success_rate = (
                (current_success_rate * (template.usage_count - 1) + (1.0 if success else 0.0)) 
                / template.usage_count
            )
            
            # Update average confidence
            template.avg_confidence = (
                (current_avg_confidence * (template.usage_count - 1) + confidence) 
                / template.usage_count
            )
        else:
            template.success_rate = 1.0 if success else 0.0
            template.avg_confidence = confidence
        
        # Record in history
        performance_record = {
            "template_id": template_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "confidence": confidence,
            "execution_time": execution_time,
            "feedback": feedback
        }
        
        self.performance_history.append(performance_record)
        
        logger.info(f"Recorded performance for {template_id}: success={success}, confidence={confidence:.2f}")
    
    def get_template_performance(self, template_id: str) -> Dict[str, Any]:
        """Get performance statistics for a template"""
        
        if template_id not in self.templates:
            return {"error": f"Template {template_id} not found"}
        
        template = self.templates[template_id]
        
        # Get recent performance history
        recent_history = [
            record for record in self.performance_history[-50:]  # Last 50 uses
            if record["template_id"] == template_id
        ]
        
        return {
            "template_id": template_id,
            "name": template.name,
            "usage_count": template.usage_count,
            "success_rate": template.success_rate,
            "avg_confidence": template.avg_confidence,
            "last_used": template.last_used,
            "recent_performance": recent_history
        }
    
    def optimize_templates(self) -> Dict[str, str]:
        """Analyze template performance and suggest optimizations"""
        
        optimizations = {}
        
        for template_id, template in self.templates.items():
            if template.usage_count < 5:
                optimizations[template_id] = "Insufficient usage data for optimization"
                continue
            
            suggestions = []
            
            # Low success rate
            if template.success_rate < 0.7:
                suggestions.append("Consider revising prompt structure or examples")
            
            # Low confidence
            if template.avg_confidence < 0.6:
                suggestions.append("Add more specific guidance or constraints")
            
            # High performance
            if template.success_rate > 0.9 and template.avg_confidence > 0.8:
                suggestions.append("Template performing well - consider as base for new templates")
            
            optimizations[template_id] = "; ".join(suggestions) if suggestions else "Template performing adequately"
        
        return optimizations
    
    def save_templates(self, output_path: str) -> bool:
        """Save prompt templates to file"""
        
        try:
            templates_data = {}
            
            for template_id, template in self.templates.items():
                templates_data[template_id] = {
                    "name": template.name,
                    "prompt_type": template.prompt_type.value,
                    "system_prompt": template.system_prompt,
                    "user_template": template.user_template,
                    "few_shot_examples": template.few_shot_examples,
                    "chain_of_thought_structure": template.chain_of_thought_structure,
                    "context_variables": template.context_variables,
                    "usage_count": template.usage_count,
                    "success_rate": template.success_rate,
                    "avg_confidence": template.avg_confidence,
                    "created_at": template.created_at,
                    "last_used": template.last_used,
                    "tags": template.tags
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(templates_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(templates_data)} templates to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")
            return False
    
    def load_templates(self, input_path: str) -> bool:
        """Load prompt templates from file"""
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            loaded_count = 0
            
            for template_id, data in templates_data.items():
                template = PromptTemplate(
                    template_id=template_id,
                    name=data["name"],
                    prompt_type=PromptType(data["prompt_type"]),
                    system_prompt=data["system_prompt"],
                    user_template=data["user_template"],
                    few_shot_examples=data.get("few_shot_examples", []),
                    chain_of_thought_structure=data.get("chain_of_thought_structure", []),
                    context_variables=data.get("context_variables", []),
                    usage_count=data.get("usage_count", 0),
                    success_rate=data.get("success_rate", 0.0),
                    avg_confidence=data.get("avg_confidence", 0.0),
                    created_at=data.get("created_at"),
                    last_used=data.get("last_used"),
                    tags=data.get("tags", [])
                )
                
                self.templates[template_id] = template
                loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} templates from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Test prompt manager
    prompt_manager = PromptManager()
    
    # Test context
    context = PromptContext(
        user_query="Find best places to build schools in flood-free zones near highways in Kerala",
        analysis_type="site_selection",
        location="Kerala",
        constraints=["flood-free zones", "near highways"],
        available_data=["administrative_boundaries", "road_network", "flood_zones"]
    )
    
    # Generate query understanding prompt
    system_prompt, user_prompt = prompt_manager.generate_prompt(
        PromptType.QUERY_UNDERSTANDING,
        context
    )
    
    print("SYSTEM PROMPT:")
    print("=" * 50)
    print(system_prompt)
    print("\nUSER PROMPT:")
    print("=" * 50)
    print(user_prompt)
    
    # Test performance recording
    prompt_manager.record_prompt_performance(
        "query_understanding_v1",
        success=True,
        confidence=0.85,
        execution_time=2.3,
        feedback="Good reasoning quality"
    )
    
    # Get performance stats
    perf_stats = prompt_manager.get_template_performance("query_understanding_v1")
    print(f"\nTemplate Performance: {perf_stats}")
