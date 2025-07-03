"""
Workflow Generator - Converts Parsed Queries to GIS Workflows
============================================================

This module takes parsed spatial queries and generates step-by-step GIS workflows
that can be executed by the GIS engine. It creates JSON/YAML workflow definitions
with detailed reasoning for each step.

Key Features:
- Converts structured queries to executable workflows
- Generates reasoning explanations for each step
- Handles error scenarios and alternatives
- Creates reusable workflow templates
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime
import logging

from .query_parser import ParsedQuery, SpatialIntent

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of GIS operations"""
    DATA_LOAD = "data_load"
    BUFFER = "buffer" 
    INTERSECT = "intersect"
    UNION = "union"
    CLIP = "clip"
    DISSOLVE = "dissolve"
    SELECT = "select"
    PROXIMITY = "proximity"
    SUITABILITY = "suitability"
    OVERLAY = "overlay"
    RASTER_CALC = "raster_calc"
    SLOPE_ANALYSIS = "slope_analysis"
    FLOOD_MODEL = "flood_model"
    VISUALIZATION = "visualization"
    EXPORT = "export"


@dataclass
class WorkflowStep:
    """Individual step in a GIS workflow"""
    
    step_id: str
    operation: OperationType
    name: str
    description: str
    
    # Input/Output
    inputs: Dict[str, Any]
    outputs: Dict[str, str]
    parameters: Dict[str, Any]
    
    # Dependencies and execution
    depends_on: List[str] = None
    optional: bool = False
    timeout: int = 300  # seconds
    
    # Reasoning and metadata
    reasoning: str = ""
    alternatives: List[str] = None
    error_handling: Dict[str, str] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.alternatives is None:
            self.alternatives = []
        if self.error_handling is None:
            self.error_handling = {}


@dataclass 
class GISWorkflow:
    """Complete GIS workflow definition"""
    
    # Workflow metadata
    workflow_id: str
    name: str
    description: str
    created_at: str
    
    # Query context
    original_query: str
    parsed_query: Dict[str, Any]
    intent: str
    
    # Workflow definition
    steps: List[WorkflowStep]
    dependencies: Dict[str, List[str]]
    
    # Data requirements
    input_datasets: List[str]
    output_formats: List[str]
    
    # Execution metadata
    estimated_runtime: int = 0  # seconds
    complexity_score: float = 0.0
    confidence_score: float = 0.0
    
    # Reasoning
    reasoning_chain: List[str] = None
    assumptions: List[str] = None
    limitations: List[str] = None
    
    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.assumptions is None:
            self.assumptions = []
        if self.limitations is None:
            self.limitations = []


class WorkflowGenerator:
    """
    Generates executable GIS workflows from parsed spatial queries
    
    This class implements the core logic for converting natural language
    requirements into step-by-step GIS operations with detailed reasoning.
    """
    
    def __init__(self):
        """Initialize workflow generator with operation templates"""
        
        # Operation templates define how to execute each type of analysis
        self.operation_templates = {
            SpatialIntent.FIND_LOCATIONS: self._template_find_locations,
            SpatialIntent.SUITABILITY_ANALYSIS: self._template_suitability_analysis,
            SpatialIntent.MAP_ANALYSIS: self._template_map_analysis,
            SpatialIntent.RISK_ASSESSMENT: self._template_risk_assessment,
            SpatialIntent.PROXIMITY_ANALYSIS: self._template_proximity_analysis,
            SpatialIntent.BUFFER_ANALYSIS: self._template_buffer_analysis
        }
        
        # Dataset configurations
        self.dataset_configs = {
            'administrative_boundaries': {
                'source': 'bhoonidhi',
                'type': 'vector',
                'format': 'geojson',
                'load_method': 'api_request'
            },
            'road_network': {
                'source': 'osm',
                'type': 'vector', 
                'format': 'geojson',
                'load_method': 'overpass_query'
            },
            'dem': {
                'source': 'srtm',
                'type': 'raster',
                'format': 'tiff',
                'load_method': 'download'
            },
            'land_cover': {
                'source': 'bhoonidhi',
                'type': 'raster',
                'format': 'tiff',
                'load_method': 'api_request'
            }
        }
        
        # Common operation parameters
        self.default_parameters = {
            'buffer_distance': '500m',
            'suitability_weights': {'distance_roads': 0.3, 'land_use': 0.4, 'elevation': 0.3},
            'flood_return_period': 100,  # years
            'slope_threshold': 15  # degrees
        }
    
    def generate_workflow(self, parsed_query: ParsedQuery) -> GISWorkflow:
        """
        Generate a complete GIS workflow from a parsed query
        
        Args:
            parsed_query: Structured query from QueryParser
            
        Returns:
            GISWorkflow: Complete executable workflow
        """
        logger.info(f"Generating workflow for intent: {parsed_query.intent}")
        
        # Initialize workflow
        workflow = GISWorkflow(
            workflow_id=str(uuid.uuid4()),
            name=self._generate_workflow_name(parsed_query),
            description=self._generate_workflow_description(parsed_query),
            created_at=datetime.now().isoformat(),
            original_query=parsed_query.raw_query,
            parsed_query=asdict(parsed_query),
            intent=parsed_query.intent.value,
            steps=[],
            dependencies={},
            input_datasets=parsed_query.required_datasets,
            output_formats=['geojson', 'png', 'html']
        )
        
        # Generate workflow steps based on intent
        if parsed_query.intent in self.operation_templates:
            template_func = self.operation_templates[parsed_query.intent]
            workflow.steps = template_func(parsed_query)
        else:
            logger.warning(f"No template for intent: {parsed_query.intent}")
            workflow.steps = self._template_generic_analysis(parsed_query)
        
        # Build dependencies
        workflow.dependencies = self._build_dependencies(workflow.steps)
        
        # Add reasoning chain
        workflow.reasoning_chain = self._generate_reasoning_chain(parsed_query, workflow.steps)
        
        # Calculate metadata
        workflow.complexity_score = self._calculate_complexity(workflow.steps)
        workflow.estimated_runtime = self._estimate_runtime(workflow.steps)
        workflow.confidence_score = parsed_query.confidence_score
        
        # Add assumptions and limitations
        workflow.assumptions = self._identify_assumptions(parsed_query)
        workflow.limitations = self._identify_limitations(parsed_query)
        
        logger.info(f"Generated workflow with {len(workflow.steps)} steps")
        
        return workflow
    
    def _template_find_locations(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Template for location finding workflows (site selection)"""
        
        steps = []
        step_counter = 1
        
        # Step 1: Load administrative boundaries
        if query.location:
            steps.append(WorkflowStep(
                step_id=f"step_{step_counter:02d}",
                operation=OperationType.DATA_LOAD,
                name="Load Administrative Boundaries",
                description=f"Load {query.location} administrative boundaries",
                inputs={},
                outputs={"boundaries": "admin_boundaries.geojson"},
                parameters={
                    "location": query.location,
                    "admin_level": query.admin_level or "state"
                },
                reasoning=f"Need to define the study area boundaries for {query.location} to constrain the analysis"
            ))
            step_counter += 1
        
        # Step 2: Load relevant datasets
        for dataset in query.required_datasets:
            if dataset in self.dataset_configs:
                config = self.dataset_configs[dataset]
                steps.append(WorkflowStep(
                    step_id=f"step_{step_counter:02d}",
                    operation=OperationType.DATA_LOAD,
                    name=f"Load {dataset.replace('_', ' ').title()}",
                    description=f"Load {dataset} from {config['source']}",
                    inputs={"boundaries": "admin_boundaries.geojson"} if query.location else {},
                    outputs={dataset: f"{dataset}.{config['format']}"},
                    parameters={
                        "dataset": dataset,
                        "source": config['source'],
                        "format": config['format']
                    },
                    depends_on=["step_01"] if query.location else [],
                    reasoning=f"Required {dataset} data for the analysis based on query requirements"
                ))
                step_counter += 1
        
        # Step 3: Apply constraints (avoid flood zones, etc.)
        if query.constraints:
            for constraint in query.constraints:
                if 'flood' in constraint.lower():
                    steps.append(WorkflowStep(
                        step_id=f"step_{step_counter:02d}",
                        operation=OperationType.SELECT,
                        name="Exclude Flood Zones",
                        description="Remove flood-prone areas from consideration",
                        inputs={"areas": "suitable_areas.geojson"},
                        outputs={"flood_free_areas": "flood_free_areas.geojson"},
                        parameters={"constraint": "flood_risk < 0.3"},
                        depends_on=[f"step_{step_counter-1:02d}"],
                        reasoning=f"Excluding flood zones as specified: '{constraint}'"
                    ))
                    step_counter += 1
        
        # Step 4: Proximity analysis
        if query.proximity_features or 'road' in str(query.required_datasets):
            distance = query.buffer_distance or self.default_parameters['buffer_distance']
            steps.append(WorkflowStep(
                step_id=f"step_{step_counter:02d}",
                operation=OperationType.PROXIMITY,
                name="Calculate Proximity to Roads",
                description=f"Calculate distance to roads within {distance}",
                inputs={"roads": "road_network.geojson"},
                outputs={"road_proximity": "road_proximity.tiff"},
                parameters={"max_distance": distance},
                reasoning=f"Proximity to roads is important for accessibility, using {distance} threshold"
            ))
            step_counter += 1
        
        # Step 5: Suitability analysis
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.SUITABILITY,
            name="Multi-Criteria Suitability Analysis",
            description="Combine all criteria to identify suitable locations",
            inputs={
                "criteria_layers": ["road_proximity.tiff", "land_use.tiff"],
                "weights": self.default_parameters['suitability_weights']
            },
            outputs={"suitability_map": "suitability_results.tiff"},
            parameters={
                "method": "weighted_overlay",
                "output_classes": 5
            },
            depends_on=[f"step_{i:02d}" for i in range(2, step_counter)],
            reasoning="Combining all spatial criteria using weighted overlay to identify most suitable locations"
        ))
        step_counter += 1
        
        # Step 6: Extract best locations
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.SELECT,
            name="Extract Top Suitable Sites",
            description="Select the highest ranked suitable locations",
            inputs={"suitability": "suitability_results.tiff"},
            outputs={"best_sites": "selected_sites.geojson"},
            parameters={
                "threshold": "top_10_percent",
                "min_area": "1000_sqm"
            },
            depends_on=[f"step_{step_counter-1:02d}"],
            reasoning="Selecting top 10% of suitable areas that meet minimum size requirements"
        ))
        step_counter += 1
        
        # Step 7: Visualization
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.VISUALIZATION,
            name="Create Interactive Map",
            description="Generate visualization of results",
            inputs={
                "sites": "selected_sites.geojson",
                "suitability": "suitability_results.tiff",
                "boundaries": "admin_boundaries.geojson"
            },
            outputs={"map": "results_map.html"},
            parameters={
                "map_type": "interactive",
                "base_layer": "openstreetmap"
            },
            depends_on=[f"step_{step_counter-1:02d}"],
            reasoning="Creating interactive visualization to display results and enable exploration"
        ))
        
        return steps
    
    def _template_suitability_analysis(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Template for suitability analysis workflows"""
        # Similar structure to find_locations but with more emphasis on criteria weighting
        return self._template_find_locations(query)  # Reuse for now, can specialize later
    
    def _template_map_analysis(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Template for mapping and visualization workflows"""
        
        steps = []
        step_counter = 1
        
        # Load data
        for dataset in query.required_datasets:
            if dataset in self.dataset_configs:
                config = self.dataset_configs[dataset]
                steps.append(WorkflowStep(
                    step_id=f"step_{step_counter:02d}",
                    operation=OperationType.DATA_LOAD,
                    name=f"Load {dataset.replace('_', ' ').title()}",
                    description=f"Load {dataset} for mapping",
                    inputs={},
                    outputs={dataset: f"{dataset}.{config['format']}"},
                    parameters={"dataset": dataset},
                    reasoning=f"Loading {dataset} as required for the mapping analysis"
                ))
                step_counter += 1
        
        # Create map
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.VISUALIZATION,
            name="Create Thematic Map",
            description="Generate thematic visualization",
            inputs={dataset: f"{dataset}.{self.dataset_configs[dataset]['format']}" 
                   for dataset in query.required_datasets if dataset in self.dataset_configs},
            outputs={"map": "thematic_map.html"},
            parameters={"style": "choropleth"},
            reasoning="Creating thematic map to visualize the spatial patterns"
        ))
        
        return steps
    
    def _template_risk_assessment(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Template for risk assessment workflows"""
        
        steps = []
        step_counter = 1
        
        # Load elevation data
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.DATA_LOAD,
            name="Load Elevation Data",
            description="Load Digital Elevation Model",
            inputs={},
            outputs={"dem": "elevation.tiff"},
            parameters={"dataset": "dem"},
            reasoning="Elevation data is fundamental for flood risk modeling"
        ))
        step_counter += 1
        
        # Calculate flow accumulation
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.RASTER_CALC,
            name="Calculate Flow Accumulation",
            description="Determine water flow patterns",
            inputs={"dem": "elevation.tiff"},
            outputs={"flow_accumulation": "flow_accumulation.tiff"},
            parameters={"algorithm": "d8_flow_accumulation"},
            depends_on=["step_01"],
            reasoning="Flow accumulation identifies areas where water naturally collects"
        ))
        step_counter += 1
        
        # Flood modeling
        if 'flood' in query.raw_query.lower():
            steps.append(WorkflowStep(
                step_id=f"step_{step_counter:02d}",
                operation=OperationType.FLOOD_MODEL,
                name="Model Flood Inundation",
                description="Calculate flood risk zones",
                inputs={
                    "dem": "elevation.tiff",
                    "flow_accumulation": "flow_accumulation.tiff"
                },
                outputs={"flood_zones": "flood_risk_zones.geojson"},
                parameters={
                    "return_period": self.default_parameters['flood_return_period'],
                    "method": "height_above_nearest_drainage"
                },
                depends_on=["step_02"],
                reasoning=f"Modeling {self.default_parameters['flood_return_period']}-year flood scenarios"
            ))
            step_counter += 1
        
        # Risk classification
        steps.append(WorkflowStep(
            step_id=f"step_{step_counter:02d}",
            operation=OperationType.RASTER_CALC,
            name="Classify Risk Levels",
            description="Categorize risk into levels (Low, Medium, High)",
            inputs={"flood_zones": "flood_risk_zones.geojson"},
            outputs={"risk_classification": "risk_levels.tiff"},
            parameters={
                "classes": ["Low", "Medium", "High", "Very High"],
                "thresholds": [0.2, 0.5, 0.8, 1.0]
            },
            depends_on=[f"step_{step_counter-1:02d}"],
            reasoning="Classifying continuous risk values into interpretable categories"
        ))
        
        return steps
    
    def _template_proximity_analysis(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Template for proximity analysis workflows"""
        
        steps = []
        distance = query.buffer_distance or "1km"
        
        # Load reference features
        steps.append(WorkflowStep(
            step_id="step_01",
            operation=OperationType.DATA_LOAD,
            name="Load Reference Features",
            description="Load features for proximity analysis",
            inputs={},
            outputs={"features": "reference_features.geojson"},
            parameters={"features": query.proximity_features},
            reasoning=f"Loading {query.proximity_features} as reference for proximity calculation"
        ))
        
        # Calculate proximity
        steps.append(WorkflowStep(
            step_id="step_02",
            operation=OperationType.PROXIMITY,
            name="Calculate Distance",
            description=f"Calculate distance within {distance}",
            inputs={"features": "reference_features.geojson"},
            outputs={"proximity": "proximity_analysis.tiff"},
            parameters={"max_distance": distance},
            depends_on=["step_01"],
            reasoning=f"Calculating proximity within {distance} as specified in query"
        ))
        
        return steps
    
    def _template_buffer_analysis(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Template for buffer analysis workflows"""
        
        steps = []
        distance = query.buffer_distance or "500m"
        
        # Load features to buffer
        steps.append(WorkflowStep(
            step_id="step_01",
            operation=OperationType.DATA_LOAD,
            name="Load Features",
            description="Load features for buffer analysis",
            inputs={},
            outputs={"features": "input_features.geojson"},
            parameters={},
            reasoning="Loading input features for buffer creation"
        ))
        
        # Create buffer
        steps.append(WorkflowStep(
            step_id="step_02",
            operation=OperationType.BUFFER,
            name="Create Buffer Zones",
            description=f"Create {distance} buffer around features",
            inputs={"features": "input_features.geojson"},
            outputs={"buffers": "buffer_zones.geojson"},
            parameters={"distance": distance},
            depends_on=["step_01"],
            reasoning=f"Creating {distance} buffer zones as specified"
        ))
        
        return steps
    
    def _template_generic_analysis(self, query: ParsedQuery) -> List[WorkflowStep]:
        """Fallback template for unrecognized analysis types"""
        
        return [
            WorkflowStep(
                step_id="step_01",
                operation=OperationType.DATA_LOAD,
                name="Load Required Data",
                description="Load datasets identified in query",
                inputs={},
                outputs={"data": "loaded_data.geojson"},
                parameters={"datasets": query.required_datasets},
                reasoning="Loading all datasets identified from the query for generic analysis"
            ),
            WorkflowStep(
                step_id="step_02",
                operation=OperationType.VISUALIZATION,
                name="Create Basic Visualization",
                description="Generate basic map of loaded data",
                inputs={"data": "loaded_data.geojson"},
                outputs={"map": "basic_map.html"},
                parameters={},
                depends_on=["step_01"],
                reasoning="Creating basic visualization since specific analysis type not recognized"
            )
        ]
    
    def _generate_workflow_name(self, query: ParsedQuery) -> str:
        """Generate a descriptive name for the workflow"""
        
        intent_names = {
            SpatialIntent.FIND_LOCATIONS: "Site Selection",
            SpatialIntent.SUITABILITY_ANALYSIS: "Suitability Analysis", 
            SpatialIntent.MAP_ANALYSIS: "Spatial Mapping",
            SpatialIntent.RISK_ASSESSMENT: "Risk Assessment",
            SpatialIntent.PROXIMITY_ANALYSIS: "Proximity Analysis",
            SpatialIntent.BUFFER_ANALYSIS: "Buffer Analysis"
        }
        
        base_name = intent_names.get(query.intent, "Spatial Analysis")
        
        if query.location:
            return f"{base_name} - {query.location}"
        else:
            return base_name
    
    def _generate_workflow_description(self, query: ParsedQuery) -> str:
        """Generate a detailed description of the workflow"""
        
        description = f"Automated GIS workflow for: {query.raw_query}\n\n"
        description += f"Analysis Type: {query.intent.value.replace('_', ' ').title()}\n"
        
        if query.location:
            description += f"Study Area: {query.location}\n"
        
        if query.target_features:
            description += f"Target Features: {', '.join(query.target_features)}\n"
        
        if query.constraints:
            description += f"Constraints: {', '.join(query.constraints)}\n"
        
        return description
    
    def _build_dependencies(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps"""
        
        dependencies = {}
        
        for step in steps:
            dependencies[step.step_id] = step.depends_on.copy()
        
        return dependencies
    
    def _generate_reasoning_chain(self, query: ParsedQuery, steps: List[WorkflowStep]) -> List[str]:
        """Generate chain-of-thought reasoning for the workflow"""
        
        reasoning_chain = [
            f"1. Query Analysis: Identified intent as '{query.intent.value}' with {query.confidence_score:.1%} confidence",
            f"2. Location Context: {'Analysis focused on ' + query.location if query.location else 'No specific location constraint'}",
            f"3. Data Requirements: Identified {len(query.required_datasets)} required datasets",
            f"4. Workflow Planning: Designed {len(steps)} step process to address the requirements"
        ]
        
        for i, step in enumerate(steps, 5):
            reasoning_chain.append(f"{i}. {step.name}: {step.reasoning}")
        
        return reasoning_chain
    
    def _calculate_complexity(self, steps: List[WorkflowStep]) -> float:
        """Calculate workflow complexity score"""
        
        complexity_weights = {
            OperationType.DATA_LOAD: 1.0,
            OperationType.BUFFER: 1.5,
            OperationType.INTERSECT: 2.0,
            OperationType.SUITABILITY: 3.0,
            OperationType.FLOOD_MODEL: 4.0,
            OperationType.VISUALIZATION: 1.0
        }
        
        total_complexity = sum(complexity_weights.get(step.operation, 2.0) for step in steps)
        return min(total_complexity / 10.0, 1.0)  # Normalize to 0-1
    
    def _estimate_runtime(self, steps: List[WorkflowStep]) -> int:
        """Estimate workflow runtime in seconds"""
        
        runtime_estimates = {
            OperationType.DATA_LOAD: 30,
            OperationType.BUFFER: 15,
            OperationType.INTERSECT: 45,
            OperationType.SUITABILITY: 120,
            OperationType.FLOOD_MODEL: 300,
            OperationType.VISUALIZATION: 20
        }
        
        return sum(runtime_estimates.get(step.operation, 60) for step in steps)
    
    def _identify_assumptions(self, query: ParsedQuery) -> List[str]:
        """Identify assumptions made in workflow generation"""
        
        assumptions = []
        
        if not query.location:
            assumptions.append("Analysis will use default extent or user-provided boundaries")
        
        if query.buffer_distance:
            assumptions.append(f"Buffer distance of {query.buffer_distance} is appropriate for the analysis scale")
        
        if 'suitability' in query.intent.value:
            assumptions.append("Equal weighting assumed for criteria unless specified otherwise")
        
        return assumptions
    
    def _identify_limitations(self, query: ParsedQuery) -> List[str]:
        """Identify limitations of the generated workflow"""
        
        limitations = []
        
        limitations.append("Results depend on quality and currency of input datasets")
        limitations.append("Analysis assumes uniform data quality across study area")
        
        if query.confidence_score < 0.7:
            limitations.append("Query interpretation confidence is below 70% - results may need manual review")
        
        return limitations
    
    def to_json(self, workflow: GISWorkflow) -> str:
        """Convert workflow to JSON representation"""
        
        # Convert dataclasses to dictionaries
        workflow_dict = asdict(workflow)
        
        # Convert enum values to strings
        for step in workflow_dict['steps']:
            step['operation'] = step['operation']
        
        return json.dumps(workflow_dict, indent=2, default=str)
    
    def to_yaml(self, workflow: GISWorkflow) -> str:
        """Convert workflow to YAML representation"""
        
        workflow_dict = asdict(workflow)
        
        # Convert enum values to strings
        for step in workflow_dict['steps']:
            step['operation'] = step['operation']
        
        return yaml.dump(workflow_dict, default_flow_style=False, sort_keys=False)


# Example usage
if __name__ == "__main__":
    from .query_parser import QueryParser
    
    # Test workflow generation
    parser = QueryParser()
    generator = WorkflowGenerator()
    
    test_query = "Find best places to build schools in flood-free zones near highways in Kerala"
    
    print(f"Query: {test_query}")
    print("="*60)
    
    # Parse query
    parsed = parser.parse(test_query)
    print(f"Parsed Intent: {parsed.intent}")
    print(f"Confidence: {parsed.confidence_score:.1%}")
    
    # Generate workflow
    workflow = generator.generate_workflow(parsed)
    print(f"\nGenerated Workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")
    print(f"Estimated Runtime: {workflow.estimated_runtime} seconds")
    
    # Output JSON
    print("\nWorkflow JSON:")
    print(generator.to_json(workflow))
