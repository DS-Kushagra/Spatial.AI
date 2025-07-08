"""
Enhanced Demo Scenarios for Spatial.AI
======================================

This module provides comprehensive demo scenarios for testing and showcasing
the complete capabilities of the Spatial.AI system. These scenarios demonstrate
the Chain-of-Thought reasoning and GIS workflow generation capabilities.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedDemoScenarios:
    """Enhanced demo scenarios for comprehensive system testing"""
    
    def __init__(self, spatial_ai_system=None):
        self.spatial_ai = spatial_ai_system
        self.demo_results = {}
        
        # Define comprehensive demo scenarios
        self.scenarios = {
            "school_site_selection": {
                "title": "🏫 School Site Selection in Bangalore",
                "description": "Find optimal locations for new schools avoiding flood zones and ensuring proximity to main roads",
                "query": "Find best locations for schools in Bangalore avoiding flood zones and ensuring 500m from main roads",
                "complexity": "high",
                "expected_operations": ["buffer", "intersect", "overlay", "suitability"],
                "validation_criteria": {
                    "min_workflow_steps": 4,
                    "required_operations": ["buffer", "overlay"],
                    "expected_outputs": ["suitability_map", "candidate_sites"]
                }
            },
            
            "solar_farm_planning": {
                "title": "☀️ Solar Farm Planning in Gujarat", 
                "description": "Identify optimal solar farm locations considering slope, land use, and transmission infrastructure",
                "query": "Identify best areas for solar farms in Gujarat considering slope, land use, and transmission lines",
                "complexity": "high",
                "expected_operations": ["slope_analysis", "raster_calc", "proximity", "multi_criteria"],
                "validation_criteria": {
                    "min_workflow_steps": 5,
                    "required_operations": ["slope_analysis", "multi_criteria"],
                    "expected_outputs": ["suitability_raster", "optimal_zones"]
                }
            },
            
            "flood_risk_mapping": {
                "title": "🌊 Flood Risk Mapping in Kerala",
                "description": "Map flood risk zones using elevation, rainfall, and drainage data", 
                "query": "Map flood risk zones in Kerala using elevation and rainfall data",
                "complexity": "medium",
                "expected_operations": ["raster_calc", "flood_model", "risk_assessment"],
                "validation_criteria": {
                    "min_workflow_steps": 4,
                    "required_operations": ["flood_model"],
                    "expected_outputs": ["flood_zones", "risk_classification"]
                }
            },
            
            "hospital_accessibility": {
                "title": "🏥 Hospital Accessibility Analysis in Mumbai",
                "description": "Analyze hospital accessibility and identify underserved areas",
                "query": "Find suitable hospital locations within 2km of residential areas in Mumbai",
                "complexity": "medium", 
                "expected_operations": ["buffer", "accessibility", "service_area"],
                "validation_criteria": {
                    "min_workflow_steps": 3,
                    "required_operations": ["accessibility"],
                    "expected_outputs": ["service_areas", "gap_analysis"]
                }
            }
        }

    def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all enhanced demo scenarios"""
        logger.info("🚀 Running Enhanced Demo Scenarios...")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "scenarios": {},
            "summary": {}
        }
        
        total_scenarios = len(self.scenarios)
        successful_scenarios = 0
        
        for scenario_id, scenario in self.scenarios.items():
            logger.info(f"Running scenario: {scenario['title']}")
            
            try:
                scenario_result = self.run_scenario(scenario_id, scenario)
                results["scenarios"][scenario_id] = scenario_result
                
                if scenario_result["status"] == "success":
                    successful_scenarios += 1
                    
            except Exception as e:
                logger.error(f"Scenario {scenario_id} failed: {e}")
                results["scenarios"][scenario_id] = {
                    "status": "error",
                    "error": str(e),
                    "scenario": scenario
                }
        
        # Generate summary
        results["summary"] = {
            "total_scenarios": total_scenarios,
            "successful_scenarios": successful_scenarios,
            "success_rate": successful_scenarios / total_scenarios,
            "completion_time": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Completed {successful_scenarios}/{total_scenarios} scenarios")
        return results

    def run_scenario(self, scenario_id: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single demo scenario"""
        start_time = time.time()
        
        try:
            # Process the query if spatial_ai system is available
            if self.spatial_ai:
                result = self.spatial_ai.process_query(
                    query=scenario["query"],
                    context=f"Demo scenario: {scenario['title']}"
                )
            else:
                # Mock result for testing without full system
                result = self._create_mock_result(scenario)
            
            processing_time = time.time() - start_time
            
            # Validate the result
            validation = self._validate_scenario_result(scenario, result)
            
            return {
                "scenario_id": scenario_id,
                "scenario": scenario,
                "status": "success",
                "processing_time": processing_time,
                "result": result,
                "validation": validation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "scenario_id": scenario_id,
                "scenario": scenario, 
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

    def _validate_scenario_result(self, scenario: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario results against expected criteria"""
        validation = {
            "passed": True,
            "checks": {},
            "score": 0,
            "max_score": 0
        }
        
        criteria = scenario.get("validation_criteria", {})
        
        # Check workflow steps count
        if "min_workflow_steps" in criteria:
            workflow = result.get("workflow", {})
            steps = workflow.get("steps", []) if workflow else []
            actual_steps = len(steps)
            min_steps = criteria["min_workflow_steps"]
            
            check_passed = actual_steps >= min_steps
            validation["checks"]["workflow_steps"] = {
                "passed": check_passed,
                "expected": f">= {min_steps}",
                "actual": actual_steps
            }
            validation["max_score"] += 1
            if check_passed:
                validation["score"] += 1
        
        # Check required operations
        if "required_operations" in criteria:
            workflow = result.get("workflow", {})
            steps = workflow.get("steps", []) if workflow else []
            operations = [step.get("operation", "") for step in steps]
            required_ops = criteria["required_operations"]
            
            missing_ops = [op for op in required_ops if op not in operations]
            check_passed = len(missing_ops) == 0
            
            validation["checks"]["required_operations"] = {
                "passed": check_passed,
                "expected": required_ops,
                "actual": operations,
                "missing": missing_ops
            }
            validation["max_score"] += 1
            if check_passed:
                validation["score"] += 1
        
        # Update overall pass status
        validation["passed"] = all(check["passed"] for check in validation["checks"].values())
        validation["success_rate"] = validation["score"] / max(validation["max_score"], 1)
        
        return validation

    def _create_mock_result(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock result for testing without full system"""
        return {
            "query": scenario["query"],
            "status": "success",
            "parsed_query": {
                "intent": "suitability_analysis",
                "location": "bangalore" if "bangalore" in scenario["query"].lower() else "unknown",
                "confidence_score": 0.85
            },
            "workflow": {
                "workflow_id": f"mock_{scenario.get('title', 'demo')}",
                "name": scenario.get("title", "Mock Demo"),
                "steps": [
                    {
                        "name": "load_data",
                        "operation": "load",
                        "description": "Load required geospatial data"
                    },
                    {
                        "name": "analyze",
                        "operation": scenario.get("expected_operations", ["analysis"])[0],
                        "description": "Perform spatial analysis"
                    }
                ]
            },
            "reasoning": {
                "confidence_score": 0.85,
                "reasoning_steps": [
                    "Parse query to understand requirements",
                    "Identify required data sources",
                    "Plan analysis workflow",
                    "Generate step-by-step instructions"
                ]
            },
            "chain_of_thought": [
                {
                    "step": 1,
                    "title": "Query Understanding",
                    "description": f"Understanding the requirements for: {scenario['title']}",
                    "status": "completed"
                },
                {
                    "step": 2,
                    "title": "Workflow Planning", 
                    "description": "Planning the spatial analysis workflow",
                    "status": "completed"
                }
            ]
        }

    def export_demo_results(self, results: Dict[str, Any], output_file: str = "demo_results.json"):
        """Export demo results to file"""
        output_path = Path("data") / "outputs" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Demo results exported to: {output_path}")
        return output_path

# Convenience function for easy import
def run_enhanced_demo(spatial_ai_system=None):
    """Run enhanced demo scenarios"""
    demo = EnhancedDemoScenarios(spatial_ai_system)
    return demo.run_all_scenarios()
