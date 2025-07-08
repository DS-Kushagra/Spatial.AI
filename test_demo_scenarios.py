"""
Test Demo Scenarios - Testing Framework for Spatial.AI
=====================================================

This module provides comprehensive testing capabilities for the Spatial.AI system,
including automated validation, performance testing, and quality assessment.
"""

import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

class DemoTester:
    """Comprehensive testing framework for Spatial.AI demo scenarios"""
    
    def __init__(self, spatial_ai_system=None):
        self.spatial_ai = spatial_ai_system
        self.test_results = {}
        
        # Define test scenarios with expected outcomes
        self.test_cases = {
            "basic_query_parsing": {
                "title": "Basic Query Parsing",
                "queries": [
                    "Find schools in Bangalore",
                    "Map flood zones in Kerala", 
                    "Analyze solar potential in Gujarat",
                    "Locate hospitals near residential areas"
                ],
                "expected_intents": [
                    "find_locations",
                    "map_analysis", 
                    "suitability_analysis",
                    "proximity_analysis"
                ]
            },
            
            "complex_spatial_queries": {
                "title": "Complex Spatial Queries",
                "queries": [
                    "Find best places to build schools in flood-free zones near highways in Kerala",
                    "Identify optimal solar farm locations in Gujarat considering slope and land use", 
                    "Map flood risk zones using elevation and rainfall data in Mumbai",
                    "Analyze hospital accessibility and identify underserved areas in Bangalore"
                ],
                "min_workflow_steps": [4, 5, 4, 3],
                "required_operations": [
                    ["buffer", "overlay"],
                    ["slope_analysis", "multi_criteria"],
                    ["flood_model"],
                    ["accessibility"]
                ]
            },
            
            "edge_cases": {
                "title": "Edge Cases and Error Handling",
                "queries": [
                    "",  # Empty query
                    "xyz123 invalid query with no spatial context",  # Invalid query
                    "Find schools in NonExistentCity",  # Unknown location
                    "Analyze temporal patterns in spatial data"  # Unsupported operation
                ],
                "expected_status": ["error", "error", "warning", "partial"]
            }
        }

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        logger.info("🧪 Starting Comprehensive Test Suite...")
        
        test_results = {
            "start_time": datetime.now().isoformat(),
            "test_cases": {},
            "performance_metrics": {},
            "summary": {}
        }
        
        total_tests = 0
        passed_tests = 0
        
        # Run each test case
        for test_case_id, test_case in self.test_cases.items():
            logger.info(f"Running test case: {test_case['title']}")
            
            try:
                case_result = self.run_test_case(test_case_id, test_case)
                test_results["test_cases"][test_case_id] = case_result
                
                # Count results
                case_total = case_result.get("total_tests", 0)
                case_passed = case_result.get("passed_tests", 0)
                
                total_tests += case_total
                passed_tests += case_passed
                
            except Exception as e:
                logger.error(f"Test case {test_case_id} failed: {e}")
                test_results["test_cases"][test_case_id] = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Performance testing
        if self.spatial_ai:
            logger.info("Running performance tests...")
            test_results["performance_metrics"] = self.run_performance_tests()
        
        # Generate summary
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / max(total_tests, 1),
            "completion_time": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Test Suite Complete: {passed_tests}/{total_tests} tests passed")
        return test_results

    def run_test_case(self, case_id: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test case"""
        results = {
            "test_case_id": case_id,
            "title": test_case["title"],
            "tests": [],
            "total_tests": 0,
            "passed_tests": 0,
            "start_time": time.time()
        }
        
        queries = test_case.get("queries", [])
        
        for i, query in enumerate(queries):
            test_result = self.run_single_test(query, test_case, i)
            results["tests"].append(test_result)
            results["total_tests"] += 1
            
            if test_result.get("passed", False):
                results["passed_tests"] += 1
        
        results["completion_time"] = time.time() - results["start_time"]
        results["success_rate"] = results["passed_tests"] / max(results["total_tests"], 1)
        
        return results

    def run_single_test(self, query: str, test_case: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """Run a single test query"""
        start_time = time.time()
        
        test_result = {
            "query": query,
            "test_index": test_index,
            "start_time": start_time,
            "status": "unknown",
            "passed": False,
            "checks": {},
            "error": None
        }
        
        try:
            # Process query if system available
            if self.spatial_ai:
                result = self.spatial_ai.process_query(query)
                test_result["system_result"] = result
            else:
                # Mock result for testing
                result = self._create_mock_test_result(query, test_case)
                test_result["system_result"] = result
            
            # Validate result based on test case expectations
            validation = self._validate_test_result(result, test_case, test_index)
            test_result.update(validation)
            
            test_result["processing_time"] = time.time() - start_time
            test_result["status"] = "completed"
            
        except Exception as e:
            test_result["error"] = str(e)
            test_result["status"] = "error"
            test_result["processing_time"] = time.time() - start_time
            logger.error(f"Test failed for query '{query}': {e}")
        
        return test_result

    def _validate_test_result(self, result: Dict[str, Any], test_case: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """Validate test result against expected criteria"""
        validation = {
            "passed": True,
            "checks": {},
            "validation_score": 0,
            "max_validation_score": 0
        }
        
        # Check status
        expected_status = test_case.get("expected_status", [])
        if expected_status and test_index < len(expected_status):
            expected = expected_status[test_index]
            actual = result.get("status", "unknown")
            
            check_passed = actual == expected
            validation["checks"]["status"] = {
                "passed": check_passed,
                "expected": expected,
                "actual": actual
            }
            validation["max_validation_score"] += 1
            if check_passed:
                validation["validation_score"] += 1
        
        # Check workflow steps if specified
        min_steps = test_case.get("min_workflow_steps", [])
        if min_steps and test_index < len(min_steps):
            workflow = result.get("workflow", {})
            steps = workflow.get("steps", []) if workflow else []
            actual_steps = len(steps)
            expected_min = min_steps[test_index]
            
            check_passed = actual_steps >= expected_min
            validation["checks"]["workflow_steps"] = {
                "passed": check_passed,
                "expected": f">= {expected_min}",
                "actual": actual_steps
            }
            validation["max_validation_score"] += 1
            if check_passed:
                validation["validation_score"] += 1
        
        # Check required operations
        required_ops = test_case.get("required_operations", [])
        if required_ops and test_index < len(required_ops):
            workflow = result.get("workflow", {})
            steps = workflow.get("steps", []) if workflow else []
            operations = [step.get("operation", "") for step in steps]
            expected_ops = required_ops[test_index]
            
            missing_ops = [op for op in expected_ops if op not in operations]
            check_passed = len(missing_ops) == 0
            
            validation["checks"]["required_operations"] = {
                "passed": check_passed,
                "expected": expected_ops,
                "actual": operations,
                "missing": missing_ops
            }
            validation["max_validation_score"] += 1
            if check_passed:
                validation["validation_score"] += 1
        
        # Overall validation
        if validation["max_validation_score"] > 0:
            validation["passed"] = all(check["passed"] for check in validation["checks"].values())
        else:
            # If no specific checks, consider success if no error
            validation["passed"] = result.get("status") not in ["error", "failed"]
        
        return validation

    def _create_mock_test_result(self, query: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock result for testing without full system"""
        if not query.strip():
            return {"status": "error", "error": "Empty query"}
        
        if "invalid" in query.lower() or "xyz123" in query:
            return {"status": "error", "error": "Invalid query format"}
        
        return {
            "query": query,
            "status": "success",
            "parsed_query": {
                "intent": "test_intent",
                "confidence_score": 0.8
            },
            "workflow": {
                "workflow_id": "test_workflow",
                "steps": [
                    {"operation": "load", "description": "Load data"},
                    {"operation": "analyze", "description": "Analyze data"}
                ]
            },
            "reasoning": {
                "confidence_score": 0.8
            }
        }

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance testing"""
        logger.info("Running performance tests...")
        
        performance_queries = [
            "Find schools in Bangalore",
            "Find best places to build schools in flood-free zones near highways in Kerala",
            "Map flood risk zones using elevation and rainfall data in Mumbai"
        ]
        
        performance_results = {
            "query_times": [],
            "memory_usage": [],
            "success_rate": 0,
            "average_time": 0,
            "max_time": 0,
            "min_time": float('inf')
        }
        
        successful_queries = 0
        
        for query in performance_queries:
            try:
                start_time = time.time()
                result = self.spatial_ai.process_query(query)
                processing_time = time.time() - start_time
                
                performance_results["query_times"].append(processing_time)
                
                if result.get("status") == "success":
                    successful_queries += 1
                
                # Update timing stats
                performance_results["max_time"] = max(performance_results["max_time"], processing_time)
                performance_results["min_time"] = min(performance_results["min_time"], processing_time)
                
            except Exception as e:
                logger.error(f"Performance test failed for query '{query}': {e}")
        
        # Calculate final metrics
        if performance_results["query_times"]:
            performance_results["average_time"] = statistics.mean(performance_results["query_times"])
            performance_results["median_time"] = statistics.median(performance_results["query_times"])
        
        performance_results["success_rate"] = successful_queries / len(performance_queries)
        performance_results["total_queries"] = len(performance_queries)
        performance_results["successful_queries"] = successful_queries
        
        return performance_results

    def export_test_results(self, results: Dict[str, Any], output_file: str = "test_results.json"):
        """Export test results to file"""
        output_path = Path("data") / "outputs" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results exported to: {output_path}")
        return output_path

    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable test report"""
        report = []
        report.append("=" * 60)
        report.append("SPATIAL.AI TEST REPORT")
        report.append("=" * 60)
        
        summary = results.get("summary", {})
        report.append(f"Total Tests: {summary.get('total_tests', 0)}")
        report.append(f"Passed: {summary.get('passed_tests', 0)}")
        report.append(f"Success Rate: {summary.get('success_rate', 0):.2%}")
        report.append("")
        
        # Test case details
        for case_id, case_result in results.get("test_cases", {}).items():
            report.append(f"Test Case: {case_result.get('title', case_id)}")
            report.append(f"  Tests: {case_result.get('passed_tests', 0)}/{case_result.get('total_tests', 0)}")
            report.append(f"  Success Rate: {case_result.get('success_rate', 0):.2%}")
            report.append("")
        
        # Performance metrics
        perf = results.get("performance_metrics", {})
        if perf:
            report.append("Performance Metrics:")
            report.append(f"  Average Processing Time: {perf.get('average_time', 0):.3f}s")
            report.append(f"  Max Processing Time: {perf.get('max_time', 0):.3f}s")
            report.append(f"  Success Rate: {perf.get('success_rate', 0):.2%}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

# Convenience functions
def run_demo_tests(spatial_ai_system=None):
    """Run demo tests"""
    tester = DemoTester(spatial_ai_system)
    return tester.run_comprehensive_tests()

def quick_test(spatial_ai_system, query: str = "Find schools in Bangalore"):
    """Run a quick test with a single query"""
    tester = DemoTester(spatial_ai_system)
    return tester.run_single_test(query, {}, 0)
