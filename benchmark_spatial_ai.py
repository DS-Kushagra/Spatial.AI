"""
Spatial.AI Benchmarking and Evaluation Suite
Comprehensive performance, accuracy, and quality assessment
"""

import os
import sys
import json
import time
import psutil
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from spatial_ai import SpatialAISystem
from enhanced_demo_scenarios import EnhancedDemoScenarios
from test_demo_scenarios import DemoTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialAIBenchmark:
    """Comprehensive benchmarking system for Spatial.AI"""
    
    def __init__(self):
        self.spatial_ai = SpatialAISystem()
        self.demo_scenarios = EnhancedDemoScenarios()
        self.demo_tester = DemoTester()
        
        self.benchmark_results = {}
        self.performance_metrics = {}
        self.accuracy_metrics = {}
        self.quality_metrics = {}
        
        # Create benchmark output directory
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        logger.info("🚀 Starting Spatial.AI Full Benchmark Suite")
        
        start_time = datetime.now()
        
        try:
            # 1. Performance Benchmarking
            logger.info("⚡ Running performance benchmarks...")
            self.performance_metrics = self.benchmark_performance()
            
            # 2. Accuracy Testing  
            logger.info("🎯 Running accuracy benchmarks...")
            self.accuracy_metrics = self.benchmark_accuracy()
            
            # 3. Quality Assessment
            logger.info("✨ Running quality benchmarks...")
            self.quality_metrics = self.benchmark_quality()
            
            # 4. Scalability Testing
            logger.info("📈 Running scalability benchmarks...")
            scalability_metrics = self.benchmark_scalability()
            
            # 5. Error Handling & Robustness
            logger.info("🛡️ Running robustness benchmarks...")
            robustness_metrics = self.benchmark_robustness()
            
            # 6. Generate comprehensive report
            self.generate_benchmark_report(start_time)
            
            # 7. Create visualizations
            self.create_benchmark_visualizations()
            
            logger.info("✅ Full benchmark suite completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Benchmark suite failed: {e}")
            raise

    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark system performance metrics"""
        logger.info("⚡ Benchmarking performance...")
        
        performance_results = {
            'query_processing': {},
            'workflow_execution': {},
            'memory_usage': {},
            'system_resources': {}
        }
        
        # Test different query complexities
        test_queries = [
            "Find schools in Kerala",  # Simple
            "Identify flood-safe areas near highways in Kerala for building schools",  # Medium
            "Perform multi-criteria analysis for optimal solar farm locations in Gujarat considering slope, land use, transmission access, and environmental constraints"  # Complex
        ]
        
        for i, query in enumerate(test_queries):
            complexity = ['Simple', 'Medium', 'Complex'][i]
            
            # Measure query processing time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = self.spatial_ai.process_query(query)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                performance_results['query_processing'][complexity] = {
                    'processing_time_ms': (end_time - start_time) * 1000,
                    'memory_delta_mb': end_memory - start_memory,
                    'success': True,
                    'tokens_used': result.get('tokens_used', 0),
                    'reasoning_steps': len(result.get('reasoning_chain', []))
                }
                
            except Exception as e:
                performance_results['query_processing'][complexity] = {
                    'processing_time_ms': 0,
                    'memory_delta_mb': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # System resource monitoring
        performance_results['system_resources'] = {
            'cpu_usage_percent': psutil.cpu_percent(interval=1),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / 1024**3,
            'disk_usage_percent': psutil.disk_usage('.').percent
        }
        
        return performance_results

    def benchmark_accuracy(self) -> Dict[str, Any]:
        """Benchmark system accuracy against known ground truth"""
        logger.info("🎯 Benchmarking accuracy...")
        
        accuracy_results = {
            'workflow_generation': {},
            'spatial_analysis': {},
            'query_understanding': {}
        }
        
        # Test workflow generation accuracy
        test_cases = [
            {
                'query': "Buffer schools by 1km in Kerala",
                'expected_operations': ['load', 'buffer', 'export'],
                'expected_params': {'distance': 1000}
            },
            {
                'query': "Find intersection of flood zones and urban areas",
                'expected_operations': ['load', 'intersect', 'export'],
                'expected_params': {}
            },
            {
                'query': "Filter schools within 500m of highways",
                'expected_operations': ['load', 'buffer', 'intersect', 'export'],
                'expected_params': {'distance': 500}
            }
        ]
        
        correct_workflows = 0
        total_workflows = len(test_cases)
        
        for test_case in test_cases:
            try:
                result = self.spatial_ai.process_query(test_case['query'])
                workflow = result.get('workflow', {})
                
                # Check if expected operations are present
                operations = [step.get('operation') for step in workflow.get('steps', [])]
                expected_ops = test_case['expected_operations']
                
                operations_match = all(op in operations for op in expected_ops)
                
                if operations_match:
                    correct_workflows += 1
                    
            except Exception as e:
                logger.warning(f"Workflow generation failed for: {test_case['query']}")
        
        accuracy_results['workflow_generation'] = {
            'correct_workflows': correct_workflows,
            'total_workflows': total_workflows,
            'accuracy_rate': correct_workflows / total_workflows if total_workflows > 0 else 0
        }
        
        # Test spatial analysis accuracy using demo scenarios
        demo_results = self.demo_tester.run_all_scenarios()
        
        spatial_accuracy = 0
        total_scenarios = len(demo_results)
        
        for scenario_name, scenario_data in demo_results.items():
            if scenario_data['success']:
                # Basic accuracy check - if scenario completed without errors
                spatial_accuracy += 1
        
        accuracy_results['spatial_analysis'] = {
            'successful_scenarios': spatial_accuracy,
            'total_scenarios': total_scenarios,
            'accuracy_rate': spatial_accuracy / total_scenarios if total_scenarios > 0 else 0
        }
        
        return accuracy_results

    def benchmark_quality(self) -> Dict[str, Any]:
        """Benchmark output quality metrics"""
        logger.info("✨ Benchmarking quality...")
        
        quality_results = {
            'output_completeness': {},
            'geometric_validity': {},
            'metadata_quality': {},
            'documentation_quality': {}
        }
        
        # Run a sample scenario and analyze output quality
        try:
            metrics, results = self.demo_scenarios.scenario_1_school_site_selection()
            
            if results:
                total_layers = len(results)
                valid_layers = 0
                total_features = 0
                valid_geometries = 0
                
                for layer_name, gdf in results.items():
                    if isinstance(gdf, gpd.GeoDataFrame):
                        valid_layers += 1
                        total_features += len(gdf)
                        
                        if len(gdf) > 0:
                            valid_geoms = gdf.geometry.is_valid.sum()
                            valid_geometries += valid_geoms
                
                quality_results['output_completeness'] = {
                    'valid_layers_ratio': valid_layers / total_layers if total_layers > 0 else 0,
                    'total_features_generated': total_features
                }
                
                quality_results['geometric_validity'] = {
                    'valid_geometries_ratio': valid_geometries / total_features if total_features > 0 else 0,
                    'total_geometries_checked': total_features
                }
                
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            quality_results['output_completeness'] = {'valid_layers_ratio': 0}
            quality_results['geometric_validity'] = {'valid_geometries_ratio': 0}
        
        return quality_results

    def benchmark_scalability(self) -> Dict[str, Any]:
        """Test system scalability with varying data sizes"""
        logger.info("📈 Benchmarking scalability...")
        
        scalability_results = {
            'data_size_performance': {},
            'concurrent_processing': {},
            'memory_scaling': {}
        }
        
        # Test with different data sizes (simulated)
        data_sizes = [100, 500, 1000, 5000]  # Number of features
        
        for size in data_sizes:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Create test dataset with specified size
                from shapely.geometry import Point
                import random
                
                test_data = []
                for i in range(size):
                    test_data.append({
                        'geometry': Point(random.uniform(75, 77), random.uniform(10, 12)),
                        'id': i,
                        'value': random.uniform(0, 100)
                    })
                
                test_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
                
                # Simulate processing (buffer operation)
                projected_crs = "EPSG:32643"
                test_proj = test_gdf.to_crs(projected_crs)
                buffered = test_proj.buffer(1000)  # 1km buffer
                result = buffered.to_crs("EPSG:4326")
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                scalability_results['data_size_performance'][f'{size}_features'] = {
                    'processing_time_ms': (end_time - start_time) * 1000,
                    'memory_delta_mb': end_memory - start_memory,
                    'features_per_second': size / (end_time - start_time),
                    'success': True
                }
                
            except Exception as e:
                scalability_results['data_size_performance'][f'{size}_features'] = {
                    'processing_time_ms': 0,
                    'memory_delta_mb': 0,
                    'features_per_second': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return scalability_results

    def benchmark_robustness(self) -> Dict[str, Any]:
        """Test system robustness and error handling"""
        logger.info("🛡️ Benchmarking robustness...")
        
        robustness_results = {
            'error_handling': {},
            'edge_cases': {},
            'malformed_inputs': {}
        }
        
        # Test error handling with problematic queries
        error_test_queries = [
            "",  # Empty query
            "asdfghjkl qwerty",  # Nonsense query
            "Find schools in XYZ planet",  # Invalid location
            "Buffer by negative distance",  # Invalid parameter
            "SELECT * FROM tables",  # SQL injection attempt
        ]
        
        handled_errors = 0
        total_error_tests = len(error_test_queries)
        
        for query in error_test_queries:
            try:
                result = self.spatial_ai.process_query(query)
                # If it doesn't crash, check if error was handled gracefully
                if 'error' in result or result.get('success') == False:
                    handled_errors += 1
                    
            except Exception as e:
                # Exception was raised - check if it's handled gracefully
                if "error" in str(e).lower() or "invalid" in str(e).lower():
                    handled_errors += 1
        
        robustness_results['error_handling'] = {
            'gracefully_handled_errors': handled_errors,
            'total_error_tests': total_error_tests,
            'error_handling_rate': handled_errors / total_error_tests if total_error_tests > 0 else 0
        }
        
        return robustness_results

    def generate_benchmark_report(self, start_time: datetime):
        """Generate comprehensive benchmark report"""
        logger.info("📊 Generating benchmark report...")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        report = {
            'benchmark_summary': {
                'timestamp': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                    'platform': sys.platform
                }
            },
            'performance_metrics': self.performance_metrics,
            'accuracy_metrics': self.accuracy_metrics,
            'quality_metrics': self.quality_metrics,
            'overall_scores': self.calculate_overall_scores()
        }
        
        # Save detailed report
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(report)
        
        logger.info(f"📄 Benchmark report saved to {report_file}")

    def calculate_overall_scores(self) -> Dict[str, float]:
        """Calculate overall performance scores"""
        scores = {}
        
        # Performance Score (0-100)
        perf_metrics = self.performance_metrics.get('query_processing', {})
        if perf_metrics:
            avg_time = np.mean([m.get('processing_time_ms', 0) for m in perf_metrics.values() if m.get('success', False)])
            performance_score = max(0, 100 - (avg_time / 100))  # Penalty for slow processing
            scores['performance_score'] = min(100, performance_score)
        else:
            scores['performance_score'] = 0
        
        # Accuracy Score (0-100)
        acc_metrics = self.accuracy_metrics
        workflow_acc = acc_metrics.get('workflow_generation', {}).get('accuracy_rate', 0)
        spatial_acc = acc_metrics.get('spatial_analysis', {}).get('accuracy_rate', 0)
        accuracy_score = (workflow_acc + spatial_acc) / 2 * 100
        scores['accuracy_score'] = accuracy_score
        
        # Quality Score (0-100)
        qual_metrics = self.quality_metrics
        completeness = qual_metrics.get('output_completeness', {}).get('valid_layers_ratio', 0)
        validity = qual_metrics.get('geometric_validity', {}).get('valid_geometries_ratio', 0)
        quality_score = (completeness + validity) / 2 * 100
        scores['quality_score'] = quality_score
        
        # Overall Score
        scores['overall_score'] = (scores['performance_score'] + scores['accuracy_score'] + scores['quality_score']) / 3
        
        return scores

    def generate_summary_report(self, full_report: Dict):
        """Generate human-readable summary report"""
        summary_file = self.output_dir / "benchmark_summary.txt"
        
        overall_scores = full_report['overall_scores']
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SPATIAL.AI BENCHMARK RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Benchmark Date: {full_report['benchmark_summary']['timestamp']}\n")
            f.write(f"Total Duration: {full_report['benchmark_summary']['total_duration_seconds']:.1f} seconds\n\n")
            
            f.write("OVERALL SCORES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Performance Score: {overall_scores['performance_score']:.1f}/100\n")
            f.write(f"Accuracy Score: {overall_scores['accuracy_score']:.1f}/100\n")
            f.write(f"Quality Score: {overall_scores['quality_score']:.1f}/100\n")
            f.write(f"Overall Score: {overall_scores['overall_score']:.1f}/100\n\n")
            
            # Performance Details
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            perf_metrics = full_report['performance_metrics'].get('query_processing', {})
            for complexity, metrics in perf_metrics.items():
                if metrics.get('success', False):
                    f.write(f"{complexity} Query: {metrics['processing_time_ms']:.1f}ms\n")
            f.write("\n")
            
            # Accuracy Details
            f.write("ACCURACY METRICS:\n")
            f.write("-" * 20 + "\n")
            acc_metrics = full_report['accuracy_metrics']
            workflow_acc = acc_metrics.get('workflow_generation', {})
            f.write(f"Workflow Generation: {workflow_acc.get('accuracy_rate', 0):.1%}\n")
            spatial_acc = acc_metrics.get('spatial_analysis', {})
            f.write(f"Spatial Analysis: {spatial_acc.get('accuracy_rate', 0):.1%}\n\n")
            
            # Quality Details
            f.write("QUALITY METRICS:\n")
            f.write("-" * 20 + "\n")
            qual_metrics = full_report['quality_metrics']
            completeness = qual_metrics.get('output_completeness', {})
            f.write(f"Output Completeness: {completeness.get('valid_layers_ratio', 0):.1%}\n")
            validity = qual_metrics.get('geometric_validity', {})
            f.write(f"Geometric Validity: {validity.get('valid_geometries_ratio', 0):.1%}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            if overall_scores['overall_score'] >= 80:
                f.write("✅ System is ready for production deployment\n")
            elif overall_scores['overall_score'] >= 60:
                f.write("⚠️ System needs minor improvements before production\n")
            else:
                f.write("❌ System needs significant improvements before production\n")
            
            if overall_scores['performance_score'] < 70:
                f.write("• Optimize query processing performance\n")
            if overall_scores['accuracy_score'] < 70:
                f.write("• Improve workflow generation accuracy\n")
            if overall_scores['quality_score'] < 70:
                f.write("• Enhance output quality validation\n")
        
        logger.info(f"📄 Summary report saved to {summary_file}")

    def create_benchmark_visualizations(self):
        """Create visualization charts for benchmark results"""
        logger.info("📊 Creating benchmark visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Overall Scores Bar Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall scores
        scores = self.calculate_overall_scores()
        score_names = ['Performance', 'Accuracy', 'Quality', 'Overall']
        score_values = [scores['performance_score'], scores['accuracy_score'], 
                       scores['quality_score'], scores['overall_score']]
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
        bars = ax1.bar(score_names, score_values, color=colors)
        ax1.set_title('Overall Benchmark Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score (0-100)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, score_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance by Query Complexity
        perf_metrics = self.performance_metrics.get('query_processing', {})
        if perf_metrics:
            complexities = list(perf_metrics.keys())
            times = [perf_metrics[c].get('processing_time_ms', 0) for c in complexities]
            
            ax2.bar(complexities, times, color='#ff7f0e', alpha=0.7)
            ax2.set_title('Processing Time by Query Complexity', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Processing Time (ms)')
            
            for i, time in enumerate(times):
                ax2.text(i, time + max(times)*0.01, f'{time:.1f}ms',
                        ha='center', va='bottom', fontweight='bold')
        
        # Accuracy Breakdown
        acc_metrics = self.accuracy_metrics
        workflow_acc = acc_metrics.get('workflow_generation', {}).get('accuracy_rate', 0) * 100
        spatial_acc = acc_metrics.get('spatial_analysis', {}).get('accuracy_rate', 0) * 100
        
        accuracy_data = ['Workflow\nGeneration', 'Spatial\nAnalysis']
        accuracy_values = [workflow_acc, spatial_acc]
        
        bars3 = ax3.bar(accuracy_data, accuracy_values, color='#2ca02c', alpha=0.7)
        ax3.set_title('Accuracy Breakdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim(0, 100)
        
        for bar, value in zip(bars3, accuracy_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Quality Metrics
        qual_metrics = self.quality_metrics
        completeness = qual_metrics.get('output_completeness', {}).get('valid_layers_ratio', 0) * 100
        validity = qual_metrics.get('geometric_validity', {}).get('valid_geometries_ratio', 0) * 100
        
        quality_data = ['Output\nCompleteness', 'Geometric\nValidity']
        quality_values = [completeness, validity]
        
        bars4 = ax4.bar(quality_data, quality_values, color='#d62728', alpha=0.7)
        ax4.set_title('Quality Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Quality (%)')
        ax4.set_ylim(0, 100)
        
        for bar, value in zip(bars4, quality_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_file = self.output_dir / "benchmark_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Benchmark charts saved to {chart_file}")

def main():
    """Main benchmark execution"""
    logger.info("🚀 Starting Spatial.AI Comprehensive Benchmark")
    
    benchmark = SpatialAIBenchmark()
    
    try:
        benchmark.run_full_benchmark()
        
        # Print summary to console
        scores = benchmark.calculate_overall_scores()
        
        print("\n" + "="*60)
        print("🎯 SPATIAL.AI BENCHMARK RESULTS")
        print("="*60)
        print(f"Performance Score: {scores['performance_score']:.1f}/100")
        print(f"Accuracy Score: {scores['accuracy_score']:.1f}/100")
        print(f"Quality Score: {scores['quality_score']:.1f}/100")
        print(f"Overall Score: {scores['overall_score']:.1f}/100")
        
        if scores['overall_score'] >= 80:
            print("\n🎉 System is ready for production deployment!")
            return 0
        elif scores['overall_score'] >= 60:
            print("\n⚠️ System needs minor improvements before production.")
            return 1
        else:
            print("\n❌ System needs significant improvements before production.")
            return 2
            
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return 3

if __name__ == "__main__":
    exit(main())
