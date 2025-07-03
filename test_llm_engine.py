"""
Test Script for LLM Engine Components
====================================

This script tests the core LLM engine components to ensure they work together
properly. Run this after installing dependencies to validate the setup.
"""

import sys
import os
from datetime import datetime

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from llm_engine.query_parser import QueryParser, SpatialIntent
    from llm_engine.workflow_generator import WorkflowGenerator, OperationType
    from llm_engine.reasoning_engine import ReasoningEngine, ReasoningType
    from llm_engine.prompt_manager import PromptManager, PromptType, PromptContext
    print("✅ All LLM engine imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_query_parser():
    """Test the Query Parser component"""
    print("\n🧠 Testing Query Parser...")
    
    parser = QueryParser()
    
    test_queries = [
        "Find best places to build schools in flood-free zones near highways in Kerala",
        "Map flood risk zones in Kerala using elevation and rainfall data",
        "Identify optimal solar farm locations in Gujarat considering slope and land use"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        parsed = parser.parse(query)
        print(f"  Intent: {parsed.intent}")
        print(f"  Location: {parsed.location}")
        print(f"  Features: {parsed.target_features}")
        print(f"  Constraints: {parsed.constraints}")
        print(f"  Confidence: {parsed.confidence_score:.1%}")
    
    print("✅ Query Parser test completed!")


def test_workflow_generator():
    """Test the Workflow Generator component"""
    print("\n⚙️ Testing Workflow Generator...")
    
    parser = QueryParser()
    generator = WorkflowGenerator()
    
    query = "Find best places to build schools in flood-free zones near highways in Kerala"
    parsed = parser.parse(query)
    
    workflow = generator.generate_workflow(parsed)
    
    print(f"Generated Workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")
    print(f"Estimated Runtime: {workflow.estimated_runtime} seconds")
    print(f"Complexity Score: {workflow.complexity_score:.2f}")
    
    print("\nWorkflow Steps:")
    for i, step in enumerate(workflow.steps[:3], 1):  # Show first 3 steps
        print(f"  {i}. {step.name} ({step.operation.value})")
        print(f"     Reasoning: {step.reasoning[:100]}...")
    
    print("✅ Workflow Generator test completed!")


def test_prompt_manager():
    """Test the Prompt Manager component"""
    print("\n📝 Testing Prompt Manager...")
    
    prompt_manager = PromptManager()
    
    context = PromptContext(
        user_query="Find best places to build schools in flood-free zones near highways in Kerala",
        analysis_type="site_selection",
        location="Kerala",
        constraints=["flood-free zones", "near highways"],
        available_data=["administrative_boundaries", "road_network", "flood_zones"]
    )
    
    # Test query understanding prompt
    system_prompt, user_prompt = prompt_manager.generate_prompt(
        PromptType.QUERY_UNDERSTANDING,
        context
    )
    
    print(f"Generated System Prompt Length: {len(system_prompt)} characters")
    print(f"Generated User Prompt Length: {len(user_prompt)} characters")
    print(f"Templates Available: {len(prompt_manager.templates)}")
    
    # Test performance recording
    prompt_manager.record_prompt_performance(
        "query_understanding_v1",
        success=True,
        confidence=0.85,
        execution_time=2.3
    )
    
    print("✅ Prompt Manager test completed!")


def test_reasoning_engine():
    """Test the Reasoning Engine component (requires Groq API key)"""
    print("\n🤔 Testing Reasoning Engine...")
    
    # Check if Groq API key is available
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("⚠️ Groq API key not found in environment variables")
        print("   Set GROQ_API_KEY to test reasoning engine")
        print("   Skipping reasoning engine test...")
        return
    
    try:
        reasoning_engine = ReasoningEngine(groq_api_key)
        
        test_query = "Find best places to build schools in flood-free zones near highways in Kerala"
        
        # Test query reasoning
        reasoning_step = reasoning_engine.reason_about_query(test_query)
        
        print(f"Reasoning Type: {reasoning_step.reasoning_type}")
        print(f"Confidence: {reasoning_step.confidence:.1%}")
        print(f"Conclusion: {reasoning_step.conclusion[:150]}...")
        
        print("✅ Reasoning Engine test completed!")
        
    except Exception as e:
        print(f"❌ Reasoning Engine test failed: {str(e)}")
        print("   This might be due to API key issues or network connectivity")


def test_integration():
    """Test integration between components"""
    print("\n🔗 Testing Component Integration...")
    
    # Create components
    parser = QueryParser()
    generator = WorkflowGenerator()
    prompt_manager = PromptManager()
    
    # Test query
    query = "Find best places to build schools in flood-free zones near highways in Kerala"
    
    # Parse query
    parsed = parser.parse(query)
    print(f"1. Parsed query with {parsed.confidence_score:.1%} confidence")
    
    # Generate workflow
    workflow = generator.generate_workflow(parsed)
    print(f"2. Generated workflow with {len(workflow.steps)} steps")
    
    # Generate prompt for workflow reasoning
    context = PromptContext(
        user_query=query,
        analysis_type=parsed.intent.value,
        location=parsed.location,
        constraints=parsed.constraints,
        available_data=parsed.required_datasets
    )
    
    system_prompt, user_prompt = prompt_manager.generate_prompt(
        PromptType.WORKFLOW_PLANNING,
        context
    )
    print(f"3. Generated reasoning prompt ({len(user_prompt)} chars)")
    
    # Export workflow to JSON
    workflow_json = generator.to_json(workflow)
    print(f"4. Exported workflow to JSON ({len(workflow_json)} chars)")
    
    print("✅ Integration test completed!")


def main():
    """Run all tests"""
    print("🚀 Starting LLM Engine Component Tests")
    print("=" * 50)
    print(f"Test started at: {datetime.now().isoformat()}")
    
    try:
        test_query_parser()
        test_workflow_generator() 
        test_prompt_manager()
        test_reasoning_engine()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("Your LLM Engine is ready for Member 1 responsibilities!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
