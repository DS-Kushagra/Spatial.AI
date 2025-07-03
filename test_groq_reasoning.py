"""
Test Groq API Connection and Reasoning Engine
============================================

This script specifically tests the Groq API connection and reasoning capabilities.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test basic Groq API connection"""
    print("🔑 Testing Groq API Connection...")
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Test basic completion
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Say 'Hello from Spatial.AI!' if you can hear me."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Groq connection failed: {str(e)}")
        return False


def test_reasoning_engine():
    """Test the full reasoning engine"""
    print("\n🧠 Testing Reasoning Engine...")
    
    try:
        from llm_engine.reasoning_engine import ReasoningEngine
        
        # Initialize with environment API key
        reasoning_engine = ReasoningEngine()
        
        # Test query reasoning
        test_query = "Find best places to build schools in flood-free zones near highways in Kerala"
        
        print(f"Query: {test_query}")
        print("Generating reasoning...")
        
        reasoning_step = reasoning_engine.reason_about_query(test_query)
        
        print(f"\n📊 Results:")
        print(f"  Reasoning Type: {reasoning_step.reasoning_type.value}")
        print(f"  Confidence: {reasoning_step.confidence:.1%}")
        print(f"  Question: {reasoning_step.question}")
        print(f"\n🤔 AI Reasoning:")
        print(f"  {reasoning_step.reasoning[:500]}...")
        print(f"\n🎯 Conclusion:")
        print(f"  {reasoning_step.conclusion}")
        
        if reasoning_step.evidence:
            print(f"\n🔍 Evidence:")
            for i, evidence in enumerate(reasoning_step.evidence[:3], 1):
                print(f"  {i}. {evidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning Engine test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_reasoning():
    """Test workflow planning with reasoning"""
    print("\n⚙️ Testing Workflow Reasoning...")
    
    try:
        from llm_engine.query_parser import QueryParser
        from llm_engine.reasoning_engine import ReasoningEngine
        
        parser = QueryParser()
        reasoning_engine = ReasoningEngine()
        
        query = "Find best places to build schools in flood-free zones near highways in Kerala"
        parsed = parser.parse(query)
        
        print(f"Parsed Intent: {parsed.intent.value}")
        print("Generating workflow reasoning...")
        
        workflow_reasoning = reasoning_engine.reason_about_workflow(parsed)
        
        print(f"\n📊 Workflow Reasoning Results:")
        print(f"  Confidence: {workflow_reasoning.confidence:.1%}")
        print(f"\n🤔 AI Workflow Planning:")
        print(f"  {workflow_reasoning.reasoning[:500]}...")
        
        if workflow_reasoning.alternatives:
            print(f"\n🔀 Alternative Approaches:")
            for i, alt in enumerate(workflow_reasoning.alternatives[:2], 1):
                print(f"  {i}. {alt}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow reasoning test failed: {str(e)}")
        return False


def test_complete_reasoning_trace():
    """Test complete reasoning trace generation"""
    print("\n🔗 Testing Complete Reasoning Trace...")
    
    try:
        from llm_engine.query_parser import QueryParser
        from llm_engine.workflow_generator import WorkflowGenerator
        from llm_engine.reasoning_engine import ReasoningEngine
        
        # Initialize components
        parser = QueryParser()
        generator = WorkflowGenerator()
        reasoning_engine = ReasoningEngine()
        
        query = "Find best places to build schools in flood-free zones near highways in Kerala"
        
        # Parse and generate workflow
        parsed = parser.parse(query)
        workflow = generator.generate_workflow(parsed)
        
        print(f"Generating complete reasoning trace...")
        
        # Generate complete reasoning trace
        trace = reasoning_engine.generate_complete_reasoning_trace(query, parsed, workflow)
        
        print(f"\n📊 Complete Reasoning Trace:")
        print(f"  Trace ID: {trace.trace_id}")
        print(f"  Reasoning Steps: {len(trace.reasoning_steps)}")
        print(f"  Final Confidence: {trace.final_confidence:.1%}")
        print(f"  Total Reasoning Time: {trace.total_reasoning_time:.2f}s")
        print(f"  LLM Model: {trace.llm_model}")
        
        print(f"\n🎯 Key Insights:")
        for insight in trace.key_insights[:3]:
            print(f"  • {insight}")
        
        print(f"\n⚠️ Assumptions:")
        for assumption in trace.assumptions[:2]:
            print(f"  • {assumption}")
        
        if trace.uncertainties:
            print(f"\n❓ Uncertainties:")
            for uncertainty in trace.uncertainties:
                print(f"  • {uncertainty}")
        
        print(f"\n💡 Recommendations:")
        for rec in trace.recommendations[:2]:
            print(f"  • {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete reasoning trace test failed: {str(e)}")
        return False


def main():
    """Run all Groq and reasoning tests"""
    print("🚀 Spatial.AI - Groq API & Reasoning Engine Test")
    print("=" * 60)
    
    success = True
    
    # Test basic connection
    success &= test_groq_connection()
    
    if success:
        # Test reasoning components
        success &= test_reasoning_engine()
        success &= test_workflow_reasoning()
        success &= test_complete_reasoning_trace()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 All Groq and Reasoning tests passed!")
        print("\n🧠 Your AI Brain is fully operational!")
        print("✅ Chain-of-Thought reasoning working")
        print("✅ Query understanding with AI enhancement")
        print("✅ Workflow planning with intelligent reasoning")
        print("✅ Complete reasoning traces with explanations")
        print("\n🚀 Ready for Phase 2 development!")
        
    else:
        print("❌ Some tests failed")
        print("Check API key and network connection")


if __name__ == "__main__":
    main()
