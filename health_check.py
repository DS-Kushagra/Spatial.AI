"""
Quick System Check for Spatial.AI
=================================

This script performs a quick health check of all system components
to ensure everything is working properly.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Check environment variables"""
    print("🔧 Checking Environment Configuration...")
    
    required_vars = ["GROQ_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"  ✅ {var}: Found")
    
    if missing_vars:
        print(f"  ❌ Missing variables: {missing_vars}")
        return False
    
    print("  ✅ Environment configuration complete")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking Dependencies...")
    
    required_packages = [
        ("groq", "Groq API client"),
        ("fastapi", "FastAPI web framework"),
        ("geopandas", "GIS data processing"),
        ("chromadb", "Vector database"),
        ("sentence_transformers", "Embeddings"),
        ("uvicorn", "API server")
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ❌ {package}: {description} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"  ❌ Missing packages: {missing_packages}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    print("  ✅ All dependencies installed")
    return True

def check_llm_engine():
    """Check LLM engine components"""
    print("🧠 Checking LLM Engine...")
    
    try:
        from llm_engine import SpatialAIEngine
        
        # Initialize engine
        engine = SpatialAIEngine()
        print("  ✅ LLM Engine initialized")
        
        # Check system status
        status = engine.get_system_status()
        if status.get("ready"):
            print("  ✅ All engine components ready")
        else:
            print("  ⚠️ Some components may have issues")
            
        return True
        
    except Exception as e:
        print(f"  ❌ LLM Engine error: {e}")
        return False

def check_gis_engine():
    """Check GIS engine components"""
    print("🗺️ Checking GIS Engine...")
    
    try:
        from gis_engine.gis_executor import GISExecutor
        
        # Initialize executor
        executor = GISExecutor()
        print("  ✅ GIS Executor initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GIS Engine error: {e}")
        return False

def check_web_interface():
    """Check web interface files"""
    print("🌐 Checking Web Interface...")
    
    web_files = [
        "web_interface/index.html",
        "web_interface/styles.css",
        "web_interface/app.js"
    ]
    
    all_present = True
    
    for file_path in web_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NOT FOUND")
            all_present = False
    
    return all_present

def quick_api_test():
    """Test API functionality"""
    print("🚀 Testing API Functionality...")
    
    try:
        from spatial_ai import SpatialAISystem
        
        # Initialize system
        system = SpatialAISystem()
        print("  ✅ System initialized")
        
        # Test query processing
        test_query = "Find schools in Bangalore"
        print(f"  🧪 Testing query: {test_query}")
        
        start_time = time.time()
        result = system.process_query(test_query)
        processing_time = time.time() - start_time
        
        if result.get("status") == "success":
            print(f"  ✅ Query processed successfully in {processing_time:.2f}s")
            
            # Check key components
            if "workflow" in result:
                print("  ✅ Workflow generated")
            if "reasoning" in result:
                print("  ✅ Reasoning included")
            if "chain_of_thought" in result:
                print("  ✅ Chain-of-thought reasoning")
                
            return True
        else:
            print(f"  ❌ Query failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False

def main():
    """Run complete system check"""
    print("🔍 Spatial.AI System Health Check")
    print("=" * 50)
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("LLM Engine", check_llm_engine),
        ("GIS Engine", check_gis_engine),
        ("Web Interface", check_web_interface),
        ("API Test", quick_api_test)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
            print()
        except Exception as e:
            print(f"  ❌ {check_name} check failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"System Health: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("✅ System is fully operational!")
        print("\nNext steps:")
        print("1. Start the system: python start.py")
        print("2. Open browser: http://localhost:8000")
        print("3. Try demo queries in the web interface")
    elif passed_checks >= total_checks - 1:
        print("⚠️ System is mostly operational with minor issues")
        print("Check the failed components above")
    else:
        print("❌ System has significant issues that need attention")
        print("Please fix the failed components before proceeding")

if __name__ == "__main__":
    main()
