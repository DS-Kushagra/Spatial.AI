"""
Spatial.AI Production Startup Script
====================================

This script provides easy commands to start the production system in different modes:
- API Server mode (for web/API access)
- CLI mode (for command-line usage)
- Interactive mode (for testing)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def start_web_interface():
    """Start the web interface dashboard"""
    print("🌐 Starting Spatial.AI Web Interface...")
    
    try:
        web_server_path = os.path.join(os.path.dirname(__file__), "web_interface", "server.py")
        subprocess.run([sys.executable, web_server_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Web interface not found. Please ensure web_interface/server.py exists.")
        sys.exit(1)


def start_api_server():
    """Start the production API server"""
    print("🚀 Starting Spatial.AI Production API Server...")
    
    host = os.getenv("API_HOST", "localhost")
    port = os.getenv("API_PORT", "8000")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api_server:app",
            "--host", host,
            "--port", port,
            "--reload" if os.getenv("DEBUG", "false").lower() == "true" else "--no-reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)


def start_cli_mode():
    """Start interactive CLI mode"""
    print("💻 Starting Spatial.AI CLI Mode...")
    
    from spatial_ai import SpatialAISystem
    
    try:
        system = SpatialAISystem()
        print("✅ System initialized successfully!")
        print("\nEnter spatial queries (type 'quit' to exit):")
        
        while True:
            try:
                query = input("\n🗺️  Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print("🤔 Processing...")
                result = system.process_query(query)
                
                if result["status"] == "success":
                    print(f"✅ Status: {result['status']}")
                    print(f"🎯 Intent: {result['parsed_query'].intent}")
                    print(f"📊 Confidence: {result['parsed_query'].confidence_score:.1%}")
                    print(f"⚙️  Workflow Steps: {len(result['workflow'].steps)}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)


def run_test_query():
    """Run a test query to verify system"""
    print("🧪 Running system test...")
    
    from spatial_ai import SpatialAISystem
    
    try:
        system = SpatialAISystem()
        
        test_query = "Find best places to build schools in flood-free zones near highways in Kerala"
        print(f"Test Query: {test_query}")
        
        result = system.process_query(test_query)
        
        if result["status"] == "success":
            print("✅ System test PASSED")
            print(f"   - Query parsed successfully")
            print(f"   - Reasoning completed")
            print(f"   - Workflow generated with {len(result['workflow'].steps)} steps")
        else:
            print("❌ System test FAILED")
            print(f"   Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ System test FAILED: {e}")
        sys.exit(1)


def check_system_status():
    """Check system health and configuration"""
    print("🔍 Checking system status...")
    
    # Check environment variables
    required_vars = ["GROQ_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Please check your .env file")
        return False
    
    print("✅ Environment variables configured")
    
    # Check system initialization
    try:
        from spatial_ai import SpatialAISystem
        system = SpatialAISystem()
        status = system.get_system_status()
        
        if status["ready"]:
            print("✅ System health check PASSED")
            print("   All components operational")
        else:
            print("⚠️  System health check shows issues:")
            for component, status in status["components"].items():
                icon = "✅" if status == "operational" else "❌"
                print(f"   {icon} {component}: {status}")
        
        return status["ready"]
        
    except Exception as e:
        print(f"❌ System health check FAILED: {e}")
        return False


def run_enhanced_demo():
    """Run the enhanced production demo scenarios"""
    print("🎯 Starting Enhanced Demo Scenarios...")
    print("This will run all 4 Problem Statement scenarios with comprehensive benchmarking")
    
    try:
        from test_enhanced_demo import main as run_demo_test
        result = run_demo_test()
        
        if result:
            print("✅ Enhanced demo completed successfully!")
            return True
        else:
            print("❌ Enhanced demo failed. Check logs for details.")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_basic_demo():
    """Run the basic demo scenarios"""
    print("🎯 Starting Basic Demo Scenarios...")
    
    try:
        from demo import main as run_basic_demo
        run_basic_demo()
        return True
        
    except Exception as e:
        print(f"❌ Basic demo failed: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Spatial.AI Production System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py api          # Start API server
  python run.py web          # Start web interface
  python run.py cli          # Start CLI mode
  python run.py test         # Run system test
  python run.py status       # Check system status
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["api", "web", "cli", "test", "status", "demo", "enhanced-demo"],
        help="Mode to run the system in"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌍 Spatial.AI - GIS Workflow Generation System")
    print("=" * 60)
    
    if args.mode == "api":
        start_api_server()
    elif args.mode == "web":
        start_web_interface()
    elif args.mode == "cli":
        start_cli_mode()
    elif args.mode == "test":
        run_test_query()
    elif args.mode == "status":
        check_system_status()
    elif args.mode == "demo":
        run_basic_demo()
    elif args.mode == "enhanced-demo":
        run_enhanced_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
