"""
Setup Script for Member 1 - LLM & Reasoning Engine
=================================================

This script sets up your development environment for the LLM and Reasoning Engine
components. Run this first to install dependencies and configure your workspace.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        return False


def setup_python_environment():
    """Set up Python environment and install dependencies"""
    print("🐍 Setting up Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install core dependencies
    dependencies = [
        "groq>=0.4.1",
        "langchain>=0.1.0", 
        "langchain-community>=0.0.10",
        "langchain-groq>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.2",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "jinja2>=3.1.2",
        "jsonschema>=4.17.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "loguru>=0.7.0",
        "rich>=13.0.0"
    ]
    
    print("\n📦 Installing Python dependencies...")
    for dep in dependencies:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    return True


def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/vector_db",
        "data/uploads", 
        "data/outputs",
        "data/temp",
        "logs",
        "tests",
        "workflows",
        "prompts"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directory structure created")


def setup_environment_file():
    """Set up environment configuration"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("   .env file already exists")
        return True
    
    # Copy from example
    env_example = Path(".env.example")
    if env_example.exists():
        try:
            env_content = env_example.read_text()
            env_file.write_text(env_content)
            print("   Created .env from .env.example")
            print("   ⚠️ Remember to add your Groq API key to .env file!")
        except Exception as e:
            print(f"   ❌ Error creating .env file: {e}")
            return False
    
    return True


def setup_logging():
    """Set up logging configuration"""
    print("\n📝 Setting up logging...")
    
    log_config = '''
import logging
from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add console handler with nice formatting
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler for errors
logger.add(
    "logs/spatial_ai.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Configure standard logging to use loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Replace standard logging with loguru
logging.basicConfig(handlers=[InterceptHandler()], level=0)
'''
    
    try:
        with open("llm_engine/logging_config.py", "w") as f:
            f.write(log_config)
        print("   Created logging configuration")
    except Exception as e:
        print(f"   ❌ Error creating logging config: {e}")


def create_sample_configs():
    """Create sample configuration files"""
    print("\n📋 Creating sample configurations...")
    
    # Sample prompt templates
    sample_prompts = {
        "custom_templates": {
            "site_selection_detailed": {
                "name": "Detailed Site Selection Analysis",
                "prompt_type": "workflow_planning",
                "system_prompt": "You are an expert in site selection analysis...",
                "user_template": "Analyze this site selection request: {user_query}",
                "tags": ["site_selection", "detailed", "custom"]
            }
        }
    }
    
    try:
        import json
        with open("prompts/custom_templates.json", "w") as f:
            json.dump(sample_prompts, f, indent=2)
        print("   Created sample prompt templates")
    except Exception as e:
        print(f"   ❌ Error creating sample prompts: {e}")


def verify_setup():
    """Verify the setup by running basic tests"""
    print("\n🔍 Verifying setup...")
    
    try:
        # Test imports
        from llm_engine.query_parser import QueryParser
        from llm_engine.workflow_generator import WorkflowGenerator  
        from llm_engine.prompt_manager import PromptManager
        print("   ✅ Core imports working")
        
        # Test basic functionality
        parser = QueryParser()
        query = "Find schools in Mumbai"
        parsed = parser.parse(query)
        print(f"   ✅ Query parsing working (confidence: {parsed.confidence_score:.1%})")
        
        generator = WorkflowGenerator()
        workflow = generator.generate_workflow(parsed)
        print(f"   ✅ Workflow generation working ({len(workflow.steps)} steps)")
        
        prompt_manager = PromptManager()
        print(f"   ✅ Prompt manager working ({len(prompt_manager.templates)} templates)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Spatial.AI - Member 1 Setup")
    print("LLM & Reasoning Engine Development Environment")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("llm_engine").exists():
        print("❌ Please run this script from the Spatial.AI root directory")
        sys.exit(1)
    
    success = True
    
    # Run setup steps
    success &= setup_python_environment()
    create_directory_structure()
    success &= setup_environment_file()
    setup_logging()
    create_sample_configs()
    
    if success:
        success &= verify_setup()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next Steps for Member 1:")
        print("1. Add your Groq API key to the .env file:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        print("\n2. Test your setup:")
        print("   python test_llm_engine.py")
        print("\n3. Start developing your components:")
        print("   - Query understanding and parsing")
        print("   - Chain-of-Thought reasoning")
        print("   - RAG system for GIS knowledge")
        print("   - Advanced prompt engineering")
        print("\n4. Coordinate with Member 2 for integration points")
        
    else:
        print("❌ Setup encountered some issues")
        print("Please check the error messages above and resolve them")
        print("You may need to install dependencies manually")
    
    print("\n🔗 Useful Resources:")
    print("- Groq API Documentation: https://console.groq.com/docs")
    print("- LangChain Documentation: https://python.langchain.com/")
    print("- ChromaDB Documentation: https://docs.trychroma.com/")


if __name__ == "__main__":
    main()
