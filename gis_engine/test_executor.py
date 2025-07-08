import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from gis_engine.gis_executor import GISExecutor


executor = GISExecutor()
with open("workflows/sample_workflow.json") as f:
    wf = json.load(f)

results = executor.execute_workflow(wf)
executor.validate_outputs(results)
print("✅ GIS Workflow execution completed successfully!")
