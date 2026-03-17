import sys
import os

# add project root and MILESTONE_2 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Import the FastAPI app
try:
    from MILESTONE_2.phishguard.app.main import app
except ImportError:
    # Handle direct access if MILESTONE_2 is not a package
    sys.path.insert(0, os.path.join(project_root, "MILESTONE_2", "phishguard"))
    from app.main import app

from mangum import Mangum

handler = Mangum(app)