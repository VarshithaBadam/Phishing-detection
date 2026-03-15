import sys
import os

# add project root to python path
sys.path.insert(0, os.getcwd())

try:
    from mangum import Mangum
    from MILESTONE_2.phishgaurd.app.main import app
    
    # Export for Vercel
    handler = Mangum(app)
except Exception as e:
    print(f"Error initializing application: {e}")
    import traceback
    traceback.print_exc()
    raise