import sys
import os

# add project root to python path
sys.path.insert(0, os.getcwd())

from mangum import Mangum
from MILESTONE_2.phishguard.app.main import app

handler = Mangum(app)