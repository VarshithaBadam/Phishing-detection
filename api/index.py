from mangum import Mangum
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MILESTONE_2.phishgaurd.app.main import app

handler = Mangum(app)