import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mangum import Mangum
from MILESTONE_2.phishgaurd.app.main import app

handler = Mangum(app)