import sys
import os

# allow python to find project folders
sys.path.append(os.getcwd())

from mangum import Mangum
from MILESTONE_2.phishgaurd.app.main import app

handler = Mangum(app)