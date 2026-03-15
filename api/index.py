from mangum import Mangum
from MILESTONE_2.phishgaurd.app.main import app

handler = Mangum(app)