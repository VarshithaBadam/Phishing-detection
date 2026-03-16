from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

try:
    response = client.post("/auth/token", data={"username": "test@test.com", "password": "testpassword"})
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
