import urllib.request
import urllib.parse
data = urllib.parse.urlencode({"username": "test@test.com", "password": "test"}).encode()
req = urllib.request.Request("http://localhost:8000/auth/token", data=data)
try:
    with urllib.request.urlopen(req) as response:
        print("STATUS", response.status)
        print("BODY", response.read().decode())
except urllib.error.HTTPError as e:
    print("STATUS", e.code)
    print("BODY", e.read().decode())
