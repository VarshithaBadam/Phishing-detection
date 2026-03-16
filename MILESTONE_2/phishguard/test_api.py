import requests

BASE_URL = "http://localhost:8000"

def test_api():
    print("Testing Signup...")
    res = requests.post(f"{BASE_URL}/auth/signup", data={"email": "test@test.com", "password": "password"})
    print("Signup Response:", res.status_code, res.text)
    
    print("\nTesting Login...")
    res = requests.post(f"{BASE_URL}/auth/token", data={"username": "test@test.com", "password": "password"})
    print("Login Response:", res.status_code)
    token = None
    if res.status_code == 200:
        token = res.json().get("access_token")
        
    print("\nTesting Check URL without token...")
    res = requests.post(f"{BASE_URL}/check", data={"url": "http://secure-login-paypal-update.com"})
    print("Check Response:", res.status_code, res.json())
    
    if token:
        print("\nTesting Check URL with token (generating data for charts)...")
        headers = {"Authorization": f"Bearer {token}"}
        
        urls_to_test = [
            "http://secure-login-paypal-update.com",
            "https://google.com",
            "http://verify-account-admin.net",
            "https://github.com/microsoft"
        ]
        
        for u in urls_to_test:
            res = requests.post(f"{BASE_URL}/check", data={"url": u}, headers=headers)
            print(f"Check {u}:", res.status_code, res.json())
        
        print("\nTesting History...")
        res = requests.get(f"{BASE_URL}/history", headers=headers)
        print("History Response:", res.status_code, res.json())

if __name__ == "__main__":
    test_api()
