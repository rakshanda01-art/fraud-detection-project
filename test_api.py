import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "amount": 1200.50,
    "transaction_type": "online",
    "is_international": 1,
    "channel": "ecom",
    "merchant_risk": "high",
    "is_weekend": 0
}

print("📡 Sending request to:", url)

try:
    response = requests.post(url, json=data, timeout=10)
    print("✅ Status Code:", response.status_code)
    print("✅ Response JSON:", response.json())
except Exception as e:
    print("❌ Error while calling API:", str(e))
