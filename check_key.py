import requests

API_KEY = ""

url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"

response = requests.get(url)

print(response.status_code)
print(response.text)
