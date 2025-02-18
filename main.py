import requests
import json

url = "http://localhost:11434/api/generate"
data = {
    "model": "deepscaler",
    "prompt": "Say something nice",
}

response = requests.post(url, json=data, stream=True)

