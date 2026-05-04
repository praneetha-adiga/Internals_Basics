import requests
import random
import time

url = "http://127.0.0.1:8080/predict"

# normal data
normal = {
    "prompt_token_count": 400,
    "system_prompt_length": 600,
    "temperature": 0.8,
    "is_few_shot": 1
}

# drift data (shifted distribution)
drift = {
    "prompt_token_count": 1200,
    "system_prompt_length": 2500,
    "temperature": 1.2,
    "is_few_shot": 0
}

# 30 normal + 20 drift = 50 requests
data = [normal]*30 + [drift]*20

for d in data:
    requests.post(url, json=d)
    time.sleep(0.1)

print("Traffic simulation completed")