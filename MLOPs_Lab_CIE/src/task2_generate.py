import requests
import json
import os

url = "http://127.0.0.1:8080"

# health check
health = requests.get(url + "/heartbeat").json()

# test input (given in assignment)
payload = {
    "prompt_token_count": 470,
    "system_prompt_length": 682,
    "temperature": 0.8,
    "is_few_shot": 1
}

pred = requests.post(url + "/predict", json=payload).json()

output = {
    "health_endpoint": "/heartbeat",
    "predict_endpoint": "/predict",
    "port": 8080,
    "health_response": health,
    "test_input": payload,
    "prediction": pred["prediction"]
}

os.makedirs("results", exist_ok=True)

with open("results/step2_s4.json", "w") as f:
    json.dump(output, f, indent=4)

print("step2_s4.json created successfully")