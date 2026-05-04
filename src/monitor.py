import json
import pandas as pd

file_path = "logs/predictions.jsonl"

data = []

with open(file_path) as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame([x["input"] for x in data])

# live means
live_token_mean = df["prompt_token_count"].mean()
live_sys_mean = df["system_prompt_length"].mean()

# training reference means (given in question)
train_token_mean = 450.56
train_sys_mean = 926.0

token_shift = abs(live_token_mean - train_token_mean)
sys_shift = abs(live_sys_mean - train_sys_mean)

# drift = token_shift > 272.06 or sys_shift > 322.36

# output = {
#     "total_predictions": len(df),
#     "mean_prediction": 0.0,
#     "drift_detected": drift,
#     "alerts": [
#         {
#             "feature": "prompt_token_count",
#             "train_mean": train_token_mean,
#             "live_mean": live_token_mean,
#             "shift": token_shift,
#             "threshold": 272.06,
#             "status": "ALERT" if token_shift > 272.06 else "OK"
#         },
#         {
#             "feature": "system_prompt_length",
#             "train_mean": train_sys_mean,
#             "live_mean": live_sys_mean,
#             "shift": sys_shift,
#             "threshold": 322.36,
#             "status": "ALERT" if sys_shift > 322.36 else "OK"
#         }
#     ]
# }

drift = bool(token_shift > 272.06 or sys_shift > 322.36)

output = {
    "total_predictions": int(len(df)),
    "mean_prediction": float(0.0),
    "drift_detected": bool(drift),
    "alerts": [
        {
            "feature": "prompt_token_count",
            "train_mean": float(train_token_mean),
            "live_mean": float(live_token_mean),
            "shift": float(token_shift),
            "threshold": float(272.06),
            "status": "ALERT" if token_shift > 272.06 else "OK"
        },
        {
            "feature": "system_prompt_length",
            "train_mean": float(train_sys_mean),
            "live_mean": float(live_sys_mean),
            "shift": float(sys_shift),
            "threshold": float(322.36),
            "status": "ALERT" if sys_shift > 322.36 else "OK"
        }
    ]
}

import os
os.makedirs("results", exist_ok=True)

with open("results/step3_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("Monitoring completed")