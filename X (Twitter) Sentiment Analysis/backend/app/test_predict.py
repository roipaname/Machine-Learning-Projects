import requests

# FastAPI server URL
url = "http://127.0.0.1:8000/predict"

# Example texts to classify
data = {
    "texts": [
        "I love this product! It's amazing.",
        "This is the worst experience ever.",
        "Meh, it was okay, nothing special."
    ],
    "classifier_type": "logistic_regression"  # or whichever model you want
}

# Send POST request
response = requests.post(url, json=data)

# Check response
if response.status_code == 200:
    result = response.json()
    print("Model used:", result["model"])
    for pred in result["predictions"]:
        print(f"Text: {pred['text']}")
        print(f"Label: {pred['label']}, Confidence: {pred['confidence']}")
        print("-" * 40)
else:
    print("Error:", response.status_code, response.text)
