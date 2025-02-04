import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load datasets
with open("dummy_risk.json", "r") as f:
    risk_data = json.load(f)

with open("dummy_forcast.json", "r") as f:
    forecast_data = json.load(f)

# Convert risk data to NumPy arrays for training
X = np.array([[d["x"], d["y"], d["cost"]] for d in risk_data])
y = np.array([d["cost"] for d in risk_data])  # Using "cost" as the prediction target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to disk
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")

# Load model in API
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/risk', methods=['GET'])
def get_risk_data():
    return jsonify(risk_data)

@app.route('/forecast', methods=['GET'])
def get_forecast_data():
    return jsonify(forecast_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not all(k in data for k in ["x", "y", "cost"]):
            return jsonify({"error": "Invalid input, expecting {'x': value, 'y': value, 'cost': value}"}), 400

        # Convert input into array
        features = np.array([[data["x"], data["y"], data["cost"]]])

        # Make prediction
        prediction = model.predict(features).tolist()

        return jsonify({"predicted_cost": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
