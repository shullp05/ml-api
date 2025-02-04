import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Ensure input format is correct
    if not data or not all(key in data for key in ["dosage", "treatmentType", "timing"]):
        return jsonify({"error": "Invalid input, expecting {'dosage': value, 'treatmentType': value, 'timing': value}"}), 400

    # Extract input features
    features = np.array([data["dosage"], data["treatmentType"], data["timing"]]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features).tolist()
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
