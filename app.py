import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Invalid input, expecting {'features': [list_of_values]}"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
