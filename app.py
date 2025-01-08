from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
studio_model = joblib.load("models/studio_fan_model.joblib")
dog_model = joblib.load("models/dog_fan_model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract inputs
    fan_type = data.get("fan_type")  # 'studio' or 'dog'
    inputs = np.array(data.get("inputs")).reshape(1, -1)

    # Predict using the appropriate model
    if fan_type == "studio":
        prediction = studio_model.predict(inputs)[0]
    elif fan_type == "dog":
        prediction = dog_model.predict(inputs)[0]
    else:
        return jsonify({"error": "Invalid fan type"}), 400

    return jsonify({"prediction": int(prediction)})
