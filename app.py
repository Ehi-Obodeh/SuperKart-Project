import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Initialize Flask application
app = Flask(__name__)

# Load the trained and serialized machine learning pipeline
model = joblib.load("superkart_model.joblib")

@app.post("/v1/predict")
def predict_online():
    """Endpoint for single online predictions (JSON payload)"""
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"predicted_sales": float(prediction)})

@app.post("/v1/predictbatch")
def predict_batch():
    """Endpoint for batch predictions (CSV file upload)"""
    file = request.files["file"]
    df = pd.read_csv(file)
    predictions = model.predict(df)
    result_dict = {i: float(pred) for i, pred in enumerate(predictions)}
    return jsonify(result_dict)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
