import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Initialize Flask application
app = Flask(__name__)

# Load the trained and serialized machine learning pipeline
model = joblib.load("superkart_model.joblib")


@app.get("/")
def health_check():
    """Health check endpoint to verify API is active"""
    return jsonify({"status": "healthy", "service": "SuperKart Sales API"})


@app.post("/v1/predict")
def predict_online():
    """Endpoint for single online predictions (JSON payload)"""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Missing or invalid JSON payload"}), 400

    df = pd.DataFrame(data if isinstance(data, list) else [data])

    try:
        prediction = model.predict(df)[0]
        return jsonify({"predicted_sales": float(prediction)}), 200
    except Exception as e:
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500


@app.post("/v1/predictbatch")
def predict_batch():
    """Endpoint for batch predictions (CSV file upload)"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided under key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        predictions = model.predict(df)
        result_dict = {
            f"record_{i}": float(pred) for i, pred in enumerate(predictions)
        }
        return jsonify({"batch_predictions": result_dict}), 200
    except Exception as e:
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)