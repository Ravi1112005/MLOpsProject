from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "movie_genre_model.joblib"

try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    model_error = None if model is not None else "Model not found. Run train_model.py first."
except Exception as exc:
    model = None
    model_error = str(exc)


# Route 1: Render the UI
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        model_ready=model is not None,
        model_error=model_error
    )


# Route 2: Predict movie genre
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": "error",
            "message": model_error or "Model unavailable"
        }), 503

    data = request.get_json(silent=True) or request.form
    plot = str(data.get("plot", "")).strip()

    if not plot:
        return jsonify({
            "status": "error",
            "message": "Please provide a non-empty plot."
        }), 400

    try:
        genre = str(model.predict([plot])[0])

        return jsonify({
            "status": "success",
            "predicted_genre": genre
        }), 200

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "status": "error",
            "message": "Prediction failed."
        }), 500


# Route 3: Health check
@app.route("/health", methods=["GET"])
def health():
    ready = model is not None

    return jsonify({
        "status": "healthy" if ready else "degraded",
        "service": "movie-genre-prediction",
        "model_loaded": ready,
        "model_error": model_error
    }), 200 if ready else 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)