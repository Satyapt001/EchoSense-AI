import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils.animal_prediction import predict_animal
from utils.speaker_prediction import predict_speaker


app = Flask(__name__)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------------------------
# Allowed audio formats
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "ogg",
    "flac",
    "m4a",
    "webm",
}


def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Audio prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether audio was uploaded
    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file uploaded."
        }), 400

    file = request.files["audio"]

    # Check filename
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported audio format."
        }), 400

    # Get prediction mode
    mode = request.form.get("mode", "animal").lower().strip()

    if mode not in {"animal", "speaker"}:
        return jsonify({
            "success": False,
            "error": "Invalid prediction mode."
        }), 400

    # Secure uploaded filename
    filename = secure_filename(file.filename)

    # Prevent filename collision
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    try:

        # --------------------------------------------------
        # Animal prediction
        # --------------------------------------------------

        if mode == "animal":

            result = predict_animal(file_path)

            return jsonify({
                "success": True,
                "mode": "animal",
                "prediction": result["class"],
                "confidence": result["confidence_percent"],
                "probabilities": result["probabilities"]
            })


        # --------------------------------------------------
        # Speaker prediction
        # --------------------------------------------------

        if mode == "speaker":

            result = predict_speaker(file_path)

            return jsonify({
                "success": True,
                "mode": "speaker",
                "prediction": result["speaker"],
                "speaker": result["speaker"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "top_predictions": result.get(
                    "top_predictions",
                    []
                )
            })


    except Exception as error:

        print(f"Prediction error: {error}")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:

        # Delete temporary uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("Audio Classification Application")
    print("=" * 50)

    print(f"Project directory : {BASE_DIR}")
    print(f"Model directory   : {MODEL_DIR}")
    print(f"Upload directory  : {UPLOAD_FOLDER}")

    print("=" * 50)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )