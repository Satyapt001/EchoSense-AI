import json
import os

import numpy as np
import tensorflow as tf

from utils.audio_processing import preprocess_audio

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "speaker_classifier.keras"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "speaker_config.json"
)


# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading Speaker Model...")

speaker_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Speaker model loaded successfully!")


# ==================================================
# LOAD CONFIGURATION
# ==================================================

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)


speaker_classes = config["speakers"]

MEAN = float(config["mel_mean"])
STD = float(config["mel_std"])


print("Speaker configuration loaded successfully!")
print("Speakers:", speaker_classes)


# ==================================================
# PREDICTION
# ==================================================

def predict_speaker(file_path):
    """
    Predict the speaker from an audio file.

    Returns:
        speaker
        confidence
        probabilities
        top_predictions
    """

    # ----------------------------------------------
    # Audio -> Mel Spectrogram
    # ----------------------------------------------

    mel = preprocess_audio(file_path)

    # ----------------------------------------------
    # Apply training normalization
    # ----------------------------------------------

    mel = (
        mel - MEAN
    ) / (STD + 1e-8)

    # ----------------------------------------------
    # Add batch dimension
    # ----------------------------------------------

    mel = np.expand_dims(
        mel,
        axis=0
    )

    # ----------------------------------------------
    # Model prediction
    # ----------------------------------------------

    probabilities = speaker_model.predict(
        mel,
        verbose=0
    )[0]

    # ----------------------------------------------
    # Best prediction
    # ----------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_speaker = speaker_classes[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # ----------------------------------------------
    # All probabilities
    # ----------------------------------------------

    all_probabilities = {
        speaker_classes[i]: round(
            float(probabilities[i]) * 100,
            2
        )
        for i in range(len(speaker_classes))
    }

    # ----------------------------------------------
    # Top predictions
    # ----------------------------------------------

    ranked_indices = np.argsort(
        probabilities
    )[::-1]

    top_predictions = [
        {
            "speaker": speaker_classes[index],
            "confidence": round(
                float(probabilities[index]) * 100,
                2
            )
        }
        for index in ranked_indices[:3]
    ]

    return {
        "speaker": predicted_speaker,
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": all_probabilities,
        "top_predictions": top_predictions
    }