import json
import os

import librosa
import numpy as np
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "animal_classifier.keras"
)

LABELS_PATH = os.path.join(
    MODEL_DIR,
    "animal_labels.json"
)

NORMALIZATION_PATH = os.path.join(
    MODEL_DIR,
    "normalization.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

animal_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)


# ============================================================
# LOAD PREPROCESSING METADATA
# ============================================================

with open(NORMALIZATION_PATH, "r") as f:
    metadata = json.load(f)


SAMPLE_RATE = metadata["sample_rate"]
DURATION = metadata["duration"]
NUM_SAMPLES = metadata["num_samples"]

N_MELS = metadata["n_mels"]
N_FFT = metadata["n_fft"]
HOP_LENGTH = metadata["hop_length"]

FMIN = metadata["fmin"]
FMAX = metadata["fmax"]

TRAIN_MEAN = metadata["train_mean"]
TRAIN_STD = metadata["train_std"]


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio(file_path):
    """
    Load audio using the same preprocessing used during training.
    """

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(np.float32)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Peak normalization
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    # Pad or truncate to fixed duration
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio


def audio_to_mel(audio):
    """
    Convert waveform into a log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel.astype(np.float32)


# ============================================================
# PREDICTION
# ============================================================

def predict_animal(file_path):
    """
    Predict animal sound from an audio file.

    Returns:
        dictionary containing:
        - predicted class
        - confidence
        - confidence percentage
        - probabilities for all classes
    """

    # --------------------------------------------------------
    # 1. Load audio
    # --------------------------------------------------------

    audio = load_audio(file_path)

    # --------------------------------------------------------
    # 2. Convert to log-Mel spectrogram
    # --------------------------------------------------------

    mel = audio_to_mel(audio)

    # --------------------------------------------------------
    # 3. Apply training normalization
    # --------------------------------------------------------

    mel = (
        mel - TRAIN_MEAN
    ) / (TRAIN_STD + 1e-8)

    # --------------------------------------------------------
    # 4. Add channel dimension
    #    (128, 157) -> (128, 157, 1)
    # --------------------------------------------------------

    mel = np.expand_dims(
        mel,
        axis=-1
    )

    # --------------------------------------------------------
    # 5. Add batch dimension
    #    (128, 157, 1) -> (1, 128, 157, 1)
    # --------------------------------------------------------

    mel = np.expand_dims(
        mel,
        axis=0
    )

    # --------------------------------------------------------
    # 6. Model prediction
    # --------------------------------------------------------

    probabilities = animal_model.predict(
        mel,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # 7. Highest probability
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # 8. Return result
    # --------------------------------------------------------

    return {
        "class": predicted_class,

        "confidence": confidence,

        "confidence_percent": round(
            confidence * 100,
            2
        ),

        "probabilities": {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
    }