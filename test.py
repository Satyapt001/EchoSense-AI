import json
import os

import numpy as np
import tensorflow as tf

from utils.audio_processing import preprocess_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Load labels
with open(LABELS_PATH, "r") as f:
    classes = json.load(f)["classes"]


# Load normalization
with open(NORMALIZATION_PATH, "r") as f:
    norm = json.load(f)

mean = norm["mean"]
std = norm["std"]


# CHANGE THIS TO YOUR CAT AUDIO
# CHANGE THIS TO YOUR CAT AUDIO
audio_file = os.path.join(
    BASE_DIR,
    "dragon-studio-cat-meow-401729.mp3"
)

# Preprocess
mel = preprocess_audio(audio_file)



print("Mel shape:", mel.shape)
print("Mel BEFORE normalization:")
print("Mean:", np.mean(mel))
print("Std :", np.std(mel))


# Normalize
mel = (mel - mean) / (std + 1e-8)

print("\nMel AFTER normalization:")
print("Mean:", np.mean(mel))
print("Std :", np.std(mel))


# Add batch dimension
mel = np.expand_dims(mel, axis=0)


# Predict
probabilities = model.predict(
    mel,
    verbose=0
)[0]


# Show all predictions
results = sorted(
    zip(classes, probabilities),
    key=lambda x: x[1],
    reverse=True
)


print("\n==============================")
print("MODEL PREDICTIONS")
print("==============================")

for name, probability in results:

    print(
        f"{name:20s} "
        f"{probability * 100:6.2f}%"
    )


print("\nPredicted:", results[0][0])
print(
    "Confidence:",
    f"{results[0][1] * 100:.2f}%"
)
