import librosa
import numpy as np

# ==============================
# Audio Configuration
# ==============================

SAMPLE_RATE = 16000
DURATION = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


# ==============================
# Load Audio
# ==============================

def load_audio(file_path):
    """
    Load audio as 16 kHz mono audio, trim leading/trailing silence,
    normalize it and make it exactly 5 seconds.
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Trim leading and trailing silence
    if len(audio) > 0:
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=25)
        if len(trimmed_audio) > 0:
            audio = trimmed_audio

    # Normalize audio
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Pad or truncate to 5 seconds
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


# ==============================
# Convert Audio → Mel Spectrogram
# ==============================

def audio_to_mel(audio):
    """
    Convert audio waveform into Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db.astype(np.float32)


# ==============================
# Complete Preprocessing
# ==============================

def preprocess_audio(file_path):
    """
    Complete preprocessing pipeline.
    Returns CNN-ready input.
    """

    audio = load_audio(file_path)

    mel = audio_to_mel(audio)

    # Add channel dimension
    mel = mel[..., np.newaxis]

    return mel