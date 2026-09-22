import librosa
import numpy as np
import torch


def extract_hybrid_features(file_path, n_mels=128):
    """
    Extract Mel-spectrogram and statistical acoustic features.
    """

    # Load audio at 16 kHz
    y, sr = librosa.load(file_path, sr=16000)

    # Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Statistical features
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )
    cent_var = np.var(centroid)

    rms = librosa.feature.rms(y=y)
    energy_dyn = np.std(rms)

    stats = np.array(
        [cent_var, energy_dyn],
        dtype=np.float32
    )

    # Normalize Mel features
    mel_db = (
        (mel_db - np.mean(mel_db))
        / np.std(mel_db)
    )

    return (
        torch.FloatTensor(mel_db).unsqueeze(0),
        torch.FloatTensor(stats)
    )