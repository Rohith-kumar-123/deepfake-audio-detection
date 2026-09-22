import os
import tempfile

import librosa
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.model import DeepfakeHybridModel
from src.preprocess import extract_hybrid_features


# -----------------------------
# 1. PAGE SETUP
# -----------------------------

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("Deepfake Audio Detection")
st.write(
    "Upload an audio file to detect whether it is "
    "real (bonafide) or AI-generated (spoof)."
)


# -----------------------------
# 2. LOAD MODEL
# -----------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = DeepfakeHybridModel().to(device)

MODEL_PATH = "weights/model_refined_e5.pth"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model weights not found: {MODEL_PATH}"
    )
    st.stop()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# -----------------------------
# 3. FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "flac"]
)


# -----------------------------
# 4. PREDICTION
# -----------------------------

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(
            uploaded_file.name
        )[1]
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = tmp_file.name

    try:

        # Extract hybrid features
        spec, stats = extract_hybrid_features(
            temp_path
        )

        # Normalize features
        spec = (
            (spec - spec.mean())
            / (spec.std() + 1e-6)
        )

        stats = (
            (stats - stats.mean())
            / (stats.std() + 1e-6)
        )

        # Add batch dimension
        spec = spec.unsqueeze(0).to(device)
        stats = stats.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():

            output = model(
                spec,
                stats
            )

            probability = torch.sigmoid(
                output
            ).item()

        # Classification
        if probability > 0.5:

            prediction = "DEEPFAKE (SPOOF)"
            confidence = probability * 100

        else:

            prediction = "REAL (BONAFIDE)"
            confidence = (1 - probability) * 100


        # -----------------------------
        # 5. DISPLAY RESULT
        # -----------------------------

        st.subheader(
            "Prediction"
        )

        st.write(
            f"### {prediction}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )


        # -----------------------------
        # 6. MEL-SPECTROGRAM
        # -----------------------------

        y, sr = librosa.load(
            temp_path,
            sr=16000
        )

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel_spec,
            ref=max
        )

        st.subheader(
            "Mel-Spectrogram"
        )

        fig, ax = plt.subplots()

        img = ax.imshow(
            mel_db,
            aspect="auto",
            origin="lower"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Mel Frequency"
        )

        fig.colorbar(
            img,
            ax=ax
        )

        st.pyplot(fig)

        plt.close(fig)

    finally:

        # Remove temporary uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)