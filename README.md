# 🎙️ Robust Deepfake Audio Detection Using Deep Learning

> A hybrid deep learning system for detecting AI-generated audio using
> **Mel-spectrograms, CNN-LSTM modeling, and statistical acoustic features.**

---

## 📌 Overview

Deepfake and synthetic audio generation has become increasingly realistic,
making automated detection an important problem in audio security and
digital media verification.

This project presents a hybrid deep learning approach that combines:

- 🎵 **Mel-spectrogram analysis** for time-frequency representation
- 🧠 **CNN** for spectral feature extraction
- 🔄 **Bidirectional LSTM** for temporal modeling
- 📊 **Statistical acoustic features** for complementary information
- 🔗 **Feature fusion** for final classification
- 🌐 **Streamlit** interface for audio-based prediction

The system classifies an uploaded audio sample as either
**Real (Bonafide)** or **Deepfake (Spoof)**.

---

## 🏗️ System Architecture

![System Architecture](docs/architecture.png)

```text
                     Audio Input
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                  Mel-Spectrogram
                         │
                         ▼
                       CNN
                         │
                         ▼
                Bidirectional LSTM
                         │
                         ├──────────────┐
                         │              │
                         │     Statistical Features
                         │              │
                         └──────┬───────┘
                                ▼
                         Feature Fusion
                                │
                                ▼
                      Fully Connected
                                │
                                ▼
                       Real / Deepfake


✨ Key Features
🎧 Deepfake audio detection
🎵 Mel-spectrogram based feature extraction
🧠 CNN-based spectral feature learning
🔄 LSTM-based temporal sequence modeling
📊 Statistical acoustic feature extraction
🔗 Hybrid feature-level fusion
🌐 Streamlit prediction interface
📈 Mel-spectrogram visualization
💾 PyTorch model training and inference


| Category         | Technology    |
| ---------------- | ------------- |
| Programming      | Python        |
| Deep Learning    | PyTorch       |
| Audio Processing | Librosa       |
| Data Processing  | NumPy, Pandas |
| Visualization    | Matplotlib    |
| Web Interface    | Streamlit     |
| Dataset          | ASVspoof 5    |


📂 Project Structure

deepfake-audio-detection/
│
├── docs/
│   └── architecture.png
│
├── src/
│   ├── preprocess.py       # Audio feature extraction
│   └── model.py            # CNN-LSTM hybrid model
│
├── data/
│   └── protocols/          # ASVspoof protocol files
│
├── weights/                # Trained model weights
│
├── train.py                # Model training pipeline
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
