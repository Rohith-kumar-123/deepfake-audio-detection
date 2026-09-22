# 🎙️ Robust Deepfake Audio Detection Using Deep Learning

> An AI-powered deepfake audio detection system that combines **Mel-spectrogram analysis**, **CNN-LSTM deep learning**, and **statistical acoustic features** to classify audio as real or AI-generated.

---

## 📖 Overview

The increasing realism of AI-generated and synthetic audio has created challenges for digital media verification and audio security. Detecting manipulated audio automatically is therefore an important machine learning problem.

This project introduces a **hybrid deep learning framework** that integrates:

- **Mel-spectrogram analysis** for time-frequency audio representation.
- **CNN** for extracting local spectral patterns.
- **Bidirectional LSTM** for modeling temporal dependencies.
- **Statistical acoustic features** for complementary audio information.
- **Feature Fusion** to combine learned and statistical representations.
- **Streamlit** web application for an easy-to-use audio detection interface.

The system classifies an uploaded audio sample as either **Real (Bonafide)** or **Deepfake (Spoof)**.

---

# ✨ Features

- 🎧 Deepfake audio detection
- 🎵 Mel-spectrogram based feature extraction
- 🧠 CNN-based spectral feature learning
- 🔄 Bidirectional LSTM temporal modeling
- 📊 Statistical acoustic feature extraction
- 🤝 Hybrid feature-level fusion
- 🌐 Streamlit-based web interface
- 📈 Mel-spectrogram visualization
- ⚡ Audio inference using PyTorch
- 📦 Modular project architecture
- 🔍 Supports `.wav` and `.flac` audio files

---

# 🏗 System Architecture

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
                         │
                         ├───────────────► Statistical Features
                         │                  │
                         └──────────┬───────┘
                                    ▼
                             Feature Fusion
                                    │
                                    ▼
                            Fully Connected
                                    │
                                    ▼
                         Real / Deepfake
```

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Audio Processing | Librosa |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Web Interface | Streamlit |
| Dataset | ASVspoof 5 |
| Version Control | Git, GitHub |

---

# 📂 Project Structure

```text
deepfake-audio-detection/
│
├── app.py                          # Streamlit application
├── train.py                        # Model training pipeline
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── .gitignore
│
├── src/
│   ├── preprocess.py               # Audio feature extraction
│   └── model.py                    # CNN-LSTM hybrid model
│
├── docs/
│   └── architecture.png            # System architecture
│
├── weights/
│   └── model_refined_e*.pth        # Trained model checkpoints
│
└── data/
    ├── protocols/
    │   └── ASVspoof5.train.tsv
    │
    └── flac_T_aa/
        └── *.flac
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/deepfake-audio-detection.git

cd deepfake-audio-detection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset

This project uses the **ASVspoof 5** dataset for training and evaluation.

The training pipeline expects the following directory structure:

```text
data/
│
├── protocols/
│   └── ASVspoof5.train.tsv
│
└── flac_T_aa/
    └── *.flac
```

The dataset is not included in this repository because of its large size.

---

# 🧠 Model Pipeline

## Audio Preprocessing

The input audio is loaded at a sampling rate of **16 kHz**.

The preprocessing pipeline extracts:

- Mel-spectrogram features
- Spectral-centroid variance
- RMS-energy dynamics

The Mel-spectrogram is normalized before being passed to the deep learning model.

---

## CNN Feature Extraction

The CNN processes the Mel-spectrogram and learns local time-frequency patterns from the audio representation.

The architecture contains:

- Convolutional layers
- Batch normalization
- ReLU activation
- Max pooling

---

## Bidirectional LSTM

The CNN output is converted into a temporal sequence and passed through a **Bidirectional LSTM**.

The LSTM captures temporal dependencies and produces a learned representation of the audio sequence.

---

## Statistical Features

In parallel with the CNN-LSTM branch, two statistical acoustic features are extracted:

- Spectral-centroid variance
- RMS-energy dynamics

These features provide complementary information about the acoustic characteristics of the audio.

---

## Feature Fusion

The LSTM representation is combined with the statistical acoustic features.

```text
CNN-LSTM Features
        │
        ├──────────────┐
        │              │
        │      Statistical Features
        │              │
        └──────┬───────┘
               ▼
        Feature Fusion
               │
               ▼
      Fully Connected Layers
               │
               ▼
       Binary Classification
```

---

# 🤖 Models Used

## CNN

Purpose:

- Extract local spectral patterns
- Learn time-frequency representations from Mel-spectrograms

---

## Bidirectional LSTM

Purpose:

- Model temporal dependencies
- Learn sequential patterns from CNN features

---

## Statistical Feature Module

Purpose:

- Provide complementary acoustic information

Features:

- Spectral-centroid variance
- RMS-energy dynamics

---

## Fully Connected Classifier

Purpose:

- Combine CNN-LSTM and statistical representations
- Produce the final binary classification output

Output:

```text
REAL (BONAFIDE)
        or
DEEPFAKE (SPOOF)
```

---

# 🏋️ Training

After placing the ASVspoof 5 files in the expected directories, run:

```bash
python train.py
```

The training pipeline:

1. Reads the ASVspoof protocol file.
2. Filters samples based on locally available audio files.
3. Creates a balanced set of bonafide and spoof samples.
4. Extracts Mel-spectrogram and statistical features.
5. Passes the features through the hybrid CNN-LSTM model.
6. Calculates binary classification loss.
7. Updates model parameters using the Adam optimizer.
8. Saves model checkpoints after each epoch.

The generated checkpoints are stored inside:

```text
weights/
```

Example:

```text
model_refined_e1.pth
model_refined_e2.pth
model_refined_e3.pth
...
model_refined_e10.pth
```

---

# 🌐 Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides an interface where users can:

- Upload `.wav` or `.flac` audio
- Listen to the uploaded audio
- Receive a Real / Deepfake prediction
- View prediction confidence
- Visualize the Mel-spectrogram

---

# 📈 Results

The accompanying project report reports **99.30% accuracy on the stated validation dataset**.

The project also describes testing on in-the-wild audio samples. The report notes that factors such as **background music and audio compression** can affect predictions.

> Results are reported from the evaluation described in the accompanying project report.

---

# 🔮 Future Improvements

- Improve robustness against background noise
- Improve performance on compressed audio
- Evaluate using larger and more diverse datasets
- Add automated testing for preprocessing and inference
- Improve inference efficiency
- Explore additional audio representations
- Experiment with alternative deep learning architectures
- Add API-based inference support
- Deploy the application using Docker
- Cloud deployment

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is intended for academic and educational purposes.

---

# 👨‍💻 Authors

**R. Rohith Kumar**  
**M. Vishnu Vardhana Raju**

Vellore Institute of Technology

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it
