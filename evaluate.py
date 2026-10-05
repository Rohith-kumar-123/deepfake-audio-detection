import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.model import DeepfakeHybridModel
from src.preprocess import extract_hybrid_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VALIDATION_PATH = "data/validation_split.tsv"
MODEL_PATH = "weights/best_model.pth"
OUTPUT_PATH = "docs/confusion_matrix.png"

AUDIO_DIR = r"C:\Users\rvroh\Downloads\flac_T_aa\flac_T"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load validation data
# --------------------------------------------------

print("Loading validation dataset...")

df = pd.read_csv(
    VALIDATION_PATH,
    sep="\t",
    header=None
)

# ASVspoof protocol:
# Column 0 = Speaker ID
# Column 1 = Audio ID
# Column 8 = Label

audio_ids = df.iloc[:, 1].astype(str)
labels = df.iloc[:, 8].astype(str)

y_true = []
y_pred = []

print(f"Validation samples: {len(df)}")


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading best model...")

model = DeepfakeHybridModel().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(f"Device: {device}")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\nRunning evaluation...")

with torch.no_grad():

    for index, audio_id in enumerate(audio_ids):

        audio_path = os.path.join(
            AUDIO_DIR,
            audio_id + ".flac"
        )

        if not os.path.exists(audio_path):
            print(
                f"Warning: audio not found: {audio_path}"
            )
            continue

        try:

            # Extract hybrid features
            spec, stats = extract_hybrid_features(
                audio_path
            )

            # Normalize spectrogram
            spec = (
                spec - spec.mean()
            ) / (
                spec.std() + 1e-6
            )

            # Normalize statistical features
            stats = (
                stats - stats.mean()
            ) / (
                stats.std() + 1e-6
            )

            # Add batch dimension
            spec = spec.unsqueeze(0).to(device)
            stats = stats.unsqueeze(0).to(device)

            # Model prediction
            output = model(
                spec,
                stats
            )

            probability = torch.sigmoid(
                output
            ).item()

            prediction = (
                1 if probability >= 0.5 else 0
            )

            # Store prediction
            y_pred.append(prediction)

            # Store TRUE label for this successfully
            # evaluated audio file
            true_label = (
                1 if labels.iloc[index] == "spoof" else 0
            )

            y_true.append(true_label)

        except Exception as e:

            print(
                f"Error processing {audio_id}: {e}"
            )

        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{len(df)}"
            )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

evaluated_count = len(y_pred)

print("\nCalculating metrics...")

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"Evaluated Samples : {evaluated_count}"
)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Precision          : {precision * 100:.2f}%"
)

print(
    f"Recall             : {recall * 100:.2f}%"
)

print(
    f"F1 Score           : {f1 * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Details:")

print(
    f"True Negative  (TN): {tn}"
)

print(
    f"False Positive (FP): {fp}"
)

print(
    f"False Negative (FN): {fn}"
)

print(
    f"True Positive  (TP): {tp}"
)


# --------------------------------------------------
# Save confusion matrix image
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

im = ax.imshow(
    cm,
    interpolation="nearest"
)

ax.set_title(
    "Deepfake Audio Detection\nConfusion Matrix"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Real",
    "Deepfake"
])

ax.set_yticklabels([
    "Real",
    "Deepfake"
])


# Add numbers inside confusion matrix
for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


fig.colorbar(
    im,
    ax=ax
)

fig.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"\nConfusion matrix saved to: {OUTPUT_PATH}"
)

print("\nEvaluation complete.")