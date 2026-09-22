import os
import sys

import pandas as pd
import torch
import torch.nn as nn

from src.model import DeepfakeHybridModel
from src.preprocess import extract_hybrid_features


# -----------------------------
# 1. CONFIGURATION
# -----------------------------

TSV_PATH = "data/protocols/ASVspoof5.train.tsv"
AUDIO_DIR = "data/flac_T_aa/"

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
EPOCHS = 10


# -----------------------------
# 2. DATA SCANNING & BALANCING
# -----------------------------

print("Scanning directory and filtering TSV...")

if not os.path.exists(AUDIO_DIR):
    print(f"Error: {AUDIO_DIR} not found.")
    sys.exit()

local_files = {
    f.split(".")[0]
    for f in os.listdir(AUDIO_DIR)
    if f.endswith(".flac")
}

df = pd.read_csv(
    TSV_PATH,
    sep=r"\s+",
    header=None,
    engine="python"
)

df = df[df[1].isin(local_files)]

real = df[df[7].str.lower() == "bonafide"]
fake = df[df[7].str.lower() != "bonafide"]


# Create 1:1 balanced dataset
num_samples = min(
    len(real),
    len(fake)
)

if num_samples == 0:
    print("Error: No files found. Check your TSV or folder.")
    sys.exit()

balanced_df = pd.concat(
    [
        real.sample(num_samples),
        fake.sample(num_samples)
    ]
).sample(
    frac=1
).reset_index(drop=True)

print(
    f"Balanced Dataset Created: "
    f"{len(balanced_df)} samples ready."
)


# -----------------------------
# 3. MODEL SETUP
# -----------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model = DeepfakeHybridModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss()

os.makedirs(
    "weights",
    exist_ok=True
)


# -----------------------------
# 4. TRAINING LOOP
# -----------------------------

model.train()

for epoch in range(EPOCHS):

    epoch_losses = []
    correct_predictions = 0
    total_processed = 0

    print(
        f"\n--- Epoch {epoch + 1} ---"
    )

    for i in range(len(balanced_df)):

        try:
            row = balanced_df.iloc[i]

            label = (
                0
                if str(row[7]).lower() == "bonafide"
                else 1
            )

            audio_path = os.path.join(
                AUDIO_DIR,
                str(row[1]) + ".flac"
            )

            spec, stats = extract_hybrid_features(
                audio_path
            )

            spec = (
                (spec - spec.mean())
                / (spec.std() + 1e-6)
            )

            stats = (
                (stats - stats.mean())
                / (stats.std() + 1e-6)
            )

            spec = spec.unsqueeze(0).to(device)
            stats = stats.unsqueeze(0).to(device)

            target = torch.tensor(
                [[label]],
                dtype=torch.float32
            ).to(device)

            output = model(
                spec,
                stats
            )

            loss = criterion(
                output,
                target
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Track training statistics
            prob = torch.sigmoid(
                output
            ).item()

            pred_label = (
                1
                if prob > 0.5
                else 0
            )

            if pred_label == label:
                correct_predictions += 1

            epoch_losses.append(
                loss.item()
            )

            total_processed += 1

            if i % 20 == 0:
                print(
                    f"E[{epoch + 1}] "
                    f"I[{i}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Pred: {prob:.4f}"
                )

        except Exception as e:
            print(
                f"Skipping sample {i}: {e}"
            )
            continue


    # -----------------------------
    # 5. EPOCH SUMMARY
    # -----------------------------

    if total_processed > 0:

        avg_loss = (
            sum(epoch_losses)
            / len(epoch_losses)
        )

        accuracy = (
            correct_predictions
            / total_processed
        ) * 100

        print(
            f"\n>> EPOCH {epoch + 1} FINISHED <<"
        )

        print(
            f">> Avg Loss: {avg_loss:.4f}"
        )

        print(
            f">> Training Accuracy: "
            f"{accuracy:.2f}%"
        )


    # Save model weights
    torch.save(
        model.state_dict(),
        f"weights/model_refined_e{epoch + 1}.pth"
    )


print("\n--- Training Finished! ---")