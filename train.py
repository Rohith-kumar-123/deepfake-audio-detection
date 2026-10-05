import os
import sys
import random

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.model import DeepfakeHybridModel
from src.preprocess import extract_hybrid_features
DRY_RUN = False  # Set to True to test dataset matching and splitting without training                                      

# ============================================================
# CONFIGURATION
# ============================================================

PROTOCOL_PATH = r"C:\Users\rvroh\Downloads\ASVspoof5_protocols\ASVspoof5.train.tsv"
AUDIO_DIR = r"C:\Users\rvroh\Downloads\flac_T_aa\flac_T"

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
EPOCHS = 10

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

CHECKPOINT_DIR = "weights"
SPLIT_DIR = "data"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 60)
print("DEEPFAKE AUDIO DETECTION - TRAINING")
print("=" * 60)

print("\nChecking dataset paths...")

if not os.path.exists(PROTOCOL_PATH):
    print(f"ERROR: Protocol file not found:\n{PROTOCOL_PATH}")
    sys.exit(1)

if not os.path.exists(AUDIO_DIR):
    print(f"ERROR: Audio directory not found:\n{AUDIO_DIR}")
    sys.exit(1)

print("Protocol file found.")
print("Audio directory found.")


# ============================================================
# FIND LOCAL AUDIO FILES
# ============================================================

print("\nScanning local audio files...")

local_files = {
    os.path.splitext(filename)[0]
    for filename in os.listdir(AUDIO_DIR)
    if filename.lower().endswith(".flac")
}

print(f"Local FLAC files found: {len(local_files)}")


# ============================================================
# LOAD PROTOCOL
# ============================================================

print("\nLoading ASVspoof protocol...")

df = pd.read_csv(
    PROTOCOL_PATH,
    sep=r"\s+",
    header=None,
    engine="python"
)

print(f"Protocol rows: {len(df)}")


# ============================================================
# MATCH PROTOCOL WITH LOCAL FILES
# ============================================================

print("\nMatching protocol entries with local audio files...")

# Column 1 contains the audio file ID.
df = df[df[1].isin(local_files)].copy()

print(f"Matched samples: {len(df)}")

if len(df) == 0:
    print("ERROR: No protocol entries matched the local audio files.")
    sys.exit(1)


# ============================================================
# CREATE BINARY LABEL
# ============================================================

# bonafide = 0 (Real)
# spoof    = 1 (Deepfake)

df["label"] = (
    df[8].str.lower().str.strip() != "bonafide"
).astype(int)

bonafide_count = (df["label"] == 0).sum()
spoof_count = (df["label"] == 1).sum()

print("\nMatched class distribution:")
print(f"Bonafide (Real): {bonafide_count}")
print(f"Spoof (Deepfake): {spoof_count}")


# ============================================================
# CREATE BALANCED DATASET
# ============================================================

print("\nCreating balanced dataset...")

real_df = df[df["label"] == 0]
fake_df = df[df["label"] == 1]

if len(real_df) == 0 or len(fake_df) == 0:
    print("ERROR: One of the classes has no samples.")
    sys.exit(1)

samples_per_class = min(len(real_df), len(fake_df))

real_df = real_df.sample(
    n=samples_per_class,
    random_state=RANDOM_STATE
)

fake_df = fake_df.sample(
    n=samples_per_class,
    random_state=RANDOM_STATE
)

balanced_df = pd.concat(
    [real_df, fake_df],
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Balanced dataset size: {len(balanced_df)}")
print(f"Real samples: {(balanced_df['label'] == 0).sum()}")
print(f"Deepfake samples: {(balanced_df['label'] == 1).sum()}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")

train_df, val_df = train_test_split(
    balanced_df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=balanced_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

print("\nTraining distribution:")
print(
    f"Real: {(train_df['label'] == 0).sum()} | "
    f"Deepfake: {(train_df['label'] == 1).sum()}"
)

print("\nValidation distribution:")
print(
    f"Real: {(val_df['label'] == 0).sum()} | "
    f"Deepfake: {(val_df['label'] == 1).sum()}"
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_split_path = os.path.join(
    SPLIT_DIR,
    "train_split.tsv"
)

val_split_path = os.path.join(
    SPLIT_DIR,
    "validation_split.tsv"
)

train_df.to_csv(
    train_split_path,
    sep="\t",
    index=False
)

val_df.to_csv(
    val_split_path,
    sep="\t",
    index=False
)
if DRY_RUN:
    print("\nDRY RUN COMPLETE.")
    print("Dataset matching and train/validation split are working.")
    print("Training has NOT started.")
    sys.exit(0)
print("\nSaved dataset splits:")
print(train_split_path)
print(val_split_path)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# ============================================================
# MODEL
# ============================================================

model = DeepfakeHybridModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, dataframe, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for i in range(len(dataframe)):

            try:
                row = dataframe.iloc[i]

                label = int(row["label"])

                audio_path = os.path.join(
                    AUDIO_DIR,
                    str(row[1]) + ".flac"
                )

                spec, stats = extract_hybrid_features(
                    audio_path
                )

                spec = (
                    spec - spec.mean()
                ) / (
                    spec.std() + 1e-6
                )

                stats = (
                    stats - stats.mean()
                ) / (
                    stats.std() + 1e-6
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

                probability = torch.sigmoid(
                    output
                ).item()

                prediction = (
                    1 if probability > 0.5 else 0
                )

                if prediction == label:
                    correct += 1

                total_loss += loss.item()
                total += 1

            except Exception as e:
                print(
                    f"Validation sample {i} skipped: {e}"
                )

    if total == 0:
        return 0.0, 0.0

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_epoch = 0

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    epoch_losses = []
    correct_predictions = 0
    total_processed = 0

    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

    for i in range(len(train_df)):

        try:
            row = train_df.iloc[i]

            label = int(row["label"])

            audio_path = os.path.join(
                AUDIO_DIR,
                str(row[1]) + ".flac"
            )

            spec, stats = extract_hybrid_features(
                audio_path
            )

            spec = (
                spec - spec.mean()
            ) / (
                spec.std() + 1e-6
            )

            stats = (
                stats - stats.mean()
            ) / (
                stats.std() + 1e-6
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

            probability = torch.sigmoid(
                output
            ).item()

            prediction = (
                1 if probability > 0.5 else 0
            )

            if prediction == label:
                correct_predictions += 1

            epoch_losses.append(
                loss.item()
            )

            total_processed += 1

            if i % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}] "
                    f"Sample [{i}/{len(train_df)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Pred: {probability:.4f}"
                )

        except Exception as e:

            print(
                f"Skipping training sample {i}: {e}"
            )

            continue

    # --------------------------------------------------------
    # TRAINING METRICS
    # --------------------------------------------------------

    if total_processed == 0:

        print("ERROR: No training samples processed.")

        sys.exit(1)

    train_loss = (
        sum(epoch_losses)
        / len(epoch_losses)
    )

    train_accuracy = (
        correct_predictions
        / total_processed
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_accuracy = evaluate(
        model,
        val_df,
        device
    )

    print("\nEpoch Results")
    print("-" * 40)
    print(
        f"Training Loss:      {train_loss:.4f}"
    )
    print(
        f"Training Accuracy:  {train_accuracy * 100:.2f}%"
    )
    print(
        f"Validation Loss:    {val_loss:.4f}"
    )
    print(
        f"Validation Accuracy:{val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE EVERY CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"model_refined_e{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        best_model_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"New best model saved! "
            f"Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best model: "
    f"{os.path.join(CHECKPOINT_DIR, 'best_model.pth')}"
)