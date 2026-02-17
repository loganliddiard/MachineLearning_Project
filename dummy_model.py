
import os
import csv
import os
import numpy as np
import pandas as pd
from collections import Counter

## build CSV

# Root dataset directory
data_dir = "UDOT WINTER ROAD CONDITIONS.v1i.folder"

# Splits
splits = {
    "train": os.path.join(data_dir, "train"),
    "valid": os.path.join(data_dir, "valid"),
    "test": os.path.join(data_dir, "test"),
}

# Output CSV file
output_csv = os.path.join(data_dir, "dataset_index.csv")

# Supported image extensions
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


rows = []

for split_name, split_path in splits.items():

    if not os.path.exists(split_path):
        print(f"Warning: {split_path} does not exist")
        continue

    for label in os.listdir(split_path):

        label_path = os.path.join(split_path, label)

        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):

            if filename.lower().endswith(IMG_EXTENSIONS):

                image_id = filename

                rows.append({
                    "image_id": image_id,
                    "split": split_name,
                    "label": label
                })


# Write CSV
with open(output_csv, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["image_id", "split", "label"]
    )

    writer.writeheader()
    writer.writerows(rows)


print(f"CSV saved to: {output_csv}")
print(f"Total images indexed: {len(rows)}")



# -----------------------------
# Config
# -----------------------------
data_dir = "UDOT WINTER ROAD CONDITIONS.v1i.folder"
csv_path = os.path.join(data_dir, "dataset_index.csv")

# For reproducible random baseline
SEED = 42

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(csv_path)

required_cols = {"image_id", "split", "label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"dataset_index.csv is missing columns: {missing}")

# Split data
train_df = df[df["split"].isin(["train"])].copy()
test_df = df[df["split"].isin(["test"])].copy()

if len(test_df) == 0:
    raise ValueError("No rows found for split='test' in dataset_index.csv")

# Labels universe (use all labels seen in TRAIN for a fair baseline)
train_labels = sorted(train_df["label"].unique().tolist())
if len(train_labels) == 0:
    raise ValueError("No labels found in TRAIN split. Check your CSV / folder structure.")

# -----------------------------
# Baseline 1: Random guess (uniform)
# -----------------------------
rng = np.random.default_rng(SEED)
random_preds = rng.choice(train_labels, size=len(test_df), replace=True)

# -----------------------------
# Baseline 2: Most-common label (from TRAIN)
# -----------------------------
label_counts = Counter(train_df["label"].tolist())
most_common_label, most_common_count = label_counts.most_common(1)[0]
majority_preds = np.array([most_common_label] * len(test_df))

# -----------------------------
# Metrics
# -----------------------------
y_true = test_df["label"].to_numpy()

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def per_class_accuracy(y_true, y_pred, labels):
    out = {}
    for lab in labels:
        mask = (y_true == lab)
        if mask.sum() == 0:
            out[lab] = None
        else:
            out[lab] = float(np.mean(y_pred[mask] == lab))
    return out

acc_random = accuracy(y_true, random_preds)
acc_majority = accuracy(y_true, majority_preds)

# use labels seen in test too, so the per-class table is readable
test_labels = sorted(test_df["label"].unique().tolist())

pca_random = per_class_accuracy(y_true, random_preds, test_labels)
pca_majority = per_class_accuracy(y_true, majority_preds, test_labels)

# -----------------------------
# Report
# -----------------------------
print("=== Dummy Baselines (TEST only) ===")
print(f"Test samples: {len(test_df)}")
print(f"Labels (from TRAIN): {train_labels}")
print()

print("TRAIN label distribution:")
total_train = sum(label_counts.values())
for lab, cnt in label_counts.most_common():
    print(f"  {lab:>12}: {cnt} ({cnt/total_train:.3%})")
print()
print(f"Most common label (TRAIN): {most_common_label} ({most_common_count}/{total_train} = {most_common_count/total_train:.3%})")
print()

print("---- Results on TEST ----")
print(f"Random guess (uniform) accuracy:     {acc_random:.3%}")
print(f"Most-common-label accuracy:          {acc_majority:.3%}")
print()

print("Per-class accuracy on TEST (Random):")
for lab in test_labels:
    val = pca_random[lab]
    print(f"  {lab:>12}: {'N/A' if val is None else f'{val:.3%}'}")

print("\nPer-class accuracy on TEST (Majority):")
for lab in test_labels:
    val = pca_majority[lab]
    print(f"  {lab:>12}: {'N/A' if val is None else f'{val:.3%}'}")

