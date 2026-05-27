import os
import shutil
import random

# ---------------- SETTINGS ----------------
source_dir = "../data/UCF-101"

train_dir = "../data/train"
val_dir = "../data/val"

split_ratio = 0.8

random.seed(42)

# ---------------- REMOVE OLD SPLITS ----------------
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

# ---------------- CREATE FOLDERS ----------------
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

classes = os.listdir(source_dir)

print("🚀 Splitting dataset...\n")

for cls in classes:

    cls_path = os.path.join(source_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    videos = os.listdir(cls_path)

    random.shuffle(videos)

    split_index = int(len(videos) * split_ratio)

    train_videos = videos[:split_index]
    val_videos = videos[split_index:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # ---------------- COPY TRAIN ----------------
    for v in train_videos:

        shutil.copy(
            os.path.join(cls_path, v),
            os.path.join(train_dir, cls, v)
        )

    # ---------------- COPY VAL ----------------
    for v in val_videos:

        shutil.copy(
            os.path.join(cls_path, v),
            os.path.join(val_dir, cls, v)
        )

    print(
        f"✅ {cls} | "
        f"Train: {len(train_videos)} | "
        f"Val: {len(val_videos)}"
    )

print("\n🎉 Dataset split completed!")