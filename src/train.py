import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import CNN3D_LSTM
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt   # ✅ NEW

# ---------------- SETTINGS ----------------
DATA_PATH = "../data/train"
CHECKPOINT_PATH = "checkpoint.pth"
BEST_MODEL_PATH = "best_model.pth"
GRAPH_PATH = "accuracy_vs_epoch.png"   # ✅ NEW

BATCH_SIZE = 4
EPOCHS = 80
LR = 3e-4

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET ----------------
dataset = VideoDataset(DATA_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

num_classes = len(dataset.classes)

print(f"Dataset size   : {len(dataset)}")
print(f"Num classes    : {num_classes}")

# ---------------- MODEL ----------------
model = CNN3D_LSTM(num_classes=num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ✅ safer AMP
scaler = torch.amp.GradScaler(
    "cuda" if torch.cuda.is_available() else "cpu"
)

start_epoch = 0
best_acc = 0

# ✅ STORE ACCURACY HISTORY
accuracy_history = []

# ---------------- LOAD CHECKPOINT ----------------
if os.path.exists(CHECKPOINT_PATH):

    print("🔄 Loading checkpoint...")

    checkpoint = torch.load(CHECKPOINT_PATH)

    old_state = checkpoint["model_state"]
    new_state = model.state_dict()

    matched, skipped = 0, 0

    for name, param in old_state.items():

        if name in new_state and param.size() == new_state[name].size():
            new_state[name] = param
            matched += 1
        else:
            skipped += 1

    model.load_state_dict(new_state)

    if "optimizer_state" in checkpoint and matched > 0:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

    best_acc = checkpoint.get("best_acc", 0)

    # ✅ LOAD OLD ACCURACY HISTORY IF AVAILABLE
    accuracy_history = checkpoint.get("accuracy_history", [])

    print(f"✅ Loaded layers : {matched}")
    print(f"⚠️ Skipped layers: {skipped}")
    print(f"Resume epoch    : {start_epoch}")

# ---------------- TIME FORMAT ----------------
def format_time(seconds):

    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hrs:02}:{mins:02}:{secs:02}"

# ---------------- TRAINING ----------------
print("\n🚀 Training started...\n")

training_start = time.time()

for epoch in range(start_epoch, EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        dynamic_ncols=True,
        leave=True
    )

    for videos, labels in progress_bar:

        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(
            "cuda" if torch.cuda.is_available() else "cpu"
        ):

            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * videos.size(0)

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total

        # ✅ CLEAN PERCENTAGE DISPLAY
        progress_bar.set_postfix_str(
            f"loss={running_loss/total:.4f} | acc={acc:.2f}%"
        )

    # ---------------- EPOCH RESULT ----------------
    epoch_acc = 100 * correct / total

    # ✅ SAVE ACCURACY
    accuracy_history.append(epoch_acc)

    epoch_time = format_time(time.time() - epoch_start)

    tqdm.write(f"\nEpoch {epoch+1} completed")
    tqdm.write(f"Accuracy : {epoch_acc:.2f}%")
    tqdm.write(f"Time     : {epoch_time}\n")

    # ---------------- SAVE CHECKPOINT ----------------
    torch.save({

        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,

        # ✅ SAVE ACCURACY HISTORY
        "accuracy_history": accuracy_history

    }, CHECKPOINT_PATH)

    # ---------------- SAVE BEST MODEL ----------------
    if epoch_acc > best_acc:

        best_acc = epoch_acc

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        # tqdm.write("🏆 Best model updated!")

# ---------------- TOTAL TIME ----------------
total_time = format_time(time.time() - training_start)

print("======================================")
print(f"🏁 Total Training Time: {total_time}")
print("======================================")

# =========================================================
# GENERATE ACCURACY vs EPOCH GRAPH
# =========================================================

epochs_range = range(1, len(accuracy_history) + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    accuracy_history,
    marker='o',
    linewidth=2
)

plt.title("Accuracy vs Epoch")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

# ✅ FORCE 0–100 SCALE
plt.ylim(0, 100)

plt.grid(True)

plt.xticks(epochs_range)

# SAVE GRAPH
plt.savefig(GRAPH_PATH)

# SHOW GRAPH
plt.show()

print(f"\n📊 Graph saved as: {GRAPH_PATH}")