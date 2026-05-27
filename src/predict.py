import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import time
from model import CNN3D_LSTM

# ---------------- TIME FORMAT ----------------
def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}h:{mins:02}m:{secs:02}s:{millis:03}ms"

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CLASS DETECTION ----------------
def get_class_names(train_dir):
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

classes = get_class_names("../data/train")

# ---------------- LOAD MODEL ----------------
model = CNN3D_LSTM(num_classes=len(classes))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- FRAME SAMPLING ----------------
def sample_frames(video_path, frames_per_clip=16):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError("❌ Video has 0 frames.")

    step = max(total_frames // frames_per_clip, 1)
    indices = [i * step for i in range(frames_per_clip)]

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (112, 112))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < frames_per_clip:
        frames.append(frames[-1])

    frames = np.array(frames)
    frames = torch.tensor(frames).float()
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)

    return frames


# ---------------- MULTI-CLIP PREDICTION ----------------
def predict_video(video_path, num_clips=3):

    clip_predictions = []

    for _ in range(num_clips):
        clip = sample_frames(video_path)
        clip = clip.to(device)

        with torch.no_grad():
            output = model(clip)
            probs = F.softmax(output, dim=1)

        clip_predictions.append(probs)

    avg_probs = torch.mean(torch.stack(clip_predictions), dim=0)

    return avg_probs


# ---------------- RUN PREDICTION ----------------
video_path = "../data/test/test_video.avi"

print("\n🚀 Running prediction...\n")

start_time = time.time()

probs = predict_video(video_path)

end_time = time.time()

inference_time = end_time - start_time

# Top 3 predictions
top_probs, top_idxs = torch.topk(probs, 3)

print("✅ Prediction Result")
print("----------------------------------")

for i in range(3):
    cls_name = classes[top_idxs[0][i].item()]
    conf = top_probs[0][i].item() * 100
    print(f"{i+1}. {cls_name} : {conf:.2f}%")

print("----------------------------------")
print(f"⏱ Inference Time : {format_time(inference_time)}")

print("\n--- All Class Probabilities ---")
for i, cls in enumerate(classes):
    print(f"{cls}: {probs[0][i].item() * 100:.2f}%")