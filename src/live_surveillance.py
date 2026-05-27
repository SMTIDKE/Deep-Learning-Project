import cv2
import torch
import numpy as np
import time
import os
from model import CNN3D_LSTM

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- AUTO CLASS DETECTION ----------------
train_path = "../data/train"
classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

# ---------------- LOAD MODEL ----------------
model = CNN3D_LSTM(num_classes=len(classes))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- PARAMETERS ----------------
FRAMES_PER_CLIP = 16
frame_buffer = []

cap = cv2.VideoCapture(0)

start_time = time.time()
prev_time = time.time()
latency = 0

print("🚀 AI Live Surveillance Started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()

    frame_resized = cv2.resize(frame, (112, 112))
    frame_normalized = frame_resized / 255.0
    frame_buffer.append(frame_normalized)

    # ================= INFERENCE =================
    if len(frame_buffer) == FRAMES_PER_CLIP:

        clip = np.array(frame_buffer)
        clip = torch.tensor(clip, dtype=torch.float32)
        clip = clip.permute(3, 0, 1, 2).unsqueeze(0).to(device)

        start_infer = time.time()

        with torch.no_grad():
            outputs = model(clip)
            probs = torch.softmax(outputs, dim=1)

        latency = (time.time() - start_infer) * 1000

        top_probs, top_indices = torch.topk(probs, 3)

        frame_buffer.pop(0)

        # ================= UI PANEL (SMALLER) =================
        panel_w, panel_h = 380, 150
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "AI ACTION RECOGNITION",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)

        # ================= TOP 3 PREDICTIONS =================
        for i in range(3):
            cls = classes[top_indices[0][i]]
            conf = top_probs[0][i].item() * 100

            y = 65 + i * 28

            cv2.putText(frame, f"{i+1}. {cls}",
                        (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            # Bar background
            cv2.rectangle(frame, (200, y - 12), (320, y - 2), (50, 50, 50), -1)

            bar_width = int(conf * 1.2)

            color = (0, 255, 0)
            if conf < 50:
                color = (0, 165, 255)
            if conf < 30:
                color = (0, 0, 255)

            cv2.rectangle(frame, (200, y - 12), (200 + bar_width, y - 2), color, -1)

            cv2.putText(frame,
                        f"{conf:.1f}%",
                        (325, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1)

    # ================= FPS =================
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2)

    # ================= LATENCY =================
    cv2.putText(frame,
                f"LAT: {latency:.1f} ms",
                (frame.shape[1] - 170, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2)

    # ================= RUNTIME =================
    runtime = int(time.time() - start_time)
    hrs = runtime // 3600
    mins = (runtime % 3600) // 60
    secs = runtime % 60

    cv2.putText(frame,
                f"TIME: {hrs:02}:{mins:02}:{secs:02}",
                (frame.shape[1] - 200, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)

    # ================= BORDER =================
    h, w, _ = frame.shape
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 255), 1)

    cv2.imshow("AI Live Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()