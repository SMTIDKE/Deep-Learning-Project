import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=16):
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for video in os.listdir(cls_path):
                self.samples.append(
                    (os.path.join(cls_path, video), label)
                )

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // self.frames_per_video, 1)

        for i in range(self.frames_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (112, 112))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])

        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)

        return frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)
        return frames, label
