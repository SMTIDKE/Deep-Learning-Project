import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=16, train=True):
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.train = train

        self.classes = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )

        self.samples = []

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)

            for video in os.listdir(cls_path):
                if video.lower().endswith(VIDEO_EXTENSIONS):
                    self.samples.append(
                        (os.path.join(cls_path, video), label)
                    )

    def __len__(self):
        return len(self.samples)

    def _get_frame_indices(self, total_frames):

        if total_frames <= self.frames_per_video:
            return np.arange(total_frames)

        if self.train:
            return np.sort(
                np.random.choice(total_frames, self.frames_per_video, replace=False)
            )
        else:
            step = total_frames // self.frames_per_video
            return np.arange(0, total_frames, step)[:self.frames_per_video]

    def extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return torch.zeros(3, self.frames_per_video, 112, 112)

        frame_indices = self._get_frame_indices(total_frames)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.resize(frame, (112, 112))

            if self.train and np.random.rand() < 0.5:
                frame = cv2.flip(frame, 1)

            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return torch.zeros(3, self.frames_per_video, 112, 112)

        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])

        frames = np.array(frames)
        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)

        return frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)
        return frames, label