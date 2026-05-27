# Video Action Recognition using CNN3D + BiLSTM

## Overview
This project performs human action recognition from videos using Deep Learning.

The model combines:
- 3D Convolutional Neural Network (CNN3D)
- Bidirectional LSTM
- Temporal Attention Mechanism

## Features
- Video action classification
- Live surveillance prediction
- Training and testing pipeline
- Accuracy visualization graphs

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy

## Project Structure

video_action_recognition/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── model.py
│   ├── dataset.py
│   └── live_surveillance.py
│
├── requirements.txt
├── README.md
└── .gitignore

## Installation

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python src/train.py
```

## Run Prediction

```bash
python src/predict.py
```

## Model Architecture
CNN3D + BiLSTM + Temporal Attention

## Author
SM TIDKE