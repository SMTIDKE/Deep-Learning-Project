import torch
import torch.nn as nn

# --------------------------------------------------
# SE ATTENTION FOR 3D CNN
# --------------------------------------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


# --------------------------------------------------
# RESIDUAL BLOCK WITH SE ATTENTION
# --------------------------------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.se = SEBlock3D(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += identity
        return self.relu(out)


# --------------------------------------------------
# TEMPORAL ATTENTION
# --------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)
        context = (weights * x).sum(dim=1)
        return context


# --------------------------------------------------
# FINAL MODEL
# --------------------------------------------------
class CNN3D_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 3D CNN BACKBONE
        self.layer1 = nn.Sequential(
            ResidualBlock3D(3, 64),
            ResidualBlock3D(64, 64),
            nn.MaxPool3d(2)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock3D(64, 128),
            ResidualBlock3D(128, 128),
            nn.MaxPool3d(2)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock3D(128, 256),
            ResidualBlock3D(256, 256),
            nn.MaxPool3d(2)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock3D(256, 512),
            ResidualBlock3D(512, 512),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        # BIDIRECTIONAL LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        # TEMPORAL ATTENTION
        self.attention = TemporalAttention(512)

        # CLASSIFIER
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.squeeze(-1).squeeze(-1)   # (B, C, T)
        x = x.permute(0, 2, 1)          # (B, T, C)

        lstm_out, _ = self.lstm(x)

        context = self.attention(lstm_out)

        out = self.fc(context)
        return out