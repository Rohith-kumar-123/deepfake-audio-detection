import torch
import torch.nn as nn


class DeepfakeHybridModel(nn.Module):

    def __init__(self):
        super(DeepfakeHybridModel, self).__init__()

        # CNN Path
        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        # LSTM Path
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # Statistical Feature Path
        self.stats_fc = nn.Linear(2, 64)

        # Final Decision Layers
        self.fc1 = nn.Linear(256 + 64, 128)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, spec, stats):

        # 1. CNN - Spatial Features
        x = self.pool(
            torch.relu(
                self.bn1(
                    self.conv1(spec)
                )
            )
        )

        x = self.pool(
            torch.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        # 2. Reshape CNN output for LSTM
        b, c, h, w = x.size()

        x = x.permute(
            0, 3, 1, 2
        ).contiguous()

        x = x.view(
            b,
            w,
            c * h
        )

        # 3. LSTM - Temporal Features
        lstm_out, _ = self.lstm(x)

        lstm_feat = lstm_out[:, -1, :]

        # 4. Statistical Features
        stats_feat = torch.relu(
            self.stats_fc(stats)
        )

        # 5. Feature Fusion
        combined = torch.cat(
            (lstm_feat, stats_feat),
            dim=1
        )

        x = torch.relu(
            self.fc1(combined)
        )

        x = self.dropout(x)

        return self.fc2(x)