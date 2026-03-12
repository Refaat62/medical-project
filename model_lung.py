import torch
import torch.nn as nn
from torchvision import models


class AttentionBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class HybridModel(nn.Module):
    """ResNet-152 + EfficientNet-B5 hybrid with attention classifier."""

    def __init__(self, num_classes: int = 6):
        super().__init__()

        # ── ResNet-152 backbone ──
        self.resnet = models.resnet152(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # ── EfficientNet-B5 backbone ──
        self.effnet = models.efficientnet_b5(weights=None)
        effnet_features = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Identity()

        combined = resnet_features + effnet_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            AttentionBlock(1024),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat((self.resnet(x), self.effnet(x)), dim=1)
        return self.classifier(feats)
