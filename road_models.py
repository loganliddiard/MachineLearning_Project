from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


@dataclass(frozen=True)
class AttentionConfig:
    # For notebook parity with AttentionPlacement_CNN.ipynb
    attention_type: str = "none"  # "none" | "CBAM"
    position: str = "none"  # "none" | "early" | "mid" | "late" | "all"


def _make_attention(attention_type: str, channels: int) -> Optional[nn.Module]:
    if attention_type == "none":
        return None
    if attention_type == "CBAM":
        return CBAM(channels)
    raise ValueError(f"Unknown attention_type: {attention_type}")


class LeNet224_Attn(nn.Module):
    """A simple CNN baseline adapted for 224x224 images.

    Uses adaptive pooling so the classifier shape is input-size agnostic.
    """

    def __init__(
        self,
        num_classes: int,
        attn: AttentionConfig = AttentionConfig(),
    ) -> None:
        super().__init__()
        self.attn_cfg = attn

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # One attention module reused (for early/mid/late), or per-stage (for all)
        self.attention = None
        self.attn1 = self.attn2 = self.attn3 = None

        if attn.attention_type != "none":
            if attn.position == "all":
                self.attn1 = _make_attention(attn.attention_type, 32)
                self.attn2 = _make_attention(attn.attention_type, 64)
                self.attn3 = _make_attention(attn.attention_type, 128)
            else:
                ch = 32 if attn.position == "early" else 64 if attn.position == "mid" else 128
                self.attention = _make_attention(attn.attention_type, ch)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        if self.attn_cfg.position == "early" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn1 is not None:
            x = self.attn1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        if self.attn_cfg.position == "mid" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn2 is not None:
            x = self.attn2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        if self.attn_cfg.position == "late" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn3 is not None:
            x = self.attn3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallAlexNet224_Attn(nn.Module):
    """A lightweight AlexNet-ish baseline adapted for 224x224 images."""

    def __init__(
        self,
        num_classes: int,
        attn: AttentionConfig = AttentionConfig(),
    ) -> None:
        super().__init__()
        self.attn_cfg = attn

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))

        self.attention = None
        self.attn1 = self.attn2 = self.attn3 = self.attn4 = None

        if attn.attention_type != "none":
            if attn.position == "all":
                self.attn1 = _make_attention(attn.attention_type, 64)
                self.attn2 = _make_attention(attn.attention_type, 128)
                self.attn3 = _make_attention(attn.attention_type, 256)
                self.attn4 = _make_attention(attn.attention_type, 512)
            else:
                ch = 64 if attn.position == "early" else 256 if attn.position == "mid" else 512
                self.attention = _make_attention(attn.attention_type, ch)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.attn_cfg.position == "early" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn1 is not None:
            x = self.attn1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        if self.attn_cfg.position == "mid" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn3 is not None:
            x = self.attn3(x)

        x = self.conv4(x)
        if self.attn_cfg.position == "late" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn4 is not None:
            x = self.attn4(x)

        x = self.gap(x)
        return self.classifier(x)


class ResNet18_224_Attn(nn.Module):
    """ResNet-18 (torchvision) with optional CBAM attention placement.

    Placement meanings (matching AttentionConfig):
    - early: after layer1
    - mid: after layer2
    - late: after layer4
    - all: after layer1, layer2, layer3, layer4
    """

    def __init__(
        self,
        num_classes: int,
        attn: AttentionConfig = AttentionConfig(),
    ) -> None:
        super().__init__()
        self.attn_cfg = attn

        from torchvision.models import resnet18

        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        self.attention = None
        self.attn1 = self.attn2 = self.attn3 = self.attn4 = None

        if attn.attention_type != "none":
            if attn.position == "all":
                self.attn1 = _make_attention(attn.attention_type, 64)
                self.attn2 = _make_attention(attn.attention_type, 128)
                self.attn3 = _make_attention(attn.attention_type, 256)
                self.attn4 = _make_attention(attn.attention_type, 512)
            else:
                ch = 64 if attn.position == "early" else 128 if attn.position == "mid" else 512
                self.attention = _make_attention(attn.attention_type, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if self.attn_cfg.position == "early" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn1 is not None:
            x = self.attn1(x)

        x = self.backbone.layer2(x)
        if self.attn_cfg.position == "mid" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn2 is not None:
            x = self.attn2(x)

        x = self.backbone.layer3(x)
        if self.attn_cfg.position == "all" and self.attn3 is not None:
            x = self.attn3(x)

        x = self.backbone.layer4(x)
        if self.attn_cfg.position == "late" and self.attention is not None:
            x = self.attention(x)
        if self.attn_cfg.position == "all" and self.attn4 is not None:
            x = self.attn4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x)


def torchvision_resnet18(num_classes: int) -> nn.Module:
    # Kept simple: no pretrained weights to avoid downloads.
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def torchvision_resnet18_attn(
    *,
    num_classes: int,
    attn: AttentionConfig = AttentionConfig(),
) -> nn.Module:
    return ResNet18_224_Attn(num_classes=num_classes, attn=attn)
