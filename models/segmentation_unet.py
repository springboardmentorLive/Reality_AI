"""
PyTorch U-Net Architecture for SpaceNet Satellite Building Footprint Segmentation
Implements:
- Contracting Encoder with residual/double-conv blocks
- Bottleneck feature representation
- Expansive Decoder with skip-connections
- Dice + BCE Loss for sharp geospatial boundary detection
- Zone analytics: Building footprint density %, Non-building open area %, Built-up classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        prev_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 classification conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class BCEDiceLoss(nn.Module):
    """Combination of Binary Cross Entropy and Dice Loss for segmentation."""
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, targets):
        bce_loss = self.bce(pred_logits, targets)
        
        preds = torch.sigmoid(pred_logits)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def calculate_segmentation_metrics(preds_prob, targets, threshold=0.5, smooth=1e-6):
    """
    Computes Intersection-over-Union (IoU), Dice Score, Precision, and Recall.
    preds_prob: Tensor of probabilities [0, 1]
    targets: Ground truth binary tensor {0, 1}
    """
    preds_bin = (preds_prob > threshold).float()
    targets_bin = (targets > threshold).float()

    tp = (preds_bin * targets_bin).sum().item()
    fp = (preds_bin * (1.0 - targets_bin)).sum().item()
    fn = ((1.0 - preds_bin) * targets_bin).sum().item()
    union = tp + fp + fn
    total = 2.0 * tp + fp + fn

    iou = (tp + smooth) / (union + smooth)
    dice = (2.0 * tp + smooth) / (total + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall)
    }


def analyze_satellite_zone(binary_mask_np):
    """
    Extracts urban land use statistics from segmented building footprint mask.
    binary_mask_np: 2D numpy array (256x256) where 1=building, 0=non-building
    """
    total_pixels = binary_mask_np.size
    building_pixels = np.sum(binary_mask_np > 0)
    density_pct = float((building_pixels / total_pixels) * 100)

    # Zoning classification based on building density
    if density_pct < 10.0:
        zone_type = "Low Density Residential / Rural"
        development_index = "Emerging"
    elif density_pct < 28.0:
        zone_type = "Medium Density Suburban Residential"
        development_index = "Developed Suburban"
    else:
        zone_type = "High Density Urban Commercial / Mixed"
        development_index = "High Density Core"

    return {
        "building_coverage_pct": round(density_pct, 2),
        "non_building_area_pct": round(100.0 - density_pct, 2),
        "open_space_pct": round(100.0 - density_pct, 2),  # Deprecated alias for non_building_area_pct
        "zone_classification": zone_type,
        "development_tier": development_index
    }
