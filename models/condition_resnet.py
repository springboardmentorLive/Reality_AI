"""
PyTorch ResNet Classifier for Property Condition Assessment
Classifies real estate photos into:
- 0: New / Renovated (modern siding, intact clean roof, spotless windows, pristine landscape)
- 1: Moderate (standard condition, minor cosmetic wear, functional structural elements)
- 2: Old / Needs Work (weathered facade, roof moss/wear, peeling paint, renovation needed)

Provides:
- Condition Score (0 to 100 rating)
- Estimated Renovation Cost Multiplier / Impact on Valuation
- Inference pipeline for uploaded inspection photos
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np


class PropertyConditionClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Use resnet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features

        # Custom MLP classifier head with dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

        self.class_names = ["global_collapse", "non_collapse", "partial_collapse"]

    def forward(self, x):
        return self.backbone(x)


ResNetConditionClassifier = PropertyConditionClassifier



def get_condition_transforms(is_train=False):
    """Standard ImageNet transforms with augmentation for training."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def evaluate_property_inspection(probs_dict):
    """
    Computes structural collapse status and integrity assessment based on PEER Task 5 collapse-mode probabilities.
    probs_dict maps class names ("non_collapse", "partial_collapse", "global_collapse") to probabilities.
    """
    p_nc = probs_dict.get("non_collapse", probs_dict.get("Non-collapse", probs_dict.get("New / Renovated", 0.0)))
    p_pc = probs_dict.get("partial_collapse", probs_dict.get("Partial collapse", probs_dict.get("Moderate", 0.0)))
    p_gc = probs_dict.get("global_collapse", probs_dict.get("Global collapse", probs_dict.get("Old / Needs Work", 0.0)))

    # Structural integrity score: 0 (total global collapse) to 100 (intact non-collapse)
    condition_score = round(p_nc * 95.0 + p_pc * 50.0 + p_gc * 5.0, 1)

    if p_nc >= p_pc and p_nc >= p_gc:
        predicted_collapse_state = "Non-collapse"
        structural_status = "Non-Collapse (Intact Structural Frame)"
        valuation_impact = "Structural Baseline (Preserved Structural Premium / No Catastrophic Loss)"
        engineering_advice = "Structure demonstrates intact structural load paths without visible framing collapse in reconnaissance view."
        hazard_level = "Low Post-Disaster Structural Risk"
    elif p_pc >= p_gc:
        predicted_collapse_state = "Partial collapse"
        structural_status = "Partial Structural Collapse (Compromised Framing)"
        valuation_impact = "Compromised Framing (Structural Renovation / Shoring Allowance Required)"
        engineering_advice = "Evidence of localized structural failure, wall shear, or compromised supports. Immediate forensic structural engineering inspection required."
        hazard_level = "Elevated Hazard (Restricted Entry)"
    else:
        predicted_collapse_state = "Global collapse"
        structural_status = "Global Structural Collapse (Total Failure)"
        valuation_impact = "Catastrophic Failure (Total Loss / Structural Reconstruction & Renovation Allowance Required)"
        engineering_advice = "Catastrophic collapse of structural framing observed. Severe life-safety hazard; structure condemned / total loss."
        hazard_level = "Extreme Life-Safety Hazard (Do Not Enter)"

    return {
        "condition_score": condition_score,
        "predicted_collapse_state": predicted_collapse_state,
        "structural_status": structural_status,
        "valuation_impact": valuation_impact,
        "hazard_level": hazard_level,
        "inspection_advice": engineering_advice,
        "domain_disclaimer": "PEER PHI-Net Task 5 post-disaster collapse classification. Evaluates extreme structural failure modes, not cosmetic wear or routine curb appeal."
    }
