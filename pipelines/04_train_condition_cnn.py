"""
Pipeline 04: Train PEER Structural Condition CNN (ResNet-18 Transfer Learning)
Phase 3: Real Model Training + Leakage-Safe Evaluation + Reproducible Metrics
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import platform

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.condition_resnet import PropertyConditionClassifier, get_condition_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

SEED = 42
CLASS_MAP = {
    0: ("global_collapse", "Global Collapse (Severe Damage)"),
    1: ("non_collapse", "Non-Collapse (Sound Structure)"),
    2: ("partial_collapse", "Partial Collapse (Moderate Damage)")
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PropertyImageDataset(Dataset):
    def __init__(self, records, base_dir, transform=None):
        self.records = records
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        img_p = item.get("image_path", item.get("filepath"))
        if not os.path.isabs(img_p):
            img_p = os.path.join(self.base_dir, img_p)

        img = Image.open(img_p).convert("RGB")
        label = int(item.get("class_id", item.get("label")))

        if self.transform:
            img = self.transform(img)

        return img, label, item.get("image_id", str(idx))


def compute_class_weights(train_records):
    """Computes inverse-frequency class weights strictly on the training partition."""
    labels = [int(r.get("class_id", r.get("label"))) for r in train_records]
    n_samples = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    weights = np.zeros(len(classes), dtype=np.float32)
    for c, count in zip(classes, counts):
        weights[c] = n_samples / (len(classes) * count)
    return torch.tensor(weights, dtype=torch.float32), dict(zip(classes.tolist(), counts.tolist()))


def train_condition_model(epochs=6, batch_size=32, lr=3e-4, seed=SEED):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training PEER Structural Condition ResNet-18 on device: {device} (Seed: {seed})")

    splits_path = os.path.join(PROCESSED_DIR, "property_condition_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    # Compute class weights strictly on training partition
    class_weights, train_class_counts = compute_class_weights(splits["train"])
    logger.info(f"Training class counts: {train_class_counts}")
    logger.info(f"Computed loss class weights (train-only): {class_weights.tolist()}")

    train_ds = PropertyImageDataset(splits["train"], BASE_DIR, transform=get_condition_transforms(is_train=True))
    val_ds = PropertyImageDataset(splits["val"], BASE_DIR, transform=get_condition_transforms(is_train=False))
    test_ds = PropertyImageDataset(splits["test"], BASE_DIR, transform=get_condition_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Loaded PEER dataset: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} benchmark test images.")

    # ResNet-18 with ImageNet Pretrained Backbone
    model = PropertyConditionClassifier(num_classes=3, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_macro_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": []}

    primary_model_path = os.path.join(SAVED_MODELS_DIR, "resnet_peer_collapse_v1.pt")
    alias_model_path = os.path.join(SAVED_MODELS_DIR, "resnet_condition.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation evaluation for model checkpoint selection
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = float(accuracy_score(val_targets, val_preds))
        _, _, val_f1_macro, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        val_f1_macro = float(val_f1_macro)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1_macro, 4))

        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1_macro:.4f}")

        if val_f1_macro > best_val_macro_f1:
            best_val_macro_f1 = val_f1_macro
            torch.save(model.state_dict(), primary_model_path)
            torch.save(model.state_dict(), alias_model_path)
            logger.info(f"  -> Saved new best model checkpoint (Val Macro F1: {best_val_macro_f1:.4f})")

    # Final Single Evaluation on Untouched Official Benchmark Test Set
    logger.info("Executing final single evaluation on untouched official benchmark test set (N=146)...")
    model.load_state_dict(torch.load(primary_model_path, weights_only=True))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())

    test_acc = float(accuracy_score(test_targets, test_preds))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="weighted", zero_division=0)
    per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds).tolist()

    class_breakdown = {}
    for cid in sorted(CLASS_MAP.keys()):
        cslug, cdesc = CLASS_MAP[cid]
        class_breakdown[cslug] = {
            "class_id": cid,
            "description": cdesc,
            "precision": round(float(per_class_prec[cid]), 4),
            "recall": round(float(per_class_rec[cid]), 4),
            "f1_score": round(float(per_class_f1[cid]), 4),
            "support": int(per_class_support[cid])
        }

    final_metrics = {
        "model_name": "PEER PHI-Net ResNet-18 Collapse Mode Classifier",
        "version": "v1.0-real",
        "task": "Structural Condition & Integrity Assessment — Collapse Mode",
        "classes": ["global_collapse", "non_collapse", "partial_collapse"],
        "architecture": "ResNet-18 (ImageNet Pretrained Backbone + Dropout MLP Head)",
        "pretrained_weights": "ResNet18_Weights.DEFAULT (ImageNet-1K)",
        "loss_function": "Weighted CrossEntropyLoss (Class Weights calculated on train split only)",
        "train_class_weights": {c: round(float(w), 4) for c, w in zip(CLASS_MAP.keys(), class_weights.tolist())},
        "optimizer": "AdamW (lr=0.0003, weight_decay=1e-4, CosineAnnealingLR)",
        "seed": seed,
        "sample_counts": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "benchmark_test": len(test_ds)
        },
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "benchmark_test_evaluation": {
            "accuracy": round(test_acc, 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_prec), 4),
            "weighted_recall": round(float(weighted_rec), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["global_collapse", "non_collapse", "partial_collapse"],
            "per_class_breakdown": class_breakdown
        },
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "os": platform.platform()
        },
        "checkpoint_path": "models/saved/resnet_peer_collapse_v1.pt",
        "dataset_manifest": "data/manifests/property_condition_manifest.csv"
    }

    with open(os.path.join(SAVED_MODELS_DIR, "condition_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("ResNet-18 Training & Benchmark Test Complete.")
    logger.info(f"Official Benchmark Test Results (N={len(test_ds)}): Accuracy={test_acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}")
    return final_metrics


if __name__ == "__main__":
    train_condition_model(epochs=6, batch_size=32, lr=3e-4, seed=42)
