"""
Pipeline 03: Train SpaceNet Satellite Building Footprint Segmentation U-Net
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
import logging
import platform

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.segmentation_unet import UNet, BCEDiceLoss, calculate_segmentation_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
ERROR_ANALYSIS_DIR = os.path.join(SAVED_MODELS_DIR, "unet_error_analysis")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpaceNetDataset(Dataset):
    def __init__(self, sample_list, base_dir, augment=False):
        self.samples = sample_list
        self.base_dir = base_dir
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_p = os.path.join(self.base_dir, item["image_path"])
        mask_p = os.path.join(self.base_dir, item["mask_path"])

        img = Image.open(img_p).convert("RGB").resize((256, 256), resample=Image.Resampling.BILINEAR)
        mask = Image.open(mask_p).convert("L").resize((256, 256), resample=Image.Resampling.NEAREST)

        # To numpy
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
        mask_np = (np.array(mask, dtype=np.float32) > 128).astype(np.float32)  # (H, W)

        if self.augment:
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np).copy()
                mask_np = np.fliplr(mask_np).copy()
            if np.random.rand() > 0.5:
                img_np = np.flipud(img_np).copy()
                mask_np = np.flipud(mask_np).copy()

        # To tensor (C, H, W)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)).copy()).contiguous()
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0).contiguous()

        # ImageNet mean / std norm
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, mask_tensor, item["image_id"]


def save_error_analysis_quad(image_id, img_tensor, gt_mask_tensor, pred_mask_prob, save_dir):
    """
    Saves a 4-panel visual comparison:
    1. Input RGB (denormalized)
    2. Ground Truth Mask
    3. Predicted Probability Mask
    4. Segmentation Overlay (Cyan=Ground Truth, Orange=Prediction, Green=Overlap)
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    img_rgb = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img_rgb = np.clip((img_rgb * std + mean) * 255.0, 0, 255).astype(np.uint8)

    gt_mask = (gt_mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    pred_prob = (pred_mask_prob.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    pred_bin = (pred_mask_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Composite overlay
    overlay = img_rgb.copy()
    # Green for true positive overlap
    tp_mask = (gt_mask > 0) & (pred_bin > 0)
    # Cyan for false negative (missed building)
    fn_mask = (gt_mask > 0) & (pred_bin == 0)
    # Orange/Red for false positive (over-prediction)
    fp_mask = (gt_mask == 0) & (pred_bin > 0)

    overlay[tp_mask] = (overlay[tp_mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    overlay[fn_mask] = (overlay[fn_mask] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
    overlay[fp_mask] = (overlay[fp_mask] * 0.4 + np.array([255, 69, 0]) * 0.6).astype(np.uint8)

    panel_w, panel_h = 256, 256
    quad = Image.new("RGB", (panel_w * 4, panel_h))
    quad.paste(Image.fromarray(img_rgb), (0, 0))
    quad.paste(Image.fromarray(gt_mask).convert("RGB"), (panel_w, 0))
    quad.paste(Image.fromarray(pred_prob).convert("RGB"), (panel_w * 2, 0))
    quad.paste(Image.fromarray(overlay), (panel_w * 3, 0))

    out_path = os.path.join(save_dir, f"{image_id}_error_analysis.png")
    quad.save(out_path)
    return out_path


def train_segmentation(epochs=12, batch_size=4, lr=1e-3, seed=SEED):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training SpaceNet U-Net on device: {device} (Seed: {seed})")

    splits_file = os.path.join(PROCESSED_DIR, "spacenet_train_splits.json")
    if not os.path.exists(splits_file):
        splits_file = os.path.join(PROCESSED_DIR, "spacenet_splits.json")
    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_ds = SpaceNetDataset(splits["train"], BASE_DIR, augment=True)
    val_ds = SpaceNetDataset(splits["val"], BASE_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    logger.info(f"Loaded SpaceNet dataset: {len(train_ds)} train chips, {len(val_ds)} validation chips.")

    # Lightweight features for fast CPU convergence
    features = [16, 32, 64, 128]
    model = UNet(in_channels=3, out_channels=1, features=features).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, smooth=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    primary_model_path = os.path.join(SAVED_MODELS_DIR, "unet_spacenet_v1.pt")
    alias_model_path = os.path.join(SAVED_MODELS_DIR, "unet_satellite.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_ious, val_dices = [], []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                m = calculate_segmentation_metrics(probs, masks, threshold=0.5)
                val_ious.append(m["iou"])
                val_dices.append(m["dice"])

        val_loss /= len(val_loader.dataset)
        avg_iou = float(np.mean(val_ious))
        avg_dice = float(np.mean(val_dices))

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_iou"].append(round(avg_iou, 4))
        history["val_dice"].append(round(avg_dice, 4))

        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {avg_iou:.4f} | Val Dice: {avg_dice:.4f}")

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), primary_model_path)
            torch.save(model.state_dict(), alias_model_path)
            logger.info(f"  -> Saved new best model checkpoint (Val Dice: {best_val_dice:.4f})")

    # Final detailed evaluation on held-out validation chips
    logger.info("Running final detailed evaluation on held-out validation chips...")
    model.load_state_dict(torch.load(primary_model_path, weights_only=True))
    model.eval()

    per_chip_metrics = []
    saved_analysis_files = []

    with torch.no_grad():
        for images, masks, chip_ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            chip_id = chip_ids[0]

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            m = calculate_segmentation_metrics(probs, masks, threshold=0.5)

            rec = {
                "chip_id": chip_id,
                "iou": round(m["iou"], 4),
                "dice": round(m["dice"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4)
            }
            per_chip_metrics.append(rec)

            # Save visual error analysis quad
            quad_p = save_error_analysis_quad(chip_id, images[0], masks[0], probs[0], ERROR_ANALYSIS_DIR)
            saved_analysis_files.append(os.path.relpath(quad_p, BASE_DIR).replace("\\", "/"))

    val_ious = [r["iou"] for r in per_chip_metrics]
    val_dices = [r["dice"] for r in per_chip_metrics]
    val_precs = [r["precision"] for r in per_chip_metrics]
    val_recs = [r["recall"] for r in per_chip_metrics]

    final_metrics = {
        "model_name": "SpaceNet 2 Building Footprint U-Net",
        "version": "v1.0-real",
        "task": "Building footprint binary segmentation",
        "architecture": "PyTorch U-Net (DoubleConv + Skip Connections)",
        "features": features,
        "loss_function": "BCEDiceLoss (50% BCE + 50% Dice, smooth=1.0)",
        "optimizer": "AdamW (lr=0.001, weight_decay=1e-4, CosineAnnealingLR)",
        "seed": seed,
        "evaluation_partition": "validation_heldout",
        "official_public_test_status": "Development evaluation performed on held-out validation chips; official public test labels unavailable locally.",
        "threshold": 0.5,
        "num_evaluated_chips": len(per_chip_metrics),
        "aggregate_metrics": {
            "mean_iou": round(float(np.mean(val_ious)), 4),
            "median_iou": round(float(np.median(val_ious)), 4),
            "std_iou": round(float(np.std(val_ious)), 4),
            "mean_dice": round(float(np.mean(val_dices)), 4),
            "median_dice": round(float(np.median(val_dices)), 4),
            "std_dice": round(float(np.std(val_dices)), 4),
            "mean_precision": round(float(np.mean(val_precs)), 4),
            "mean_recall": round(float(np.mean(val_recs)), 4)
        },
        "per_chip_distribution": per_chip_metrics,
        "error_analysis_panels": saved_analysis_files,
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "os": platform.platform()
        },
        "checkpoint_path": "models/saved/unet_spacenet_v1.pt",
        "dataset_manifest": "data/manifests/spacenet_train_manifest.csv"
    }

    with open(os.path.join(SAVED_MODELS_DIR, "unet_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"U-Net Training & Evaluation Complete.")
    logger.info(f"Aggregate Validation Metrics (N={len(per_chip_metrics)} chips): IoU = {final_metrics['aggregate_metrics']['mean_iou']:.4f}, Dice = {final_metrics['aggregate_metrics']['mean_dice']:.4f}, Precision = {final_metrics['aggregate_metrics']['mean_precision']:.4f}, Recall = {final_metrics['aggregate_metrics']['mean_recall']:.4f}")
    return final_metrics


if __name__ == "__main__":
    train_segmentation(epochs=12, batch_size=4, lr=1e-3, seed=42)
