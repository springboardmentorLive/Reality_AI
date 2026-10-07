"""
Organize PEER Hub ImageNet Task 5 into Official Source Splits.

Strictly follows Phase 2 Decision Rule B (Case 1):
- Official Test Set (146 images from task5_X_test.npy) is preserved exclusively in data/raw/property_conditions/test/
- Official Training Set (1,226 images from task5_X_train.npy) is partitioned into:
    * train: 1,042 images (85% stratified)
    * val: 184 images (15% stratified)
- ZERO test images are mixed into training or validation.
"""

import os
import sys
import json
import hashlib
import numpy as np
from PIL import Image
import pandas as pd
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROP_DIR = os.path.join(BASE_DIR, "data", "raw", "property_conditions")
RAW_PHI_DIR = os.path.join(PROP_DIR, "raw_phi_net")
MANIFESTS_DIR = os.path.join(BASE_DIR, "data", "manifests")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

CLASS_MAP = {
    0: ("global_collapse", "gc", "Global collapse"),
    1: ("non_collapse", "nc", "Non-collapse"),
    2: ("partial_collapse", "pc", "Partial collapse"),
}

# ImageNet Caffe BGR mean for zero-loss reconstitution
BGR_MEAN = np.array([103.939, 116.779, 123.680], dtype=np.float32)

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def reconstitute_rgb(img_array):
    img_bgr = img_array + BGR_MEAN
    img_rgb = np.clip(img_bgr[:, :, ::-1], 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img_rgb)

def main():
    train_x_path = os.path.join(RAW_PHI_DIR, "task5_X_train.npy")
    train_y_path = os.path.join(RAW_PHI_DIR, "task5_y_train.npy")
    test_x_path = os.path.join(RAW_PHI_DIR, "task5_X_test.npy")
    test_y_path = os.path.join(RAW_PHI_DIR, "task5_y_test.npy")

    if not os.path.exists(train_x_path) or not os.path.exists(train_y_path):
        raise FileNotFoundError(f"Training arrays not found in {RAW_PHI_DIR}")
    if not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
        raise FileNotFoundError(f"Test arrays not found in {RAW_PHI_DIR}")

    print("Loading PEER training and test numpy arrays...")
    x_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    x_test = np.load(test_x_path)
    y_test = np.load(test_y_path)

    print(f"Loaded official training set: {x_train.shape}, official test set: {x_test.shape}")

    # Remove old flat class directories from Phase 1 if present
    for cls_name, _, _ in CLASS_MAP.values():
        old_dir = os.path.join(PROP_DIR, cls_name)
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)

    # Create clean directory structure: train/, val/, test/
    for partition in ["train", "val", "test"]:
        for cls_name, _, _ in CLASS_MAP.values():
            os.makedirs(os.path.join(PROP_DIR, partition, cls_name), exist_ok=True)

    manifest_records = []
    splits = {"train": [], "val": [], "test": []}

    # 1. Process Official Test Set (146 images) -> strictly into test/
    test_classes = np.argmax(y_test, axis=1)
    for i in range(len(x_test)):
        cid = int(test_classes[i])
        cls_slug, cls_code, cls_desc = CLASS_MAP[cid]

        img_pil = reconstitute_rgb(x_test[i])
        filename = f"phi_net_test_{cls_code}_{i:04d}.jpg"
        save_path = os.path.join(PROP_DIR, "test", cls_slug, filename)
        img_pil.save(save_path, quality=95)

        rel_path = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")
        img_hash = sha256_file(save_path)

        rec = {
            "image_id": f"phi_net_test_{i:04d}",
            "image_path": rel_path,
            "label": cls_slug,
            "class_id": cid,
            "class_code": cls_code.upper(),
            "class_description": cls_desc,
            "source_partition": "OFFICIAL_BENCHMARK_TEST",
            "assigned_split": "test",
            "source_dataset": "PEER Hub ImageNet (PHI-Net) Task 5",
            "source_author": "Yuqing Gao (UC Berkeley PEER Center)",
            "license": "CC BY-NC-SA 4.0",
            "width": 224,
            "height": 224,
            "channels": 3,
            "sha256": img_hash
        }
        manifest_records.append(rec)
        splits["test"].append({
            "image_id": rec["image_id"],
            "image_path": rel_path,
            "label": cls_slug,
            "class_id": cid
        })

    # 2. Partition Official Training Set (1,226 images) into train (85%) and val (15%)
    train_classes = np.argmax(y_train, axis=1)
    rng = np.random.RandomState(42)

    train_indices = []
    val_indices = []

    for cid in sorted(CLASS_MAP.keys()):
        class_idxs = np.where(train_classes == cid)[0]
        rng.shuffle(class_idxs)
        n_c = len(class_idxs)
        n_val_c = max(1, int(round(n_c * 0.15)))
        val_indices.extend(class_idxs[:n_val_c])
        train_indices.extend(class_idxs[n_val_c:])

    train_set = set(train_indices)

    for i in range(len(x_train)):
        cid = int(train_classes[i])
        cls_slug, cls_code, cls_desc = CLASS_MAP[cid]
        split_name = "train" if i in train_set else "val"

        img_pil = reconstitute_rgb(x_train[i])
        filename = f"phi_net_{split_name}_{cls_code}_{i:04d}.jpg"
        save_path = os.path.join(PROP_DIR, split_name, cls_slug, filename)
        img_pil.save(save_path, quality=95)

        rel_path = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")
        img_hash = sha256_file(save_path)

        rec = {
            "image_id": f"phi_net_train_{i:04d}",
            "image_path": rel_path,
            "label": cls_slug,
            "class_id": cid,
            "class_code": cls_code.upper(),
            "class_description": cls_desc,
            "source_partition": "OFFICIAL_TRAIN",
            "assigned_split": split_name,
            "source_dataset": "PEER Hub ImageNet (PHI-Net) Task 5",
            "source_author": "Yuqing Gao (UC Berkeley PEER Center)",
            "license": "CC BY-NC-SA 4.0",
            "width": 224,
            "height": 224,
            "channels": 3,
            "sha256": img_hash
        }
        manifest_records.append(rec)
        splits[split_name].append({
            "image_id": rec["image_id"],
            "image_path": rel_path,
            "label": cls_slug,
            "class_id": cid
        })

    # Save manifest CSV
    manifest_csv = os.path.join(MANIFESTS_DIR, "property_condition_manifest.csv")
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.sort_values(by="image_id", inplace=True)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Saved complete property condition manifest ({len(df_manifest)} records) to {manifest_csv}")

    # Save splits JSON
    splits_json_path = os.path.join(PROCESSED_DIR, "property_condition_splits.json")
    with open(splits_json_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved complete property condition splits JSON to {splits_json_path}")

    # Class breakdown summary
    print("\n--- PEER Structural Condition Official Split Summary ---")
    print(f"Total Dataset Images: {len(df_manifest)}")
    print(f"Official Training Set (Split into Train {len(splits['train'])} + Val {len(splits['val'])}): {len(splits['train']) + len(splits['val'])}")
    print(f"Official Test Set (Preserved strictly as Test): {len(splits['test'])}")
    print("\nClass breakdown by partition:")
    print(pd.crosstab(df_manifest["assigned_split"], df_manifest["label"], margins=True))

if __name__ == "__main__":
    main()
