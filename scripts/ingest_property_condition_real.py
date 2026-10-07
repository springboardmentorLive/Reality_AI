"""
Ingest and Process Real Building Structural Condition Data (Collapse Mode).

Source: PEER Hub ImageNet (PHI-Net) - Task 5 (Collapse Mode)
Authors: Yuqing Gao (UC Berkeley PEER Center, gaoyuqing@berkeley.edu)
License: CC BY-NC-SA 4.0
Ground-truth Classes:
  0: Global collapse (GC)
  1: Non-collapse (NC)
  2: Partial collapse (PC)
"""

import os
import sys
import json
import hashlib
import io
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROP_COND_DIR = os.path.join(BASE_DIR, "data", "raw", "property_conditions")
RAW_PHI_DIR = os.path.join(PROP_COND_DIR, "raw_phi_net")
MANIFESTS_DIR = os.path.join(BASE_DIR, "data", "manifests")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

HF_DATA_ZIP_URL = "https://huggingface.co/datasets/kks32/building-damage/resolve/main/data.zip"

CLASS_MAP = {
    0: ("global_collapse", "GC", "Global collapse"),
    1: ("non_collapse", "NC", "Non-collapse"),
    2: ("partial_collapse", "PC", "Partial collapse"),
}

# ImageNet Caffe BGR mean for reconstitution
BGR_MEAN = np.array([103.939, 116.779, 123.680], dtype=np.float32)

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self._len = int(r.headers["Content-Length"])
        self._pos = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self._pos = offset
        elif whence == io.SEEK_CUR: self._pos += offset
        elif whence == io.SEEK_END: self._pos = self._len + offset
        return self._pos

    def readinto(self, b):
        if self._pos >= self._len: return 0
        end = min(self._pos + len(b) - 1, self._len - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            chunk = r.read()
        b[:len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    os.makedirs(RAW_PHI_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for class_id, (class_slug, _, _) in CLASS_MAP.items():
        os.makedirs(os.path.join(PROP_COND_DIR, class_slug), exist_ok=True)

    print("Connecting to PEER dataset via HTTP Range...")
    f = HTTPRangeFile(HF_DATA_ZIP_URL)
    zf = zipfile.ZipFile(f)

    # Save license and README
    print("Extracting license and README...")
    license_bytes = zf.read("task5/license.txt")
    readme_bytes = zf.read("task5/README.txt")

    with open(os.path.join(RAW_PHI_DIR, "license.txt"), "wb") as lf:
        lf.write(license_bytes)
    with open(os.path.join(RAW_PHI_DIR, "README.txt"), "wb") as rf:
        rf.write(readme_bytes)

    # Read X_test and y_test
    print("Downloading task5_X_test.npy (24.6 MB)...")
    raw_x_test = zf.read("task5/task5_X_test.npy")
    x_test = np.load(io.BytesIO(raw_x_test))

    print("Downloading task5_y_test.npy...")
    raw_y_test = zf.read("task5/task5_y_test.npy")
    y_test = np.load(io.BytesIO(raw_y_test))

    # Save raw arrays
    np.save(os.path.join(RAW_PHI_DIR, "task5_X_test.npy"), x_test)
    np.save(os.path.join(RAW_PHI_DIR, "task5_y_test.npy"), y_test)

    num_samples = len(x_test)
    print(f"Loaded {num_samples} real structural condition images.")

    # Convert one-hot to class integer
    # y is shape (N, 3): [1, 0, 0] -> 0, [0, 1, 0] -> 1, [0, 0, 1] -> 2
    class_indices = np.argmax(y_test, axis=1)

    # Stratified split: 70% train (102), 15% val (22), 15% test (22)
    # Reproducible seed
    rng = np.random.RandomState(42)
    splits = {"train": [], "val": [], "test": []}

    manifest_records = []

    # Assign splits per class to preserve stratification
    for cid in sorted(CLASS_MAP.keys()):
        idx_for_class = np.where(class_indices == cid)[0]
        rng.shuffle(idx_for_class)
        n_class = len(idx_for_class)

        n_val = max(1, int(round(n_class * 0.15)))
        n_test = max(1, int(round(n_class * 0.15)))
        n_train = n_class - n_val - n_test

        train_indices = idx_for_class[:n_train]
        val_indices = idx_for_class[n_train:n_train + n_val]
        test_indices = idx_for_class[n_train + n_val:]

        split_dict = {}
        for i in train_indices: split_dict[i] = "train"
        for i in val_indices: split_dict[i] = "val"
        for i in test_indices: split_dict[i] = "test"

        for idx in idx_for_class:
            split_name = split_dict[idx]
            class_slug, class_code, class_name = CLASS_MAP[cid]

            # Reconstruct RGB: add BGR mean, reverse BGR to RGB, clip 0-255
            img_bgr = x_test[idx] + BGR_MEAN
            img_rgb = np.clip(img_bgr[:, :, ::-1], 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_rgb)

            filename = f"phi_net_task5_{class_code.lower()}_{idx:04d}.jpg"
            img_path = os.path.join(PROP_COND_DIR, class_slug, filename)
            img_pil.save(img_path, quality=95)

            rel_path = os.path.relpath(img_path, BASE_DIR).replace("\\", "/")
            img_hash = sha256_file(img_path)

            record = {
                "image_id": f"phi_net_{idx:04d}",
                "image_path": rel_path,
                "label": class_slug,
                "class_id": cid,
                "class_code": class_code,
                "class_description": class_name,
                "split": split_name,
                "source_dataset": "PEER Hub ImageNet (PHI-Net) Task 5",
                "source_author": "Yuqing Gao (UC Berkeley PEER Center)",
                "license": "CC BY-NC-SA 4.0",
                "width": 224,
                "height": 224,
                "channels": 3,
                "sha256": img_hash,
            }
            manifest_records.append(record)

            splits[split_name].append({
                "image_id": f"phi_net_{idx:04d}",
                "image_path": rel_path,
                "label": class_slug,
                "class_id": int(cid),
            })

    # Save manifest CSV
    manifest_csv = os.path.join(MANIFESTS_DIR, "property_condition_manifest.csv")
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.sort_values(by="image_id", inplace=True)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Saved property condition manifest ({len(df_manifest)} records) to {manifest_csv}")

    # Save splits JSON
    splits_json_path = os.path.join(PROCESSED_DIR, "property_condition_splits.json")
    with open(splits_json_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved property condition splits JSON to {splits_json_path}")

    # Summary
    print("\n--- Property Condition Summary ---")
    print(f"Total images: {len(df_manifest)}")
    print(f"Train images: {len(splits['train'])}")
    print(f"Val images: {len(splits['val'])}")
    print(f"Test images: {len(splits['test'])}")
    print("\nClass breakdown:")
    print(df_manifest["label"].value_counts())
    print("\nClass breakdown per split:")
    print(pd.crosstab(df_manifest["split"], df_manifest["label"]))

if __name__ == "__main__":
    main()
