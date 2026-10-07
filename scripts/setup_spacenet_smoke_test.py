"""
Sets up the SpaceNet Smoke-Test / CI dataset in data/raw/spacenet/smoke_test/.
Preserves the 20-chip sample explicitly designated for pipeline validation, CI, and development.
"""

import os
import shutil
import pandas as pd
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACENET_DIR = os.path.join(BASE_DIR, "data", "raw", "spacenet")
SMOKE_DIR = os.path.join(SPACENET_DIR, "smoke_test")
RAW_CHIPS_DIR = os.path.join(SMOKE_DIR, "raw_chips")
IMAGES_DIR = os.path.join(SMOKE_DIR, "images")
MASKS_DIR = os.path.join(SMOKE_DIR, "masks")
MANIFESTS_DIR = os.path.join(BASE_DIR, "data", "manifests")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def main():
    os.makedirs(RAW_CHIPS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    # Source directories from Phase 1
    src_raw_chips = os.path.join(SPACENET_DIR, "raw_chips")
    src_images = os.path.join(SPACENET_DIR, "images")
    src_masks = os.path.join(SPACENET_DIR, "masks")

    # Copy files into smoke_test
    if os.path.exists(src_raw_chips):
        for f in os.listdir(src_raw_chips):
            shutil.copy2(os.path.join(src_raw_chips, f), os.path.join(RAW_CHIPS_DIR, f))
    if os.path.exists(src_images):
        for f in os.listdir(src_images):
            if f.startswith("spacenet_vegas_"):
                shutil.copy2(os.path.join(src_images, f), os.path.join(IMAGES_DIR, f))
    if os.path.exists(src_masks):
        for f in os.listdir(src_masks):
            if f.startswith("spacenet_vegas_"):
                shutil.copy2(os.path.join(src_masks, f), os.path.join(MASKS_DIR, f))

    # Read original manifest and update paths for smoke_test
    orig_manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_manifest.csv")
    if os.path.exists(orig_manifest_p):
        df = pd.read_csv(orig_manifest_p)
        df["purpose"] = "PIPELINE VALIDATION / CI / DEVELOPMENT"
        df["image_path"] = df["image_path"].apply(lambda p: p.replace("data/raw/spacenet/images/", "data/raw/spacenet/smoke_test/images/"))
        df["mask_path"] = df["mask_path"].apply(lambda p: p.replace("data/raw/spacenet/masks/", "data/raw/spacenet/smoke_test/masks/"))
        df["raw_tif_path"] = df["raw_tif_path"].apply(lambda p: p.replace("data/raw/spacenet/raw_chips/", "data/raw/spacenet/smoke_test/raw_chips/"))
        df["raw_geojson_path"] = df["raw_geojson_path"].apply(lambda p: p.replace("data/raw/spacenet/raw_chips/", "data/raw/spacenet/smoke_test/raw_chips/"))

        smoke_manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_smoke_test_manifest.csv")
        df.to_csv(smoke_manifest_p, index=False)
        print(f"Saved smoke test manifest ({len(df)} records) to {smoke_manifest_p}")

        # Update smoke test splits JSON
        splits_json = {"train": [], "val": [], "test": []}
        for _, row in df.iterrows():
            split_name = row["source_split"]
            if split_name in splits_json:
                splits_json[split_name].append({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "num_buildings": int(row["num_buildings"]),
                    "building_pixel_pct": float(row["building_pixel_pct"])
                })
        smoke_splits_p = os.path.join(PROCESSED_DIR, "spacenet_smoke_test_splits.json")
        with open(smoke_splits_p, "w") as f:
            json.dump(splits_json, f, indent=2)
        print(f"Saved smoke test splits JSON to {smoke_splits_p}")

    print("Smoke-test dataset setup complete.")

if __name__ == "__main__":
    main()
