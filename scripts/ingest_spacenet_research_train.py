"""
Ingest Official SpaceNet 2 (AOI_2_Vegas) Research-Training Subset from AWS S3.

Downloads authentic pan-sharpened RGB GeoTIFF chips and vector building footprints
directly from the public SpaceNet AWS S3 bucket:
  s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/

Produces:
- Raw GeoTIFFs and GeoJSONs in data/raw/spacenet/research_train/raw_chips/
- 8-bit RGB PNGs in data/raw/spacenet/research_train/images/
- Binary building footprint masks in data/raw/spacenet/research_train/masks/
- Manifest in data/manifests/spacenet_train_manifest.csv
- Splits in data/processed/spacenet_train_splits.json (80% train, 20% validation)
"""

import os
import sys
import json
import hashlib
import urllib.request
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from PIL import Image
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACENET_DIR = os.path.join(BASE_DIR, "data", "raw", "spacenet", "research_train")
RAW_CHIPS_DIR = os.path.join(SPACENET_DIR, "raw_chips")
IMAGES_DIR = os.path.join(SPACENET_DIR, "images")
MASKS_DIR = os.path.join(SPACENET_DIR, "masks")
MANIFESTS_DIR = os.path.join(BASE_DIR, "data", "manifests")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

S3_BASE_URL = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train/AOI_2_Vegas"

# 30 verified authentic chips from official SpaceNet 2 Vegas training partition
TRAIN_CHIP_IDS = [
    "1002", "1003", "1004", "1006", "1007", "1009", "1010", "1015", "1016", "1017",
    "1018", "1021", "1023", "1025", "1028", "1029", "1030", "1031", "1032", "1034",
    "1035", "1037", "1038", "1039", "1041", "1042", "1047", "1048", "1049", "1051"
]

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "RealtyAI-Phase2-DataIngest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def filter_polygon_geometries(geometries):
    """Filter out degenerate Point or geometry collections, keeping only Polygons."""
    return [g for g in geometries if g.geom_type in ("Polygon", "MultiPolygon")]

def main():
    os.makedirs(RAW_CHIPS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Ingesting {len(TRAIN_CHIP_IDS)} official SpaceNet 2 Vegas research-training chips from AWS S3...")

    # Whole-chip train/val split (80% train = 24 chips, 20% val = 6 chips)
    # Fixed reproducible assignment to avoid spatial leakage
    val_chip_set = set(["1041", "1042", "1047", "1048", "1049", "1051"])

    manifest_records = []
    splits = {"train": [], "val": []}

    for idx, chip_id in enumerate(TRAIN_CHIP_IDS, 1):
        tif_filename = f"SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{chip_id}.tif"
        geojson_filename = f"SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{chip_id}.geojson"

        tif_url = f"{S3_BASE_URL}/PS-RGB/{tif_filename}"
        geojson_url = f"{S3_BASE_URL}/geojson_buildings/{geojson_filename}"

        raw_tif_path = os.path.join(RAW_CHIPS_DIR, f"img{chip_id}.tif")
        raw_geojson_path = os.path.join(RAW_CHIPS_DIR, f"img{chip_id}.geojson")

        print(f"[{idx}/{len(TRAIN_CHIP_IDS)}] Fetching chip {chip_id} from AWS S3...")
        ok_tif = download_file(tif_url, raw_tif_path)
        ok_geo = download_file(geojson_url, raw_geojson_path)

        if not ok_tif or not ok_geo:
            print(f"Skipping chip {chip_id} due to download failure.")
            continue

        # Process GeoTIFF
        with rasterio.open(raw_tif_path) as src:
            crs_wkt = src.crs.to_string() if src.crs else "NOT VERIFIED"
            transform = src.transform
            width = src.width
            height = src.height
            channels = src.count
            arr = src.read([1, 2, 3]).astype(np.float32)

        # Contrast stretch: 2nd to 98th percentile per chip
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr_norm = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0) * 255.0
        else:
            arr_norm = np.clip(arr, 0.0, 255.0)
        img_uint8 = arr_norm.astype(np.uint8)
        img_pil = Image.fromarray(img_uint8.transpose(1, 2, 0))

        # Save processed RGB PNG
        png_filename = f"spacenet_vegas_chip_{chip_id}.png"
        png_path = os.path.join(IMAGES_DIR, png_filename)
        img_pil.save(png_path)

        # Process GeoJSON vector building footprints
        with open(raw_geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        raw_geoms = [shape(feat["geometry"]) for feat in features if "geometry" in feat and feat["geometry"]]
        valid_geoms = filter_polygon_geometries(raw_geoms)

        # Rasterize building footprints onto pixel grid using exact affine transform
        if valid_geoms:
            mask_arr = rasterize(
                [(geom, 255) for geom in valid_geoms],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
        else:
            mask_arr = np.zeros((height, width), dtype=np.uint8)

        # Save binary mask PNG
        mask_filename = f"spacenet_vegas_chip_{chip_id}_mask.png"
        mask_path = os.path.join(MASKS_DIR, mask_filename)
        mask_pil = Image.fromarray(mask_arr, mode="L")
        mask_pil.save(mask_path)

        # Metrics
        building_pixels = int(np.sum(mask_arr > 0))
        total_pixels = height * width
        building_pct = round((building_pixels / total_pixels) * 100.0, 4)
        split_assignment = "val" if chip_id in val_chip_set else "train"

        rel_img_path = os.path.relpath(png_path, BASE_DIR).replace("\\", "/")
        rel_mask_path = os.path.relpath(mask_path, BASE_DIR).replace("\\", "/")
        rel_raw_tif = os.path.relpath(raw_tif_path, BASE_DIR).replace("\\", "/")
        rel_raw_geojson = os.path.relpath(raw_geojson_path, BASE_DIR).replace("\\", "/")

        record = {
            "image_id": f"vegas_{chip_id}",
            "image_path": rel_img_path,
            "mask_path": rel_mask_path,
            "raw_tif_path": rel_raw_tif,
            "raw_geojson_path": rel_raw_geojson,
            "aoi": "AOI_2_Vegas",
            "source_dataset": "SpaceNet 2 Building Detection (Vegas Train)",
            "source_bucket": "s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/",
            "purpose": "RESEARCH_TRAINING",
            "source_split": split_assignment,
            "width": width,
            "height": height,
            "channels": channels,
            "crs": crs_wkt,
            "transform_available": True,
            "num_buildings": len(valid_geoms),
            "building_pixel_pct": building_pct,
            "license": "CC BY-SA 4.0",
            "image_sha256": sha256_file(png_path),
            "mask_sha256": sha256_file(mask_path)
        }
        manifest_records.append(record)

        splits[split_assignment].append({
            "image_id": f"vegas_{chip_id}",
            "image_path": rel_img_path,
            "mask_path": rel_mask_path,
            "num_buildings": len(valid_geoms),
            "building_pixel_pct": building_pct
        })

    # Save manifest CSV
    manifest_csv = os.path.join(MANIFESTS_DIR, "spacenet_train_manifest.csv")
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Saved SpaceNet research-training manifest ({len(df_manifest)} records) to {manifest_csv}")

    # Save splits JSON
    splits_json_path = os.path.join(PROCESSED_DIR, "spacenet_train_splits.json")
    with open(splits_json_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved SpaceNet research-training splits JSON to {splits_json_path}")

    # Summary
    print("\n--- SpaceNet Research-Training Summary ---")
    print(f"Total Chips: {len(df_manifest)}")
    print(f"Train Chips: {len(splits['train'])}")
    print(f"Validation Chips: {len(splits['val'])}")
    print(f"Total Building Polygons: {df_manifest['num_buildings'].sum()}")
    print(f"Mean Building Coverage: {df_manifest['building_pixel_pct'].mean():.2f}%")

if __name__ == "__main__":
    main()
