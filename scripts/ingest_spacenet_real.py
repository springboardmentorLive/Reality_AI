"""
Ingest and Process Real SpaceNet 2 (AOI_2_Vegas) Satellite Data.

Downloads authentic Maxar/DigitalGlobe WorldView-3 satellite chips and building
footprint GeoJSON annotations, rasterizes the building polygons to binary masks,
and builds reproducible manifests and split files.
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACENET_DIR = os.path.join(BASE_DIR, "data", "raw", "spacenet")
RAW_CHIPS_DIR = os.path.join(SPACENET_DIR, "raw_chips")
IMAGES_DIR = os.path.join(SPACENET_DIR, "images")
MASKS_DIR = os.path.join(SPACENET_DIR, "masks")
MANIFESTS_DIR = os.path.join(BASE_DIR, "data", "manifests")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

HF_BASE_URL = "https://huggingface.co/datasets/khalilurrahmanridoykhan/spacenet-buildings-vegas-smoketest-sample/resolve/main"

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "RealtyAI-DataMigration/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())

def filter_polygon_geometries(geometries):
    """Filter out degenerate Point or geometry collections, keeping only Polygons."""
    return [g for g in geometries if g.geom_type in ("Polygon", "MultiPolygon")]

def main():
    os.makedirs(RAW_CHIPS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Fetching SpaceNet Vegas split.json...")
    split_url = f"{HF_BASE_URL}/split.json"
    split_dest = os.path.join(RAW_CHIPS_DIR, "split.json")
    download_file(split_url, split_dest)

    with open(split_dest, "r") as f:
        split_data = json.load(f)

    all_ids = []
    id_to_split = {}
    for split_name, ids in split_data.items():
        for chip_id in ids:
            all_ids.append(chip_id)
            id_to_split[chip_id] = split_name

    print(f"Found {len(all_ids)} chips across splits: "
          f"train={len(split_data['train'])}, val={len(split_data['val'])}, test={len(split_data['test'])}")

    manifest_records = []
    processed_splits = {"train": [], "val": [], "test": []}

    for idx, chip_id in enumerate(all_ids, 1):
        tif_filename = f"img{chip_id}.tif"
        geojson_filename = f"img{chip_id}.geojson"

        tif_url = f"{HF_BASE_URL}/raw/images/{tif_filename}"
        geojson_url = f"{HF_BASE_URL}/raw/labels/{geojson_filename}"

        raw_tif_path = os.path.join(RAW_CHIPS_DIR, tif_filename)
        raw_geojson_path = os.path.join(RAW_CHIPS_DIR, geojson_filename)

        print(f"[{idx}/{len(all_ids)}] Downloading chip {chip_id}...")
        download_file(tif_url, raw_tif_path)
        download_file(geojson_url, raw_geojson_path)

        # Process GeoTIFF
        with rasterio.open(raw_tif_path) as src:
            profile = src.profile
            crs_wkt = src.crs.to_string() if src.crs else "NOT VERIFIED"
            transform = src.transform
            width = src.width
            height = src.height
            channels = src.count
            arr = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)

        # Contrast stretch: 2nd to 98th percentile per chip
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr_norm = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0) * 255.0
        else:
            arr_norm = np.clip(arr, 0.0, 255.0)
        img_uint8 = arr_norm.astype(np.uint8)  # (3, H, W)
        img_pil = Image.fromarray(img_uint8.transpose(1, 2, 0))

        # Save processed RGB PNG
        png_filename = f"spacenet_vegas_chip_{chip_id}.png"
        png_path = os.path.join(IMAGES_DIR, png_filename)
        img_pil.save(png_path)

        # Process GeoJSON labels
        with open(raw_geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        raw_geoms = [shape(feat["geometry"]) for feat in features if "geometry" in feat and feat["geometry"]]
        valid_geoms = filter_polygon_geometries(raw_geoms)

        # Rasterize building footprints
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

        # Save mask PNG
        mask_filename = f"spacenet_vegas_chip_{chip_id}_mask.png"
        mask_path = os.path.join(MASKS_DIR, mask_filename)
        mask_pil = Image.fromarray(mask_arr, mode="L")
        mask_pil.save(mask_path)

        # Calculate statistics
        building_pixels = int(np.sum(mask_arr > 0))
        total_pixels = height * width
        building_pct = round((building_pixels / total_pixels) * 100.0, 4)

        rel_img_path = os.path.relpath(png_path, BASE_DIR).replace("\\", "/")
        rel_mask_path = os.path.relpath(mask_path, BASE_DIR).replace("\\", "/")
        rel_raw_tif = os.path.relpath(raw_tif_path, BASE_DIR).replace("\\", "/")
        rel_raw_geojson = os.path.relpath(raw_geojson_path, BASE_DIR).replace("\\", "/")

        split_assignment = id_to_split[chip_id]

        record = {
            "image_id": f"vegas_{chip_id}",
            "image_path": rel_img_path,
            "mask_path": rel_mask_path,
            "raw_tif_path": rel_raw_tif,
            "raw_geojson_path": rel_raw_geojson,
            "aoi": "AOI_2_Vegas",
            "source_dataset": "SpaceNet 2 Building Footprints (Vegas)",
            "source_split": split_assignment,
            "width": width,
            "height": height,
            "channels": channels,
            "crs": crs_wkt,
            "transform_available": True,
            "num_buildings": len(valid_geoms),
            "building_pixel_pct": building_pct,
            "label_source": "SpaceNet / CosmiQ Works Building Footprint GeoJSON",
            "license": "CC BY-SA 4.0",
            "image_sha256": sha256_file(png_path),
            "mask_sha256": sha256_file(mask_path),
        }
        manifest_records.append(record)

        processed_splits[split_assignment].append({
            "image_id": f"vegas_{chip_id}",
            "image_path": rel_img_path,
            "mask_path": rel_mask_path,
            "num_buildings": len(valid_geoms),
            "building_pixel_pct": building_pct,
        })

    # Save manifest CSV
    manifest_csv = os.path.join(MANIFESTS_DIR, "spacenet_manifest.csv")
    import pandas as pd
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Saved SpaceNet manifest ({len(df_manifest)} records) to {manifest_csv}")

    # Save split JSON
    splits_json_path = os.path.join(PROCESSED_DIR, "spacenet_splits.json")
    with open(splits_json_path, "w") as f:
        json.dump(processed_splits, f, indent=2)
    print(f"Saved SpaceNet splits JSON to {splits_json_path}")

    # Summary validation
    print("\n--- Validation Summary ---")
    print(f"Total chips: {len(df_manifest)}")
    print(f"Train chips: {len(processed_splits['train'])}")
    print(f"Val chips: {len(processed_splits['val'])}")
    print(f"Test chips: {len(processed_splits['test'])}")
    print(f"Total buildings across dataset: {df_manifest['num_buildings'].sum()}")
    print(f"Mean building pixel coverage: {df_manifest['building_pixel_pct'].mean():.2f}%")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
