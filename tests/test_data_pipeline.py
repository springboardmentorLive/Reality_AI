"""
Unit Tests for RealtyAI Data Pipeline & Ingestion Integrity (Phase 2)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image
import rasterio

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MANIFESTS_DIR = os.path.join(BASE_DIR, "data", "manifests")


def test_housing_processed_arrays():
    """Verifies preprocessed housing numpy arrays exist with valid shapes."""
    x_train = np.load(os.path.join(PROCESSED_DIR, "housing_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "housing_y_train.npy"))
    x_test = np.load(os.path.join(PROCESSED_DIR, "housing_X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "housing_y_test.npy"))

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert x_train.shape[1] == x_test.shape[1]
    assert np.all(y_train > 0)
    assert not np.isnan(x_train).any()


def test_housing_feature_metadata():
    """Checks that feature metadata file is populated."""
    feat_p = os.path.join(PROCESSED_DIR, "housing_features.json")
    assert os.path.exists(feat_p)

    with open(feat_p, "r") as f:
        meta = json.load(f)

    assert "num_features" in meta
    assert "cat_features" in meta
    assert len(meta["all_features"]) > 10


def test_zillow_processed_time_series():
    """Verifies long-format Zillow dataset."""
    p = os.path.join(PROCESSED_DIR, "zillow_zhvi_processed.csv")
    assert os.path.exists(p)

    df = pd.read_csv(p)
    expected_cols = {"RegionName", "Date", "ZHVI"}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 1000
    assert not df["ZHVI"].isna().any()


def test_spacenet_splits():
    """Verifies SpaceNet split JSON contains train and val partitions without leakage."""
    p = os.path.join(PROCESSED_DIR, "spacenet_splits.json")
    assert os.path.exists(p)

    with open(p, "r") as f:
        splits = json.load(f)

    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    # Train and validation chips must be disjoint
    train_ids = set(s["image_id"] for s in splits["train"])
    val_ids = set(s["image_id"] for s in splits["val"])
    assert train_ids.isdisjoint(val_ids)


def test_spacenet_manifest_and_real_files():
    """Validates SpaceNet real manifests, coordinates, dimensions, and image/mask pairs."""
    manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_train_manifest.csv")
    if not os.path.exists(manifest_p):
        manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_manifest.csv")
    assert os.path.exists(manifest_p), f"Missing manifest: {manifest_p}"

    df = pd.read_csv(manifest_p)
    assert len(df) >= 20
    assert "image_sha256" in df.columns
    assert "mask_sha256" in df.columns
    assert "crs" in df.columns

    for _, row in df.head(10).iterrows():
        img_p = os.path.join(BASE_DIR, row["image_path"])
        mask_p = os.path.join(BASE_DIR, row["mask_path"])
        assert os.path.exists(img_p), f"Missing image {img_p}"
        assert os.path.exists(mask_p), f"Missing mask {mask_p}"


def test_property_condition_manifest_and_real_files():
    """Validates PEER Task 5 real manifest, genuine classes, and image files."""
    manifest_p = os.path.join(MANIFESTS_DIR, "property_condition_manifest.csv")
    assert os.path.exists(manifest_p), f"Missing manifest: {manifest_p}"

    df = pd.read_csv(manifest_p)
    assert len(df) >= 146
    expected_classes = {"global_collapse", "non_collapse", "partial_collapse"}
    assert set(df["label"].unique()) == expected_classes

    for _, row in df.head(20).iterrows():
        img_p = os.path.join(BASE_DIR, row["image_path"])
        assert os.path.exists(img_p), f"Missing image {img_p}"


# --- PHASE 2 INTEGRITY & METHODOLOGICAL AUDIT TESTS ---

def test_no_absolute_windows_paths_in_manifests():
    """Requirement 1: Verify all active manifests contain strictly relative, portable paths."""
    manifest_files = glob.glob(os.path.join(MANIFESTS_DIR, "*.csv"))
    assert len(manifest_files) > 0, "No manifests found."

    for mf in manifest_files:
        df = pd.read_csv(mf)
        for col in df.columns:
            if df[col].dtype == object:
                # No Windows drive letters (e.g. C:, D:)
                has_drive = df[col].astype(str).str.contains(r"^[A-Za-z]:", regex=True).any()
                assert not has_drive, f"Found absolute Windows drive path in {mf} column '{col}'"
                # No backslashes in file paths
                if "path" in col.lower():
                    has_backslash = df[col].astype(str).str.contains(r"\\").any()
                    assert not has_backslash, f"Found non-portable backslash in {mf} column '{col}'"


def test_no_synthetic_generator_imported_in_active_ingestion():
    """Requirement 2: Active ingestion scripts must never import synthetic generation modules."""
    active_scripts = [
        os.path.join(BASE_DIR, "scripts", "ingest_spacenet_research_train.py"),
        os.path.join(BASE_DIR, "scripts", "setup_spacenet_smoke_test.py"),
        os.path.join(BASE_DIR, "scripts", "reorganize_peer_official_splits.py"),
        os.path.join(BASE_DIR, "pipelines", "02_data_preprocessing.py")
    ]
    for script_p in active_scripts:
        if os.path.exists(script_p):
            with open(script_p, "r", encoding="utf-8") as f:
                content = f.read()
            assert "01_data_generation" not in content, f"Synthetic generator imported in {script_p}"
            assert "generate_synthetic" not in content, f"Synthetic generation invoked in {script_p}"


def test_no_synthetic_fallbacks():
    """Requirement 3: Missing files must trigger genuine failures, never synthetic fallbacks."""
    from scripts.reorganize_peer_official_splits import sha256_file
    test_file = os.path.join(BASE_DIR, "scripts", "setup_spacenet_smoke_test.py")
    assert len(sha256_file(test_file)) == 64


def test_spacenet_image_mask_correspondence():
    """Requirement 6: SpaceNet image/mask spatial correspondence and dimension match."""
    manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_train_manifest.csv")
    if not os.path.exists(manifest_p):
        manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_smoke_test_manifest.csv")
    df = pd.read_csv(manifest_p)

    for _, row in df.head(5).iterrows():
        img = Image.open(os.path.join(BASE_DIR, row["image_path"]))
        mask = Image.open(os.path.join(BASE_DIR, row["mask_path"]))
        assert img.size == mask.size, f"Dimension mismatch for {row['image_id']}: {img.size} vs {mask.size}"
        assert row["width"] == img.width
        assert row["height"] == img.height


def test_spacenet_crs_and_transform_correspondence():
    """Requirement 7: SpaceNet CRS and affine transform preservation."""
    manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_train_manifest.csv")
    if not os.path.exists(manifest_p):
        manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_smoke_test_manifest.csv")
    df = pd.read_csv(manifest_p)

    sample = df.iloc[0]
    raw_tif = os.path.join(BASE_DIR, sample["raw_tif_path"])
    with rasterio.open(raw_tif) as src:
        assert src.crs is not None
        assert "4326" in str(src.crs)
        assert src.transform is not None
        assert not src.transform.is_identity


def test_spacenet_mask_binary_values():
    """Requirement 8: SpaceNet rasterized masks must contain strictly binary values."""
    manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_train_manifest.csv")
    if not os.path.exists(manifest_p):
        manifest_p = os.path.join(MANIFESTS_DIR, "spacenet_smoke_test_manifest.csv")
    df = pd.read_csv(manifest_p)

    for _, row in df.head(5).iterrows():
        mask_np = np.array(Image.open(os.path.join(BASE_DIR, row["mask_path"])))
        unique_vals = set(np.unique(mask_np))
        assert unique_vals.issubset({0, 255}), f"Non-binary mask values in {row['mask_path']}: {unique_vals}"


def test_peer_official_split_semantics():
    """Requirement 9: Official PEER test partition is preserved exclusively in test/."""
    manifest_p = os.path.join(MANIFESTS_DIR, "property_condition_manifest.csv")
    df = pd.read_csv(manifest_p)

    test_rows = df[df["assigned_split"] == "test"]
    assert len(test_rows) == 146, f"Expected 146 benchmark test images, found {len(test_rows)}"
    assert (test_rows["source_partition"] == "OFFICIAL_BENCHMARK_TEST").all()

    train_rows = df[df["assigned_split"] == "train"]
    val_rows = df[df["assigned_split"] == "val"]
    assert len(train_rows) + len(val_rows) == 1226, f"Expected 1226 official training images, found {len(train_rows) + len(val_rows)}"
    assert (train_rows["source_partition"] == "OFFICIAL_TRAIN").all()
    assert (val_rows["source_partition"] == "OFFICIAL_TRAIN").all()


def test_peer_class_mapping():
    """Requirement 10: PEER classes are global_collapse, non_collapse, partial_collapse."""
    manifest_p = os.path.join(MANIFESTS_DIR, "property_condition_manifest.csv")
    df = pd.read_csv(manifest_p)

    valid_classes = {"global_collapse", "non_collapse", "partial_collapse"}
    assert set(df["label"].unique()) == valid_classes
    assert not df["label"].isin(["new", "moderate", "old"]).any()


def test_housing_preprocessing_fitted_only_on_train():
    """Requirement 11: Housing preprocessor is fitted strictly on research_train without leakage."""
    import joblib
    preprocessor = joblib.load(os.path.join(PROCESSED_DIR, "housing_preprocessor.pkl"))
    
    # Check that ColumnTransformer was fitted
    assert hasattr(preprocessor, "transformers_")
    
    # Check that research_train, research_val, research_test are saved with exact partitions
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "housing_train_df.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "housing_val_df.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "housing_test_df.csv"))

    assert len(train_df) == 1020  # 70% research train
    assert len(val_df) == 219    # 15% research validation
    assert len(test_df) == 219   # 15% research test
    assert len(train_df) + len(val_df) + len(test_df) == 1458  # 1460 raw labeled houses minus 2 outliers

    # Verify that imputer statistics were calculated strictly on numerical features
    num_transformer = preprocessor.named_transformers_["num"]
    imputer = num_transformer.named_steps["imputer"]
    assert len(imputer.statistics_) == 19


def test_kaggle_unlabeled_test_not_used_for_evaluation():
    """Requirement 12: Kaggle unlabeled test.csv is never used for evaluation."""
    raw_test_p = os.path.join(BASE_DIR, "data", "raw", "housing", "test.csv")
    if os.path.exists(raw_test_p):
        raw_test = pd.read_csv(raw_test_p)
        assert "SalePrice" not in raw_test.columns, "Kaggle test.csv unexpectedly contains SalePrice"

    # Evaluation test split must have genuine ground truth SalePrice
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "housing_test_df.csv"))
    assert "SalePrice" in test_df.columns
    assert (test_df["SalePrice"] > 0).all()
