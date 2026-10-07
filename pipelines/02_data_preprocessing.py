"""
Pipeline 02: Data Preprocessing & Feature Engineering
Performs:
1. Tabular cleaning, missing value imputation, outlier handling, feature encoding for Housing Prices
2. Time-series transformation, unpivoting, rolling statistics, and appreciation metrics for Zillow ZHVI
3. Train/Val/Test dataset splitting for SpaceNet satellite segmentation & Property Condition CNN
4. Generates comprehensive EDA summary statistics
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def preprocess_housing_data():
    """Cleans and encodes Kaggle Housing dataset for regression modeling."""
    raw_path = os.path.join(RAW_DIR, "housing", "train.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Housing train dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw housing data: {df.shape}")

    # Remove extreme outliers in living area as recommended by dataset authors
    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))].copy()

    # Define key numerical and categorical features
    num_features = [
        "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath",
        "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
        "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF"
    ]
    # Keep only those present in df
    num_features = [f for f in num_features if f in df.columns]

    cat_features = [
        "Neighborhood", "MSZoning", "BldgType", "HouseStyle",
        "ExterQual", "Foundation", "BsmtQual", "KitchenQual"
    ]
    cat_features = [f for f in cat_features if f in df.columns]

    target = "SalePrice"
    all_features = num_features + cat_features
    X = df[all_features].copy()
    y = df[target].values

    # Leakage-Safe Research Train / Val / Test Partitioning (70 / 15 / 15)
    # Split directly from labeled Ames training population BEFORE any imputation or scaling
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.17647, random_state=42)

    # Preprocessing Pipeline: ALL imputers, scalers, and encoders are fit ONLY on X_train
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features)
        ]
    )

    # Fit ONLY on training data; transform validation and test splits without leakage
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save processed arrays and preprocessor
    joblib.dump(preprocessor, os.path.join(PROCESSED_DIR, "housing_preprocessor.pkl"))
    with open(os.path.join(PROCESSED_DIR, "housing_features.json"), "w") as f:
        json.dump({
            "num_features": num_features,
            "cat_features": cat_features,
            "all_features": all_features,
            "target": target
        }, f, indent=2)

    np.save(os.path.join(PROCESSED_DIR, "housing_X_train.npy"), X_train_trans)
    np.save(os.path.join(PROCESSED_DIR, "housing_y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "housing_X_val.npy"), X_val_trans)
    np.save(os.path.join(PROCESSED_DIR, "housing_y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "housing_X_test.npy"), X_test_trans)
    np.save(os.path.join(PROCESSED_DIR, "housing_y_test.npy"), y_test)

    # Save research partition DataFrames (raw scale for EDA, feature exploration, and audit)
    X_train.assign(SalePrice=y_train).to_csv(os.path.join(PROCESSED_DIR, "housing_train_df.csv"), index=False)
    X_val.assign(SalePrice=y_val).to_csv(os.path.join(PROCESSED_DIR, "housing_val_df.csv"), index=False)
    X_test.assign(SalePrice=y_test).to_csv(os.path.join(PROCESSED_DIR, "housing_test_df.csv"), index=False)

    logger.info(f"Preprocessed housing data saved: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    return df, num_features, cat_features


def preprocess_zillow_data():
    """Unpivots Zillow wide monthly table into a clean time-series format."""
    raw_path = os.path.join(RAW_DIR, "zillow", "Metro_zhvi_month.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Zillow raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw Zillow data: {df.shape}")

    # Identify metadata columns vs date columns
    meta_cols = [c for c in ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"] if c in df.columns]
    date_cols = [c for c in df.columns if c not in meta_cols]

    # Filter top 25 prominent metros (or all if fewer)
    df_top = df.head(35).copy()

    # Melt to long format
    df_long = pd.melt(
        df_top,
        id_vars=meta_cols,
        value_vars=date_cols,
        var_name="Date",
        value_name="ZHVI"
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"])
    df_long = df_long.dropna(subset=["ZHVI"]).sort_values(["RegionName", "Date"])

    # Compute 1-year and 3-year historical appreciation rates per region
    summary_records = []
    for region, grp in df_long.groupby("RegionName"):
        grp = grp.sort_values("Date")
        if len(grp) >= 24:
            current_val = grp["ZHVI"].iloc[-1]
            val_1yr_ago = grp["ZHVI"].iloc[-13] if len(grp) >= 13 else grp["ZHVI"].iloc[0]
            val_5yr_ago = grp["ZHVI"].iloc[-61] if len(grp) >= 61 else grp["ZHVI"].iloc[0]
            yoy_growth = ((current_val - val_1yr_ago) / val_1yr_ago) * 100
            five_yr_growth = ((current_val - val_5yr_ago) / val_5yr_ago) * 100

            summary_records.append({
                "RegionName": region,
                "StateName": grp["StateName"].iloc[0] if "StateName" in grp else "US",
                "LatestZHVI": round(current_val, 2),
                "YoY_Growth_Pct": round(yoy_growth, 2),
                "5Yr_Growth_Pct": round(five_yr_growth, 2),
                "StartDate": grp["Date"].iloc[0].strftime("%Y-%m-%d"),
                "EndDate": grp["Date"].iloc[-1].strftime("%Y-%m-%d"),
                "TotalMonths": len(grp)
            })

    summary_df = pd.DataFrame(summary_records).sort_values("LatestZHVI", ascending=False)
    summary_df.to_csv(os.path.join(PROCESSED_DIR, "zillow_metro_summary.csv"), index=False)
    df_long.to_csv(os.path.join(PROCESSED_DIR, "zillow_zhvi_processed.csv"), index=False)

    logger.info(f"Saved processed Zillow time-series ({len(df_long)} rows across {len(summary_df)} metros).")
    return summary_df


def prepare_vision_dataset_splits():
    """Prepares split manifests for SpaceNet and Property Condition vision datasets preserving authentic splits."""
    # 1. SpaceNet splits (Preserve official chip-level split without tile leakage)
    sn_train_manifest_path = os.path.join(BASE_DIR, "data", "manifests", "spacenet_train_manifest.csv")
    sn_manifest_path = os.path.join(BASE_DIR, "data", "manifests", "spacenet_manifest.csv")
    
    # Priority to research-training manifest, fallback to smoke-test / initial manifest
    active_sn_manifest = sn_train_manifest_path if os.path.exists(sn_train_manifest_path) else sn_manifest_path
    if os.path.exists(active_sn_manifest):
        sn_df = pd.read_csv(active_sn_manifest)
        spacenet_splits = {"train": [], "val": [], "test": []}
        for _, row in sn_df.iterrows():
            split_name = row.get("assigned_split", row.get("source_split", "train"))
            if split_name in spacenet_splits:
                spacenet_splits[split_name].append({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "num_buildings": int(row.get("num_buildings", 0)),
                    "building_pixel_pct": float(row.get("building_pixel_pct", 0.0))
                })
        with open(os.path.join(PROCESSED_DIR, "spacenet_splits.json"), "w") as f:
            json.dump(spacenet_splits, f, indent=2)
        logger.info(f"SpaceNet splits preserved from manifest: {len(spacenet_splits['train'])} train, {len(spacenet_splits['val'])} val, {len(spacenet_splits['test'])} test.")

    # 2. Property condition splits (Official source semantics: test isolated, train/val from official train)
    cond_manifest_path = os.path.join(BASE_DIR, "data", "manifests", "property_condition_manifest.csv")
    if os.path.exists(cond_manifest_path):
        cond_df = pd.read_csv(cond_manifest_path)
        condition_splits = {"train": [], "val": [], "test": []}
        for _, row in cond_df.iterrows():
            split_name = row.get("assigned_split", row.get("split", "train"))
            if split_name in condition_splits:
                condition_splits[split_name].append({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "class_id": int(row["class_id"])
                })
        with open(os.path.join(PROCESSED_DIR, "property_condition_splits.json"), "w") as f:
            json.dump(condition_splits, f, indent=2)
        logger.info(f"Property condition splits preserved from manifest: {len(condition_splits['train'])} train, {len(condition_splits['val'])} val, {len(condition_splits['test'])} test.")


def generate_eda_summary(housing_df, zillow_summary):
    """Generates unified exploratory data analysis summary JSON."""
    sn_cnt = 0
    for sn_p in [
        os.path.join(BASE_DIR, "data", "manifests", "spacenet_train_manifest.csv"),
        os.path.join(BASE_DIR, "data", "manifests", "spacenet_smoke_test_manifest.csv"),
        os.path.join(BASE_DIR, "data", "manifests", "spacenet_manifest.csv")
    ]:
        if os.path.exists(sn_p):
            sn_cnt = len(pd.read_csv(sn_p))
            break

    cond_p = os.path.join(BASE_DIR, "data", "manifests", "property_condition_manifest.csv")
    cond_cnt = len(pd.read_csv(cond_p)) if os.path.exists(cond_p) else 0

    eda = {
        "dataset_statistics": {
            "housing_records": len(housing_df),
            "housing_features_count": len(housing_df.columns),
            "housing_avg_price": float(housing_df["SalePrice"].mean()),
            "housing_median_price": float(housing_df["SalePrice"].median()),
            "housing_min_price": float(housing_df["SalePrice"].min()),
            "housing_max_price": float(housing_df["SalePrice"].max()),
            "zillow_total_metros": len(zillow_summary),
            "zillow_highest_zhvi_metro": zillow_summary.iloc[0]["RegionName"],
            "zillow_highest_zhvi_value": float(zillow_summary.iloc[0]["LatestZHVI"]),
            "spacenet_tiles_count": sn_cnt,
            "condition_images_count": cond_cnt
        },
        "housing_price_percentiles": {
            "p10": float(housing_df["SalePrice"].quantile(0.10)),
            "p25": float(housing_df["SalePrice"].quantile(0.25)),
            "p50": float(housing_df["SalePrice"].quantile(0.50)),
            "p75": float(housing_df["SalePrice"].quantile(0.75)),
            "p90": float(housing_df["SalePrice"].quantile(0.90))
        },
        "top_neighborhoods": housing_df["Neighborhood"].value_counts().head(8).to_dict() if "Neighborhood" in housing_df else {}
    }

    with open(os.path.join(PROCESSED_DIR, "eda_summary.json"), "w") as f:
        json.dump(eda, f, indent=2)
    logger.info("Generated eda_summary.json")


def run_preprocessing():
    logger.info("=== Starting RealtyAI Data Preprocessing Pipeline ===")
    housing_df, _, _ = preprocess_housing_data()
    zillow_summary = preprocess_zillow_data()
    prepare_vision_dataset_splits()
    generate_eda_summary(housing_df, zillow_summary)
    logger.info("=== Data Preprocessing Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_preprocessing()
