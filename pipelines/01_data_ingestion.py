"""
Pipeline 01: Data Ingestion (Real Data Only - Phase 1 Migration)
Acquires and verifies authentic datasets:
1. Ames Housing Prices dataset (authentic Dean De Cock 2011 dataset from Kaggle)
2. Zillow ZHVI Metro Time Series dataset (from Zillow Research)
3. SpaceNet 2 Satellite Building Footprints (authentic Maxar/DigitalGlobe WorldView-3 Vegas imagery)
4. PEER Hub ImageNet Structural Condition dataset (authentic UC Berkeley PEER reconnaissance photos)

STRICT RULE: NO SYNTHETIC FALLBACKS. All datasets must be authentic or fail loudly.
"""

import os
import shutil
import urllib.request
import logging
import pandas as pd

from scripts.ingest_spacenet_real import main as ingest_spacenet_data
from scripts.ingest_property_condition_real import main as ingest_property_condition_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
HOUSING_DIR = os.path.join(RAW_DIR, "housing")
ZILLOW_DIR = os.path.join(RAW_DIR, "zillow")
SPACENET_DIR = os.path.join(RAW_DIR, "spacenet")
CONDITION_DIR = os.path.join(RAW_DIR, "property_conditions")

DOWNLOADS_DIR = os.path.expanduser(r"~\Downloads\house-prices-advanced-regression-techniques")
ZILLOW_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def ingest_kaggle_housing():
    """
    Verifies presence of authentic Ames Housing dataset in data/raw/housing/.
    Copies from ~/Downloads if available.
    FAILS LOUDLY if dataset is missing; never generates synthetic housing data.
    """
    os.makedirs(HOUSING_DIR, exist_ok=True)
    train_dest = os.path.join(HOUSING_DIR, "train.csv")
    test_dest = os.path.join(HOUSING_DIR, "test.csv")

    # If missing locally, attempt to copy from local downloads
    if not os.path.exists(train_dest) and os.path.isdir(DOWNLOADS_DIR):
        src_train = os.path.join(DOWNLOADS_DIR, "train.csv")
        src_test = os.path.join(DOWNLOADS_DIR, "test.csv")
        src_desc = os.path.join(DOWNLOADS_DIR, "data_description.txt")

        if os.path.exists(src_train):
            shutil.copy2(src_train, train_dest)
            logger.info(f"Copied {src_train} -> {train_dest}")
        if os.path.exists(src_test):
            shutil.copy2(src_test, test_dest)
            logger.info(f"Copied {src_test} -> {test_dest}")
        if os.path.exists(src_desc):
            shutil.copy2(src_desc, os.path.join(HOUSING_DIR, "data_description.txt"))

    if not os.path.exists(train_dest):
        raise FileNotFoundError(
            f"Authentic Ames Housing dataset not found at '{train_dest}'. "
            "Synthetic generation is disabled. Please download 'train.csv' from "
            "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data "
            f"and place it in '{HOUSING_DIR}'."
        )

    df = pd.read_csv(train_dest)
    logger.info(f"Verified authentic Ames Housing dataset at {train_dest}: {len(df)} rows, {len(df.columns)} columns.")
    return df


def ingest_zillow_data():
    """
    Acquires or verifies authentic Zillow ZHVI Metro monthly time series data.
    FAILS LOUDLY if dataset cannot be acquired; never generates synthetic time series.
    """
    os.makedirs(ZILLOW_DIR, exist_ok=True)
    zillow_path = os.path.join(ZILLOW_DIR, "Metro_zhvi_month.csv")

    if not os.path.exists(zillow_path) or os.path.getsize(zillow_path) < 1000:
        logger.info(f"Downloading official Zillow ZHVI data from {ZILLOW_URL}...")
        try:
            req = urllib.request.Request(
                ZILLOW_URL,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(zillow_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            logger.info(f"Downloaded official Zillow dataset to {zillow_path} ({os.path.getsize(zillow_path)} bytes)")
        except Exception as e:
            if not os.path.exists(zillow_path) or os.path.getsize(zillow_path) < 1000:
                raise FileNotFoundError(
                    f"Failed to download Zillow ZHVI dataset ({e}) and no local copy exists at '{zillow_path}'. "
                    "Synthetic fallback generation is strictly disabled. Please download "
                    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv from "
                    f"https://www.zillow.com/research/data/ and save it as '{zillow_path}'."
                )

    df_zillow = pd.read_csv(zillow_path)
    logger.info(f"Verified authentic Zillow ZHVI dataset at {zillow_path}: {len(df_zillow)} metros, {len(df_zillow.columns)} columns.")
    return df_zillow


def run_ingestion():
    """Executes authentic data ingestion across all 4 project components."""
    logger.info("=== Starting RealtyAI Authentic Data Ingestion Pipeline ===")
    ingest_kaggle_housing()
    ingest_zillow_data()
    ingest_spacenet_data()
    ingest_property_condition_data()
    logger.info("=== Authentic Data Ingestion Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_ingestion()
