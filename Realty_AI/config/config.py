import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample')

# Assets
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
SAMPLE_IMAGES_DIR = os.path.join(ASSETS_DIR, 'sample_images')

# Model Configurations
SEGMENTATION_CLASSES = ['Urban', 'Vegetation', 'Water']
CONDITION_CLASSES = ['New', 'Moderate', 'Old']
IMAGE_SIZE = (256, 256)

# Random Seed for Reproducibility
RANDOM_SEED = 42
