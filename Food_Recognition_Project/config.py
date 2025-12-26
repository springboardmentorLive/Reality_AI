import os
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    APP_NAME = "FoodVision: Food Classification using PyTorch and Groq"
    GROQ_API_KEY = os.getenv("PUBLIC_GROQ_API_KEY")
    TORCH_PATH = "./models/torch"
    ONNX_PATH = {
        "lenet64": "./models/onnx/lenet64.onnx",
        "tinyvgg": "./models/onnx/tinyvgg.onnx",
        "resnet18": "./models/onnx/resnet18.onnx"
    }
    NUTRIENTS_PATH = "./data/nutrients.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ARCH_PATH = {
        "lenet64": "./architecture/lenet64.py",
        "tinyvgg": "./architecture/tinyvgg.py",
    }
    CLASS_NAMES = ["pizza", "steak", "sushi"]


settings = Config()