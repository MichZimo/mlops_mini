import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv() # Load variables from .env into os.environ

BASE_DIR = Path(__file__).resolve().parent.parent  # Define project root

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MODEL_PATH = BASE_DIR /  os.getenv("MODEL_PATH",
				 "models/cali_housing_model.joblib") # Path joining
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
