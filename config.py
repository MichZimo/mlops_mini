from dotenv import load_dotenv
from pathlib import Path

load_dotenv() # Load variables from .env into os.environ

BASE_DIR = Path(__file__).resolve().parent # Define project root
