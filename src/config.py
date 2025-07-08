import os

# === Main project directory ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Data paths ===
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "telco_churn_clean.csv")

# === Model paths ===
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

# === Reports ===
REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_report.md")

# === Streamlit ===
ASSETS_DIR = os.path.join(BASE_DIR, "app", "assets")

# === Model parameters ===
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"