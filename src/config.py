# src/config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
VALID_DATA_PATH = PROCESSED_DATA_DIR / "validation.csv" # Optional

MODEL_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_PATH_XGB = MODEL_DIR / "churn_model_xgb.pkl"
MODEL_PATH_LR = MODEL_DIR / "churn_model_lr.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Features
TARGET_COLUMN = 'Churn'
# Identified during EDA or known from dataset description
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
# Features to drop
DROP_FEATURES = ['customerID']

# Model Training parameters
TEST_SIZE = 0.2
VALID_SIZE = 0.1 # For train_test_split, proportion of original data for validation
RANDOM_STATE = 42
CV_FOLDS = 5

# MLflow settings
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # Local SQLite database
MLFLOW_EXPERIMENT_NAME = "CustomerChurnPrediction"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Tenure buckets for feature engineering
TENURE_BINS = [0, 12, 24, 36, 48, 60, 100] # Max tenure is 72, so 100 is fine
TENURE_LABELS = ['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5yr+']

# Interaction terms for feature engineering (example)
INTERACTION_FEATURES = [('Contract', 'PaymentMethod')]