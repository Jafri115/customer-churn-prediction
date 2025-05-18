# Modified data_ingestion.py
import pandas as pd
import logging
from src import config
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file and performs basic validation."""
    logger.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"{file_path} does not exist")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic cleaning like converting TotalCharges and encoding Churn."""
    logger.info("Cleaning data...")

    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Impute missing TotalCharges with 0
        missing_total = df['TotalCharges'].isnull().sum()
        if missing_total > 0:
            logger.warning(f"{missing_total} missing values in 'TotalCharges'. Filling with 0.")
            df['TotalCharges'].fillna(0, inplace=True)

    # Encode Churn column to 0/1
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        logger.info("Converted 'Churn' column to binary 0/1.")
        
        # Check for null values in Churn column
        churn_nulls = df['Churn'].isnull().sum()
        if churn_nulls > 0:
            logger.warning(f"{churn_nulls} null values found in 'Churn' column")

    return df

if __name__ == "__main__":
    logger.info("Starting Telco Customer Churn data ingestion...")

    df_raw = load_data(config.RAW_DATA_FILE)
    df_clean = clean_data(df_raw)

    logger.info("Sample cleaned data:\n" + str(df_clean.head()))
    logger.info("Data ingestion and cleaning completed.")