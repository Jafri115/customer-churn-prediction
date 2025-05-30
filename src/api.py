# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import logging
from typing import List, Union, Optional
import joblib
from pathlib import Path

from src import config
from src.utils import load_object
from src.preprocess import preprocess_pipeline
from sklearn.preprocessing import LabelEncoder

# For Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict customer churn based on Telco data.",
    version="0.1.0"
)

# Prometheus metrics
Instrumentator().add(
    metrics.default(
        should_include_status=True,  # important for status_code
        should_include_method=True,
        should_include_path=False,
    )
).instrument(app).expose(app)
PREDICTION_COUNT = Counter("predictions_total", "Total number of predictions made")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of prediction requests")

# --- Globals to store loaded model and preprocessor ---
MODEL = None
PREPROCESSOR = None
FEATURE_NAMES = None

@app.on_event("startup")
async def load_model_and_preprocessor():
    global MODEL, PREPROCESSOR, FEATURE_NAMES
    try:
        # Load the trained model
        MODEL = load_object(config.MODEL_PATH_XGB) 
        logger.info(f"Model loaded successfully from {config.MODEL_PATH_XGB}")
        
        # Load the preprocessing pipeline
        PREPROCESSOR = load_object(config.PREPROCESSOR_PATH)
        logger.info(f"Preprocessor loaded successfully from {config.PREPROCESSOR_PATH}")
        
        # Load feature names
        FEATURE_NAMES = joblib.load(config.FEATURE_NAMES_PATH)
        logger.info(f"Feature names loaded successfully: {len(FEATURE_NAMES)} features")

    except Exception as e:
        logger.error(f"Error loading model or preprocessor at startup: {e}", exc_info=True)
        MODEL = None 
        PREPROCESSOR = None
        FEATURE_NAMES = None

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # you may want to handle this as float after cleaning

class CustomerList(BaseModel):
    customers: List[CustomerInput]

class ChurnPrediction(BaseModel):
    customer_id: Optional[int] = None
    churn_probability: float
    churn_prediction: int
    churn_label: str

@app.get("/health", summary="Check API Health")
async def health_check():
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded. API is not healthy.")
    return {
        "status": "healthy", 
        "model_loaded": MODEL is not None, 
        "preprocessor_loaded": PREPROCESSOR is not None,
        "feature_names_loaded": FEATURE_NAMES is not None
    }

def process_customer_data(input_data: pd.DataFrame):
    """Process customer data through the preprocessing pipeline"""
    # Convert TotalCharges to numeric
    if 'TotalCharges' in input_data.columns:
        input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce')
        input_data['TotalCharges'].fillna(0, inplace=True)
    
    # Use the preprocessor directly
    if PREPROCESSOR is None:
        raise ValueError("Preprocessor not loaded")
    
    # Handle customerID if present
    if 'customerID' in input_data.columns:
        customer_ids = input_data['customerID'].tolist()
        input_data = input_data.drop(columns=['customerID'])
    else:
        customer_ids = None
    
    # Apply feature engineering manually (similar to what happens in preprocess_pipeline)
    from src.preprocess import feature_engineering
    input_data = feature_engineering(input_data)
    
    # Transform the data using the loaded preprocessor
    X_processed = PREPROCESSOR.transform(input_data)
    
    return X_processed, customer_ids

@app.post("/predict", response_model=List[ChurnPrediction])
def make_prediction(data: CustomerList):
    global MODEL, PREPROCESSOR, FEATURE_NAMES
    
    # Start latency tracking
    with PREDICTION_LATENCY.time():
        try:
            # Check if model and preprocessor are loaded
            if MODEL is None or PREPROCESSOR is None:
                raise HTTPException(status_code=503, 
                                  detail="Model or preprocessor not loaded. Try again later.")
            
            # Convert input to DataFrame
            input_data = pd.DataFrame([customer.dict() for customer in data.customers])
            logger.info(f"Received prediction request for {len(input_data)} customers")
            
            # Process the input data
            X_processed, _ = process_customer_data(input_data)
            
            # Make predictions
            churn_probabilities = MODEL.predict_proba(X_processed)[:, 1]
            churn_predictions = (churn_probabilities >= 0.5).astype(int)
            
            # Increment prediction counter
            PREDICTION_COUNT.inc(len(input_data))
            
            # Format results
            results = []
            for i, (prob, pred) in enumerate(zip(churn_probabilities, churn_predictions)):
                results.append(ChurnPrediction(
                    customer_id=i,
                    churn_probability=float(prob),
                    churn_prediction=int(pred),
                    churn_label="Yes" if pred == 1 else "No"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# To run the app locally (from the project root):
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload