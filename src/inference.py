# src/inference.py
import pandas as pd
import logging
from src import config
from src.utils import load_object
from src.preprocess import feature_engineering # Re-use the same FE logic

logger = logging.getLogger(__name__)

def predict(input_data_path: str, model_path: str, preprocessor_path: str, output_path: str):
    """
    Loads data, preprocesses it, makes predictions, and saves the results.
    """
    logger.info(f"Loading data from {input_data_path}")
    try:
        input_df = pd.read_csv(input_data_path)
    except FileNotFoundError:
        logger.error(f"Input data file not found: {input_data_path}")
        return
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        return

    logger.info("Loading preprocessor and model...")
    try:
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
    except FileNotFoundError as e:
        logger.error(f"Model or preprocessor file not found: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {e}")
        return

    # Store customerID if present, then drop for preprocessing
    customer_ids = None
    if 'customerID' in input_df.columns:
        customer_ids = input_df['customerID']
        input_df_processed = input_df.drop(columns=['customerID'], errors='ignore')
    else:
        input_df_processed = input_df.copy()
    
    # Handle 'TotalCharges' - convert to numeric, impute NaNs (same as in ingestion/api)
    if 'TotalCharges' in input_df_processed.columns:
        input_df_processed['TotalCharges'] = pd.to_numeric(input_df_processed['TotalCharges'], errors='coerce').fillna(0)

    logger.info("Applying feature engineering...")
    input_df_fe = feature_engineering(input_df_processed)

    # Ensure columns are in the same order as during fitting
    # This is crucial and should align with how the preprocessor was trained
    # A more robust way is to save the feature names from training preprocessor
    # For now, we'll assume the order config.NUMERIC_FEATURES + (config.CATEGORICAL_FEATURES + engineered_cat_features)
    # and that the preprocessor handles any extra/missing columns based on its 'remainder' setting
    
    # Reconstruct the order of features *before* one-hot encoding
    # as expected by the preprocessor.transform()
    expected_cols_for_transform = config.NUMERIC_FEATURES.copy()
    engineered_cat_features = []
    if 'tenure_bucket' in input_df_fe.columns: # Check if FE created it
        engineered_cat_features.append('tenure_bucket')
        input_df_fe['tenure_bucket'] = input_df_fe['tenure_bucket'].astype(str)
    if config.INTERACTION_FEATURES:
        for col1, col2 in config.INTERACTION_FEATURES:
            if f'{col1}_x_{col2}' in input_df_fe.columns: # Check if FE created it
                 engineered_cat_features.append(f'{col1}_x_{col2}')
                 input_df_fe[f'{col1}_x_{col2}'] = input_df_fe[f'{col1}_x_{col2}'].astype(str)
    
    all_categorical_features_ohe = [col for col in config.CATEGORICAL_FEATURES + engineered_cat_features if col in input_df_fe.columns]
    
    # Add any missing columns that the preprocessor expects (fill with NaN, imputer will handle)
    # This is simplified. A robust approach uses preprocessor.feature_names_in_ if available
    # or saves the exact feature list from training.
    expected_cols_for_transform = config.NUMERIC_FEATURES + all_categorical_features_ohe
    for col in expected_cols_for_transform:
        if col not in input_df_fe.columns:
            input_df_fe[col] = np.nan 
            
    # Reorder columns to match the order used for fitting the preprocessor
    # This is critical for ColumnTransformer.
    # The `transformers` list in `get_preprocessor` defines this order for the transformed features.
    # We need to ensure `input_df_fe` has columns in the order they were presented to `fit_transform`.
    # The best practice is to save `preprocessor.feature_names_in_` during training and use it here.
    # For this example, we re-create the order based on config.
    original_feature_order = config.NUMERIC_FEATURES + [col for col in config.CATEGORICAL_FEATURES if col in input_df_fe.columns]
    if 'tenure_bucket' in input_df_fe.columns:
        original_feature_order.append('tenure_bucket')
    if config.INTERACTION_FEATURES:
        for col1, col2 in config.INTERACTION_FEATURES:
            interaction_col_name = f'{col1}_x_{col2}'
            if interaction_col_name in input_df_fe.columns:
                original_feature_order.append(interaction_col_name)
    
    # Ensure all columns used for fitting the preprocessor are present and in order
    # This might need adjustment based on how 'get_preprocessor' is structured, especially 'remainder'
    try:
        input_df_fe_ordered = input_df_fe[preprocessor.feature_names_in_]
    except AttributeError: # Fallback if feature_names_in_ is not available (older scikit-learn or custom transformer)
        # Attempt to use the order defined in config if 'feature_names_in_' is not set
        # This is less robust
        present_cols = [col for col in original_feature_order if col in input_df_fe.columns]
        input_df_fe_ordered = input_df_fe[present_cols]
        logger.warning("Using feature order based on config; preprocessor.feature_names_in_ not found.")
    except KeyError as e:
        logger.error(f"Missing columns for preprocessor: {e}. Columns available: {input_df_fe.columns.tolist()}")
        raise
        
    logger.info("Transforming data with preprocessor...")
    X_processed_array = preprocessor.transform(input_df_fe_ordered)
    
    # Get feature names after transformation to create DataFrame for model prediction
    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        logger.warning("Using generic feature names as get_feature_names_out() failed.")
        transformed_feature_names = [f"feature_{i}" for i in range(X_processed_array.shape[1])]
    
    X_processed_df = pd.DataFrame(X_processed_array, columns=transformed_feature_names)

    logger.info("Making predictions...")
    probabilities = model.predict_proba(X_processed_df)[:, 1]
    predictions_numeric = model.predict(X_processed_df)

    results_df = pd.DataFrame({
        'ProbabilityChurn': probabilities,
        'PredictionChurn': predictions_numeric
    })
    
    # Map numeric predictions back to Yes/No
    results_df['PredictionChurnLabel'] = results_df['PredictionChurn'].map({1: 'Yes', 0: 'No'})

    if customer_ids is not None:
        results_df = pd.concat([customer_ids, results_df], axis=1)

    logger.info(f"Saving predictions to {output_path}")
    results_df.to_csv(output_path, index=False)
    logger.info("Inference complete.")

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input file for testing
    sample_data = {
        'gender': ['Female', 'Male'], 'SeniorCitizen': [0, 1], 'Partner': ['Yes', 'No'], 
        'Dependents': ['No', 'No'], 'tenure': [1, 24], 'PhoneService': ['No', 'Yes'], 
        'MultipleLines': ['No phone service', 'Yes'], 'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes'], 'OnlineBackup': ['Yes', 'No'], 'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'Yes'], 'StreamingTV': ['No', 'Yes'], 'StreamingMovies': ['No', 'Yes'],
        'Contract': ['Month-to-month', 'One year'], 'PaperlessBilling': ['Yes', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check'], 'MonthlyCharges': [29.85, 90.00],
        'TotalCharges': ['29.85', '2700.00'] # Note: TotalCharges as string initially
    }
    sample_df_for_inference = pd.DataFrame(sample_data)
    sample_input_path = config.DATA_DIR / "sample_inference_input.csv"
    sample_df_for_inference.to_csv(sample_input_path, index=False)

    predict(
        input_data_path=str(sample_input_path),
        model_path=str(config.MODEL_PATH_XGB),
        preprocessor_path=str(config.PREPROCESSOR_PATH),
        output_path=str(config.PROCESSED_DATA_DIR / "predictions.csv")
    )