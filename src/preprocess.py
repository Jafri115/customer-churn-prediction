# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import logging
from pathlib import Path
import joblib
from src import config
from src.utils import save_object, load_object
from src.data_ingestion import load_data, clean_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")
    df_fe = df.copy()

    if 'tenure' in df_fe.columns:
        df_fe['tenure_bucket'] = pd.cut(df_fe['tenure'], bins=config.TENURE_BINS, labels=config.TENURE_LABELS, right=False)
        logger.info("Created 'tenure_bucket' feature.")
    else:
        logger.warning("Tenure column not found for bucketing.")

    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    existing_service_cols = [col for col in service_cols if col in df_fe.columns]

    if existing_service_cols:
        for col in existing_service_cols:
            if df_fe[col].dtype == 'object':
                df_fe[col + '_numeric'] = df_fe[col].apply(lambda x: 1 if x == 'Yes' else 0)

        numeric_service_cols = [col + '_numeric' for col in existing_service_cols]
        df_fe['service_count'] = df_fe[numeric_service_cols].sum(axis=1)
        df_fe.drop(columns=numeric_service_cols, inplace=True)
        logger.info("Created 'service_count' feature.")
    else:
        logger.warning("No service columns found for 'service_count' feature.")

    for col1, col2 in config.INTERACTION_FEATURES:
        if col1 in df_fe.columns and col2 in df_fe.columns:
            df_fe[f'{col1}_x_{col2}'] = df_fe[col1].astype(str) + '_' + df_fe[col2].astype(str)
            logger.info(f"Created interaction feature '{col1}_x_{col2}'.")
        else:
            logger.warning(f"Could not create interaction term for {col1} and {col2}: one or both columns missing.")

    logger.info(f"Feature engineering complete. Columns: {df_fe.columns.tolist()}")
    return df_fe


def get_preprocessor(numeric_features: list, categorical_features_ohe: list, categorical_features_le: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer_ohe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_ohe', categorical_transformer_ohe, categorical_features_ohe)
        ],
        remainder='passthrough'  # keep other columns as is
    )
    return preprocessor


def get_feature_names_from_column_transformer(ct):
    """
    Get feature names from a ColumnTransformer.
    This is a more robust implementation that doesn't require passing the input_features.
    """
    # Get all transformers
    all_transformers = [(name, trans, cols) for name, trans, cols in ct.transformers_ 
                       if name != 'remainder' or cols != 'drop']
    
    feature_names = []
    
    # Process each transformer
    for name, transformer, columns in all_transformers:
        if name == 'remainder' and transformer == 'passthrough':
            # For passthrough features, use the column names directly
            if isinstance(columns, list):
                feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            # For transformers like OneHotEncoder that have get_feature_names_out
            if hasattr(transformer, 'feature_names_in_'):
                # Use the transformer's record of what it was fitted with
                names = transformer.get_feature_names_out()
            else:
                # Fallback for transformers without feature_names_in_
                names = [f"{name}_{i}" for i in range(transformer.transform(columns).shape[1])]
            feature_names.extend(names)
        elif hasattr(transformer, 'steps'):
            # For pipelines, check the last step
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, 'get_feature_names_out'):
                names = last_step.get_feature_names_out()
                feature_names.extend(names)
            else:
                # Use column names as is
                feature_names.extend(columns)
        else:
            # For other transformers, use the column names
            feature_names.extend(columns)
    
    return feature_names


def preprocess_pipeline(
    X: pd.DataFrame,
    y: pd.Series = None,
    fit: bool = True,
    preprocessor_path: Path = None,
):
    """
    Full preprocessing pipeline:
    - Cleans the data
    - Engineers new features
    - Transforms using a ColumnTransformer (scaling, encoding)
    - Optionally fits and saves feature names and preprocessor

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series, optional
        Target variable, by default None
    fit : bool, optional
        Whether to fit or just transform, by default True
    preprocessor_path : Path, optional
        Path to save or load the fitted preprocessor, by default None

    Returns
    -------
    X_processed : np.ndarray
    y : pd.Series or None
    preprocessor : fitted ColumnTransformer or None
    """
    # Clean and feature engineer
    X = clean_data(X)
    X = feature_engineering(X)
    
    # Drop 'customerID' if present
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])
    
    if fit:
        # Get features from config
        numeric_features = [col for col in config.NUMERIC_FEATURES if col in X.columns]
        if 'service_count' in X.columns:
            numeric_features.append('service_count')
            
        categorical_features_ohe = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
        if 'tenure_bucket' in X.columns:
            categorical_features_ohe.append('tenure_bucket')
        if 'Contract_x_PaymentMethod' in X.columns:
            categorical_features_ohe.append('Contract_x_PaymentMethod')
        
        preprocessor = get_preprocessor(numeric_features, categorical_features_ohe, [])
        
        # If we have the target column, remove it before fitting the preprocessor
        if config.TARGET_COLUMN in X.columns:
            y = X[config.TARGET_COLUMN]
            X = X.drop(columns=[config.TARGET_COLUMN])
        
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        # Extract feature names from the fitted preprocessor
        feature_names = get_feature_names_from_column_transformer(preprocessor)
        
        logger.info(f"Number of transformed features: {X_processed.shape[1]}")
        logger.info(f"Number of feature names: {len(feature_names)}")
        
        assert X_processed.shape[1] == len(feature_names), f"Feature count mismatch! {X_processed.shape[1]} vs {len(feature_names)}"
        
        # Save preprocessor and feature names
        if preprocessor_path:
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        feature_names_path = config.FEATURE_NAMES_PATH
        joblib.dump(feature_names, feature_names_path)
        logger.info(f"Saved feature names to {feature_names_path}")
        
        return X_processed, y, preprocessor
    
    else:
        # Load preprocessor for transform
        if preprocessor_path is None:
            raise ValueError("preprocessor_path must be provided when fit=False")
        
        preprocessor = joblib.load(preprocessor_path)
        
        # If we have the target column, separate it
        if config.TARGET_COLUMN in X.columns:
            y = X[config.TARGET_COLUMN]
            X = X.drop(columns=[config.TARGET_COLUMN])
        
        X_processed = preprocessor.transform(X)
        
        return X_processed, y, None


def main():
    logger.info("--- Starting Preprocessing Script ---")
    
    df = load_data(config.RAW_DATA_FILE)
    df = clean_data(df.copy())
    
    logger.info("Splitting data into train and test sets...")
    X = df.drop(config.TARGET_COLUMN, axis=1)
    y = df[config.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info(f"Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # For training data processing
    X_train_processed, y_train_processed, fitted_preprocessor = preprocess_pipeline(
        X_train.copy(),  # Pass X_train without merging with y_train
        y_train,
        fit=True,
        preprocessor_path=config.PREPROCESSOR_PATH,
    )
    
    # For test data processing
    X_test_processed, y_test_processed, _ = preprocess_pipeline(
        X_test.copy(),  # Pass X_test without merging with y_test
        y_test,
        fit=False,
        preprocessor_path=config.PREPROCESSOR_PATH,
    )
    
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load feature names for DataFrame columns
    feature_names = joblib.load(config.FEATURE_NAMES_PATH)
    
    # Convert numpy arrays back to DataFrames with correct feature names
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Combine features and target back into one DataFrame for saving
    train_processed_df = pd.concat([X_train_df.reset_index(drop=True), 
                                   pd.DataFrame(y_train_processed, columns=[config.TARGET_COLUMN]).reset_index(drop=True)], 
                                  axis=1)
    test_processed_df = pd.concat([X_test_df.reset_index(drop=True), 
                                  pd.DataFrame(y_test_processed, columns=[config.TARGET_COLUMN]).reset_index(drop=True)], 
                                 axis=1)
    
    train_processed_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    logger.info(f"Saved processed training data to {config.TRAIN_DATA_PATH}")
    
    test_processed_df.to_csv(config.TEST_DATA_PATH, index=False)
    logger.info(f"Saved processed test data to {config.TEST_DATA_PATH}")
    
    logger.info("Preprocessing script finished.")


if __name__ == "__main__":
    main()
