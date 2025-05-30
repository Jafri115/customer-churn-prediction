# src/train.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import joblib # For saving the model
import matplotlib
matplotlib.use("Agg")

from src import config
from src.utils import save_object, load_object

logger = logging.getLogger(__name__)

def load_processed_data():
    """Loads processed training and testing data."""
    logger.info("Loading processed data...")
    try:
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        test_df = pd.read_csv(config.TEST_DATA_PATH)

        logger.info(f"train_df shape: {train_df.shape}, test_df shape: {test_df.shape}")

        if pd.isnull(train_df[config.TARGET_COLUMN]).any():
            raise ValueError("train_df contains NaNs. Please check the label preprocessing step.")
        else:
            logger.info("train_df does not contain NaNs.")
        
        X_train = train_df.drop(columns=[config.TARGET_COLUMN])
        y_train = train_df[config.TARGET_COLUMN]
        X_test = test_df.drop(columns=[config.TARGET_COLUMN])
        y_test = test_df[config.TARGET_COLUMN]
        
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Error loading processed data: {e}. Make sure preprocess.py has been run.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'xgb'):
    """Trains a model, logs with MLflow, and saves it."""
    
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{model_type}_training_run") as run:
        run_id = run.info.run_uuid
        logger.info(f"Starting MLflow Run ID: {run_id} for model: {model_type}")
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_data_shape", X_train.shape)
        mlflow.log_param("random_state", config.RANDOM_STATE)

        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        if model_type == 'xgb':
            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=config.RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
            mlflow.xgboost.autolog(log_models=False)  # <- fixed
            mlflow.log_params(model.get_params())
        elif model_type == 'lr':
            model = LogisticRegression(
                random_state=config.RANDOM_STATE,
                solver='liblinear',
                class_weight='balanced',
                max_iter=200
            )
            mlflow.sklearn.autolog(log_models=False)  # <- fixed
            mlflow.log_params(model.get_params())

        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        cv_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        logger.info(f"CV ROC AUC scores: {cv_auc_scores}")
        logger.info(f"Mean CV ROC AUC: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std():.4f})")
        mlflow.log_metric("cv_mean_roc_auc", cv_auc_scores.mean())
        mlflow.log_metric("cv_std_roc_auc", cv_auc_scores.std())

        model.fit(X_train, y_train)
        logger.info("Model training complete.")

        # Save model
        model_path = config.MODEL_PATH_XGB if model_type == 'xgb' else config.MODEL_PATH_LR
        save_object(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        if model_type == 'xgb':
            mlflow.xgboost.log_model(model, artifact_path="xgboost-model")
        elif model_type == 'lr':
            mlflow.sklearn.log_model(model, artifact_path="logistic-regression-model")
        
        mlflow.log_artifact(config.PREPROCESSOR_PATH, artifact_path="preprocessor")

        logger.info(f"Finished MLflow Run ID: {run_id}")
        return model, run_id

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, run_id: str = None):
    """Evaluates the model and logs metrics to MLflow if run_id is provided."""
    logger.info("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual_No', 'Actual_Yes'], columns=['Predicted_No', 'Predicted_Yes'])
    
    logger.info(f"Test Metrics:\n{pd.Series(metrics_dict)}")
    logger.info(f"Confusion Matrix:\n{cm_df}")

    if run_id:
        with mlflow.start_run(run_id=run_id, nested=True) as eval_run: # Using nested run
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics_dict.items()})
            mlflow.log_dict(cm_df.to_dict(), "confusion_matrix.json")
            logger.info(f"Logged evaluation metrics to MLflow run ID: {run_id}")
    else:
        logger.warning("No MLflow run_id provided for evaluation. Metrics not logged to MLflow.")
        
    return metrics_dict

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_processed_data()



    # Train and evaluate XGBoost
    logger.info("----- Training XGBoost -----")
    xgb_model, xgb_run_id = train_model(X_train, y_train, model_type='xgb')
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, run_id=xgb_run_id)

    # Train and evaluate Logistic Regression
    logger.info("\n----- Training Logistic Regression -----")
    lr_model, lr_run_id = train_model(X_train, y_train, model_type='lr')
    lr_metrics = evaluate_model(lr_model, X_test, y_test, run_id=lr_run_id)

    logger.info("Training and evaluation script finished.")
    logger.info(f"XGBoost Test ROC AUC: {xgb_metrics['roc_auc']:.4f}")
    logger.info(f"Logistic Regression Test ROC AUC: {lr_metrics['roc_auc']:.4f}")