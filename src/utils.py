import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_object(obj, path):
    """Saves a Python object to a file using joblib."""
    try:
        joblib.dump(obj, path)
        logger.info(f"Object saved to {path}")
    except Exception as e:
        logger.error(f"Error saving object to {path}: {e}")
        raise

def load_object(path):
    """Loads a Python object from a file using joblib."""
    try:
        obj = joblib.load(path)
        logger.info(f"Object loaded from {path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {path}: {e}")
        raise