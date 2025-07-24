import pandas as pd
import joblib

def load_sklearn_model(path):
    """Load a scikit-learn model from a file."""
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_action(model, features: pd.DataFrame):
    """Predict trading action using a scikit-learn model and feature DataFrame."""
    if model is None:
        return 0  # Hold if no model
    try:
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0
