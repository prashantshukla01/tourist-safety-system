# config.py
import os
from datetime import timedelta

class DevelopmentConfig:
    """Simple development configuration for localhost"""
    DEBUG = True
    SECRET_KEY = "simple-dev-key-123"  # Ye bas localhost ke liye hai
    JWT_SECRET_KEY = "jwt-simple-key-456"  # Ye bhi localhost ke liye
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    # SQLite database - ye automatically ban jayega
    SQLALCHEMY_DATABASE_URI = 'sqlite:///tourist_safety.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML model paths
    MODEL_PATH = os.path.join('models', 'anomaly_detection_model.joblib')
    SCALER_PATH = os.path.join('models', 'scaler.joblib')

    # Risk zones (latitude, longitude, radius in meters)
    RISK_ZONES = [
        ((40.7589, -73.9851), 500),  # Times Square example
        ((40.7484, -73.9857), 300),  # Empire State Building example
    ]

# Simple config for localhost
config = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}