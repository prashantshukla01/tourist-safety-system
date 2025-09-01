# retrain_simple.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import random
from datetime import datetime, timedelta
import os


# Copy the same class definition here
class EnsembleAnomalyDetector(BaseEstimator, TransformerMixin):
    """Ensemble of Isolation Forest and Local Outlier Factor for anomaly detection"""

    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,
            n_neighbors=20
        )
        self.weights = [0.6, 0.4]  # Weight for ISO Forest and LOF respectively

    def fit(self, X, y=None):
        self.iso_forest.fit(X)
        self.lof.fit(X)
        return self

    def predict(self, X):
        iso_scores = self.iso_forest.decision_function(X)
        lof_scores = self.lof.decision_function(X)

        # Normalize scores to [0, 1] range
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

        # Weighted ensemble score
        ensemble_scores = self.weights[0] * iso_scores_norm + self.weights[1] * lof_scores_norm

        # Convert to binary predictions (1 for normal, -1 for anomaly)
        threshold = np.percentile(ensemble_scores, 100 * self.contamination)
        predictions = np.where(ensemble_scores < threshold, -1, 1)

        return predictions, ensemble_scores

    def decision_function(self, X):
        _, scores = self.predict(X)
        return scores


def generate_synthetic_data(n_samples=10000):
    """Generate simple synthetic data for testing"""
    np.random.seed(42)

    # Simple features: velocity, acceleration, distance
    velocity = np.concatenate([
        np.random.normal(1.4, 0.5, 9000),  # Normal walking
        np.random.normal(5.0, 2.0, 1000)  # Anomalous (running)
    ])

    acceleration = np.concatenate([
        np.random.normal(0, 0.3, 9000),  # Normal
        np.random.normal(1.5, 1.0, 1000)  # Anomalous
    ])

    distance = np.concatenate([
        np.random.exponential(10, 9000),  # Close to itinerary
        np.random.exponential(100, 1000)  # Far from itinerary
    ])

    # Time features
    hour_of_day = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'velocity': velocity,
        'acceleration': acceleration,
        'distance_from_itinerary': distance,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week
    })

    return data


def train_and_save_model():
    """Train and save a simple model"""
    print("Generating synthetic data...")
    X = generate_synthetic_data(10000)

    print("Training model...")
    # Simple pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('detector', EnsembleAnomalyDetector(contamination=0.1))
    ])

    # Train the model
    pipeline.fit(X)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save the model
    joblib.dump(pipeline, 'models/anomaly_detection_model.joblib')
    joblib.dump(pipeline.named_steps['scaler'], 'models/scaler.joblib')

    print("✅ Model trained and saved successfully!")
    return pipeline


if __name__ == '__main__':
    train_and_save_model()