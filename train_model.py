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
    """Generate synthetic training data for the anomaly detection model"""
    np.random.seed(42)

    # Normal behavior parameters
    normal_velocity_mean = 1.4  # m/s (walking speed)
    normal_velocity_std = 0.5
    normal_acceleration_mean = 0
    normal_acceleration_std = 0.3

    # Anomalous behavior parameters
    anomaly_velocity_mean = 5.0  # m/s (running speed)
    anomaly_velocity_std = 2.0
    anomaly_acceleration_mean = 1.5
    anomaly_acceleration_std = 1.0

    # Generate normal data (90% of samples)
    n_normal = int(n_samples * 0.9)
    normal_velocity = np.random.normal(normal_velocity_mean, normal_velocity_std, n_normal)
    normal_acceleration = np.random.normal(normal_acceleration_mean, normal_acceleration_std, n_normal)
    normal_distance = np.random.exponential(10, n_normal)  # Most points are close to itinerary

    # Generate anomaly data (10% of samples)
    n_anomaly = n_samples - n_normal
    anomaly_velocity = np.random.normal(anomaly_velocity_mean, anomaly_velocity_std, n_anomaly)
    anomaly_acceleration = np.random.normal(anomaly_acceleration_mean, anomaly_acceleration_std, n_anomaly)
    anomaly_distance = np.random.exponential(100, n_anomaly)  # Anomalies are farther from itinerary

    # Combine data
    velocity = np.concatenate([normal_velocity, anomaly_velocity])
    acceleration = np.concatenate([normal_acceleration, anomaly_acceleration])
    distance = np.concatenate([normal_distance, anomaly_distance])

    # Add time-based features (hour of day, day of week)
    timestamps = [datetime.now() - timedelta(minutes=random.randint(0, 10000)) for _ in range(n_samples)]
    hour_of_day = [ts.hour for ts in timestamps]
    day_of_week = [ts.weekday() for ts in timestamps]

    # Create DataFrame
    data = pd.DataFrame({
        'velocity': velocity,
        'acceleration': acceleration,
        'distance_from_itinerary': distance,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week
    })

    # Create target labels (1 for normal, -1 for anomaly)
    labels = np.concatenate([np.ones(n_normal), -1 * np.ones(n_anomaly)])

    return data, labels


def train_and_save_model():
    """Train the anomaly detection model and save it to disk"""
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(10000)

    print("Training ensemble anomaly detection model...")
    # Create pipeline with scaling and ensemble model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('detector', EnsembleAnomalyDetector(contamination=0.1))
    ])

    # Train the model
    pipeline.fit(X)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the entire pipeline
    joblib.dump(pipeline, 'models/anomaly_detection_model.joblib')
    print("Model saved to models/anomaly_detection_model.joblib")

    # Also save the scaler separately for potential use elsewhere
    joblib.dump(pipeline.named_steps['scaler'], 'models/scaler.joblib')
    print("Scaler saved to models/scaler.joblib")

    # Evaluate the model
    predictions, scores = pipeline.named_steps['detector'].predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy:.4f}")

    return pipeline


if __name__ == '__main__':
    train_and_save_model()