# ensemble_model.py
import numpy as np
from sklearn.ensemble import IsolationForest

class EnsembleAnomalyDetector:
    def __init__(self):
        self.models = [
            IsolationForest(n_estimators=100, contamination=0.05, random_state=42),
            IsolationForest(n_estimators=150, contamination=0.05, random_state=52)
        ]

    def fit(self, X):
        for model in self.models:
            model.fit(X)

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        majority_vote = np.apply_along_axis(lambda x: 1 if np.sum(x == -1) > len(self.models)//2 else 0, axis=0, arr=preds)
        return majority_vote
