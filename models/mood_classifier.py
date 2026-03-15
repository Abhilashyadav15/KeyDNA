"""
KeyDNA — Layer 1: Mood Classifier

Detects emotional state BEFORE authenticating.
Routes to the correct mood-specific profile.

Fixes applied:
- Fix #1: Mood changes rhythm → classify mood first
- Fix #2: Wide cluster → tight mood-separated clusters
"""

import numpy as np
import os
import pickle
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


MOODS = ['RELAXED', 'FOCUSED', 'STRESSED', 'TIRED']

MOOD_CHARACTERISTICS = {
    'RELAXED': {
        'avg_speed':      (100, 130),   # ms — slower, comfortable
        'speed_variance': (8,   18),    # low variance
        'avg_dwell':      (75,  100),   # longer holds
        'dwell_variance': (8,   18),
        'error_rate':     (0.0, 0.02),  # few errors
        'acceleration':   (-0.1, 0.1),  # flat curve
    },
    'FOCUSED': {
        'avg_speed':      (75,  100),   # ms — efficient
        'speed_variance': (6,   14),    # consistent
        'avg_dwell':      (60,  80),
        'dwell_variance': (6,   14),
        'error_rate':     (0.0, 0.015), # very few errors
        'acceleration':   (-0.15, 0.05),# slight deceleration
    },
    'STRESSED': {
        'avg_speed':      (55,  80),    # ms — fast, rushing
        'speed_variance': (20,  40),    # erratic
        'avg_dwell':      (40,  65),    # short holds
        'dwell_variance': (15,  30),
        'error_rate':     (0.05, 0.15), # more errors
        'acceleration':   (0.05, 0.25), # speeding up
    },
    'TIRED': {
        'avg_speed':      (130, 180),   # ms — slow
        'speed_variance': (12,  25),    # moderate variance
        'avg_dwell':      (90,  130),   # long holds
        'dwell_variance': (12,  25),
        'error_rate':     (0.03, 0.08), # some errors
        'acceleration':   (-0.2, 0.0),  # slowing down
    },
}


class MoodClassifier:
    """
    Layer 1: Classifies user's emotional state from typing behavior.
    Uses Random Forest trained on mood-labeled keystroke samples.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_accuracy = 0.0

    def generate_training_data(self, samples_per_mood: int = 300,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic mood training data based on known
        physiological typing characteristics per mood.
        In real deployment: user types in different moods during enrollment.
        """
        import random
        random.seed(seed)
        rng = np.random.RandomState(seed)

        X, y = [], []

        for mood_idx, mood in enumerate(MOODS):
            chars = MOOD_CHARACTERISTICS[mood]

            for _ in range(samples_per_mood):
                # Sample each feature from mood-specific range
                avg_speed = rng.uniform(*chars['avg_speed'])
                speed_var = rng.uniform(*chars['speed_variance'])
                avg_dwell = rng.uniform(*chars['avg_dwell'])
                dwell_var = rng.uniform(*chars['dwell_variance'])
                error_rate = rng.uniform(*chars['error_rate'])
                acceleration = rng.uniform(*chars['acceleration'])

                # Add realistic noise
                noise_scale = 0.08
                avg_speed    += rng.normal(0, avg_speed * noise_scale)
                speed_var    += rng.normal(0, speed_var * noise_scale)
                avg_dwell    += rng.normal(0, avg_dwell * noise_scale)
                dwell_var    += rng.normal(0, dwell_var * noise_scale)
                error_rate    = max(0, error_rate + rng.normal(0, 0.01))
                acceleration += rng.normal(0, 0.05)

                X.append([avg_speed, speed_var, avg_dwell,
                          dwell_var, error_rate, acceleration])
                y.append(mood_idx)

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray = None,
              y: np.ndarray = None,
              samples_per_mood: int = 300) -> float:
        """
        Train mood classifier.
        If no data provided, generates synthetic training data.
        Returns cross-validation accuracy.
        """
        if X is None or y is None:
            X, y = self.generate_training_data(samples_per_mood)

        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation accuracy
        scores = cross_val_score(self.model, X_scaled, y, cv=5)
        self.training_accuracy = scores.mean()

        # Train on full data
        self.model.fit(X_scaled, y)
        self.is_trained = True

        return self.training_accuracy

    def predict(self, mood_features: np.ndarray) -> Tuple[str, float]:
        """
        Predict mood from 6 mood features.
        Returns (mood_label, confidence).
        Confidence threshold: >0.70 → use mood-specific cluster
                              <0.70 → check all clusters (Fix #2)
        """
        if not self.is_trained:
            return 'RELAXED', 0.5

        if mood_features.ndim == 1:
            mood_features = mood_features.reshape(1, -1)

        scaled = self.scaler.transform(mood_features)
        proba = self.model.predict_proba(scaled)[0]
        mood_idx = np.argmax(proba)
        confidence = proba[mood_idx]

        return MOODS[mood_idx], float(confidence)

    def predict_all_probabilities(self, mood_features: np.ndarray) -> dict:
        """Returns probability for each mood class."""
        if not self.is_trained:
            return {m: 0.25 for m in MOODS}

        if mood_features.ndim == 1:
            mood_features = mood_features.reshape(1, -1)

        scaled = self.scaler.transform(mood_features)
        proba = self.model.predict_proba(scaled)[0]

        return {MOODS[i]: float(proba[i]) for i in range(len(MOODS))}

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model,
                        'scaler': self.scaler,
                        'accuracy': self.training_accuracy,
                        'trained': self.is_trained}, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.training_accuracy = data['accuracy']
        self.is_trained = data['trained']
        return True
