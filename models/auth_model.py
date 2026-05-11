"""
KeyDNA — Unified Authentication Model (One-Class SVM)

ARCHITECTURE:
  Single unified model — NO mood classifier.
  Uses One-Class SVM to detect anomalies (impostors).
  Uses 27-feature vector (typing rhythm only, no time-of-day).

DECISION RULES:
  Score > 0    → ACCEPT
  0 >= Score > -0.01 → RETRY
  Score <= -0.01 → REJECT
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.svm import OneClassSVM

from config import (
    TOTAL_DIM,
    MIN_ENROLLMENT_SAMPLES,
    MAX_ENROLLMENT_SAMPLES,
    SVM_KERNEL,
    SVM_GAMMA,
    SVM_NU,
    THRESHOLD_ACCEPT,
    THRESHOLD_RETRY,
    REPLAY_CONSISTENCY_THRESHOLD,
    ADAPTIVE_ATTACK_WINDOW,
)


class UnifiedAuthModel:
    """
    Single unified authentication model.
    Uses One-Class SVM for highly secure bounding of typing rhythm.
    """

    def __init__(self) -> None:
        # ── Enrollment data ──
        self.samples: List[np.ndarray] = []
        self._model: Optional[OneClassSVM] = None
        self._is_fitted: bool = False
        self._threshold_accept: float = THRESHOLD_ACCEPT
        self._threshold_retry: float = THRESHOLD_RETRY

        # ── Anti-attack state ──
        self._recent_scores: List[float] = []
        self._failed_attempts: int = 0

    # ══════════════════════════════════════════════════════════════
    # ENROLLMENT & TRAINING
    # ══════════════════════════════════════════════════════════════

    def enroll(self, features: np.ndarray) -> bool:
        """Enroll a single feature vector. Retrains if threshold met."""
        if features is None or len(features) != TOTAL_DIM:
            return False

        self.samples.append(features.copy())

        # Retrain model if we have enough samples
        if len(self.samples) >= MIN_ENROLLMENT_SAMPLES:
            self._train_model()

        return True

    def _train_model(self) -> None:
        """Train the One-Class SVM on enrolled samples."""
        X: np.ndarray = np.array(self.samples)

        self._model = OneClassSVM(
            kernel=SVM_KERNEL, gamma=SVM_GAMMA, nu=SVM_NU
        )
        self._model.fit(X)
        self._is_fitted = True

        self._threshold_accept = THRESHOLD_ACCEPT
        self._threshold_retry = THRESHOLD_RETRY

    @property
    def is_ready(self) -> bool:
        """True if model is fitted with enough enrollment samples."""
        return self._is_fitted and len(self.samples) >= MIN_ENROLLMENT_SAMPLES

    @property
    def sample_count(self) -> int:
        """Number of enrolled samples."""
        return len(self.samples)

    @property
    def enrollment_progress(self) -> float:
        """Progress towards recommended enrollment (0.0–1.0)."""
        return min(1.0, self.sample_count / MAX_ENROLLMENT_SAMPLES)

    # ══════════════════════════════════════════════════════════════
    # AUTHENTICATION
    # ══════════════════════════════════════════════════════════════

    def authenticate(self, features: np.ndarray) -> Dict:
        """
        Authenticate a feature vector against the enrolled model.

        Returns a dict with 'decision', 'reason', 'replay_detected',
        'fallback_required'. NEVER exposes internal scores.
        """
        result: Dict = {
            'decision': 'REJECT',
            'reason': 'Authentication failed.',
            'replay_detected': False,
            'fallback_required': False,
        }

        if not self.is_ready:
            result['reason'] = (
                f'Insufficient enrollment data. '
                f'{self.sample_count}/{MIN_ENROLLMENT_SAMPLES} samples enrolled.'
            )
            return result

        if features is None or len(features) != TOTAL_DIM:
            result['reason'] = 'Invalid feature vector.'
            return result

        # ── Step 1: Replay attack detection ──
        consistency: float = float(features[11])  # consistency_score
        if consistency >= REPLAY_CONSISTENCY_THRESHOLD:
            result['decision'] = 'REJECT'
            result['replay_detected'] = True
            result['reason'] = 'Authentication failed.'
            self._record_attempt(-10.0)
            return result

        # ── Step 2: Compute SVM Decision Score ──
        X_test: np.ndarray = features.reshape(1, -1)
        score: float = float(self._model.decision_function(X_test)[0])

        # ── Step 3: Adaptive attack detection ──
        if self._detect_adaptive_attack(score):
            result['decision'] = 'REJECT'
            result['reason'] = 'Authentication failed.'
            self._record_attempt(score)
            return result

        self._record_attempt(score)

        # ── Step 4: Apply thresholds ──
        if score > self._threshold_accept:
            result['decision'] = 'ACCEPT'
            result['reason'] = 'Authentication successful.'
        elif score > self._threshold_retry:
            result['decision'] = 'RETRY'
            result['reason'] = 'Authentication failed.'
        else:
            result['decision'] = 'REJECT'
            result['reason'] = 'Authentication failed.'

        return result

    # ══════════════════════════════════════════════════════════════
    # ANTI-ATTACK: Adaptive Attack Detection
    # ══════════════════════════════════════════════════════════════

    def _detect_adaptive_attack(self, current_score: float) -> bool:
        """Detect if an attacker is progressively improving scores."""
        if len(self._recent_scores) < ADAPTIVE_ATTACK_WINDOW:
            return False

        recent: List[float] = self._recent_scores[-ADAPTIVE_ATTACK_WINDOW:]

        # For SVM, improving means score is INCREASING
        is_improving: bool = all(
            recent[i] < recent[i + 1] for i in range(len(recent) - 1)
        )

        if is_improving and current_score > recent[-1]:
            return True

        return False

    def _record_attempt(self, score: float) -> None:
        """Record an attempt score for adaptive attack tracking."""
        self._recent_scores.append(score)
        max_history: int = ADAPTIVE_ATTACK_WINDOW + 5
        if len(self._recent_scores) > max_history:
            self._recent_scores = self._recent_scores[-max_history:]

    def reset_attack_tracking(self) -> None:
        """Reset adaptive attack detection state."""
        self._recent_scores.clear()
        self._failed_attempts = 0

    # ══════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════

    def get_samples_serializable(self) -> List[List[float]]:
        """Convert samples to JSON-serializable format."""
        return [s.tolist() for s in self.samples]

    def load_samples(self, samples_list: List[List[float]]) -> None:
        """Load samples from serialized format and retrain."""
        self.samples = [np.array(s) for s in samples_list]
        if len(self.samples) >= MIN_ENROLLMENT_SAMPLES:
            self._train_model()

    def save_to_dict(self) -> Dict:
        """Serialize model state to dict."""
        return {
            'samples': self.get_samples_serializable(),
            'sample_count': self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'UnifiedAuthModel':
        """Reconstruct model from dict."""
        model: UnifiedAuthModel = cls()
        if 'samples' in data:
            model.load_samples(data['samples'])
        return model

    def get_enrollment_summary(self) -> Dict:
        """Return enrollment status summary."""
        return {
            'samples': self.sample_count,
            'ready': self.is_ready,
            'progress': self.enrollment_progress,
            'min_needed': MIN_ENROLLMENT_SAMPLES,
            'recommended': MAX_ENROLLMENT_SAMPLES,
        }
