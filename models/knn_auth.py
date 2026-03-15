"""
KeyDNA — Layer 2: KNN Authenticator

Per-mood KNN authentication with replay attack detection.

Fixes applied:
- Fix #2:  Wide cluster → tight mood-separated clusters
- Fix #4:  Cold start → progressive enrollment
- Fix #6:  Replay attack → confidence > 97% rejected
- Fix #7:  Not strong alone → confidence thresholds
- Fix #11: Short passwords → rich 12-dimension features
- Fix #12: Similar typing styles → 15+ dimensions
"""

import numpy as np
import os
import pickle
from typing import Tuple, Optional, Dict, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Authentication decision thresholds
THRESHOLD_ACCEPT      = 0.60   # below this = impostor
THRESHOLD_OTP         = 0.95   # above this but below replay = borderline
THRESHOLD_REPLAY      = 0.97   # above this = replay attack (Fix #6)
MIN_ENROLLMENT_SAMPLES = 5     # minimum before auth works
RECOMMENDED_SAMPLES    = 15    # target per mood for good accuracy


class MoodProfile:
    """
    Stores and manages KNN model for a single mood cluster.
    Each mood has its own tight, independent profile.
    """

    def __init__(self, mood: str):
        self.mood = mood
        self.samples: List[np.ndarray] = []
        self.scaler = StandardScaler()
        self.knn: Optional[KNeighborsClassifier] = None
        self.is_ready = False

    def add_sample(self, features: np.ndarray):
        """Add enrollment sample. Progressive enrollment (Fix #4)."""
        self.samples.append(features.copy())
        # Retrain after each new sample if we have minimum
        if len(self.samples) >= MIN_ENROLLMENT_SAMPLES:
            self._train()

    def _train(self):
        """Train KNN on current samples."""
        X = np.array(self.samples)
        # K = min(5, samples//2) for small datasets
        k = min(5, max(1, len(self.samples) // 2))

        X_scaled = self.scaler.fit_transform(X)

        # KNN: user samples = class 1, need negative samples for class 0
        # Using distance threshold approach instead of binary classification
        self.knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='euclidean'
        )
        # Train with all samples as "authentic" class
        y = np.ones(len(X_scaled))
        self.knn.fit(X_scaled, y)
        self.is_ready = True

    def compute_distance(self, features: np.ndarray) -> float:
        """
        Compute normalized distance from profile center.
        Returns 0.0 (identical) to 1.0+ (very different).
        Uses adaptive normalization based on intra-cluster spread.
        """
        if not self.is_ready or len(self.samples) < MIN_ENROLLMENT_SAMPLES:
            return 1.0

        profile_matrix = np.array(self.samples)
        scaled_profile = self.scaler.transform(profile_matrix)
        scaled_query   = self.scaler.transform(features.reshape(1, -1))

        # Distance to each enrolled sample
        distances = np.linalg.norm(scaled_profile - scaled_query, axis=1)

        # Use mean of k-nearest distances
        k = min(5, len(distances))
        knn_distances = np.sort(distances)[:k]
        mean_distance = np.mean(knn_distances)

        # Adaptive normalization: scale by intra-cluster spread
        # This makes threshold relative to how tight the cluster is
        intra_distances = []
        for i in range(len(scaled_profile)):
            for j in range(i + 1, len(scaled_profile)):
                d = np.linalg.norm(scaled_profile[i] - scaled_profile[j])
                intra_distances.append(d)

        if intra_distances:
            intra_spread = np.mean(intra_distances)
            # Normalize: distance relative to cluster spread
            normalized = mean_distance / (intra_spread * 2.5 + 1e-6)
        else:
            normalized = mean_distance / 3.0

        return float(min(normalized, 2.0))  # cap at 2.0

    def compute_confidence(self, features: np.ndarray) -> float:
        """
        Convert distance to confidence score (0–1).
        Higher = more confident this is the enrolled user.
        """
        distance = self.compute_distance(features)
        # Sigmoid-like mapping: distance 0 → confidence 1.0
        #                       distance 1 → confidence 0.5
        #                       distance 2 → confidence 0.12
        confidence = 1.0 / (1.0 + distance ** 1.5)
        return float(confidence)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def enrollment_progress(self) -> float:
        """Progress toward recommended enrollment (Fix #4)."""
        return min(1.0, self.sample_count / RECOMMENDED_SAMPLES)


class KNNAuthenticator:
    """
    Layer 2: Per-mood KNN authentication.
    Routes to correct mood cluster based on Layer 1 output.
    Applies replay attack detection via confidence ceiling.
    """

    def __init__(self):
        from models.mood_classifier import MOODS
        self.profiles: Dict[str, MoodProfile] = {
            mood: MoodProfile(mood) for mood in MOODS
        }
        self.MOODS = MOODS

    def enroll(self, features: np.ndarray, mood: str):
        """
        Add enrollment sample to mood-specific profile.
        Progressive enrollment — system improves with each sample (Fix #4).
        """
        if mood not in self.profiles:
            return False
        self.profiles[mood].add_sample(features)
        return True

    def authenticate(self,
                     features: np.ndarray,
                     mood: str,
                     mood_confidence: float) -> Dict:
        """
        Authenticate against mood-specific profile.

        Decision logic:
        - mood_confidence > 0.70 → check mood-specific cluster only
        - mood_confidence < 0.70 → check all clusters (Fix #2)
        - confidence > 0.97 → replay attack detected (Fix #6)
        - confidence < 0.60 → impostor
        - confidence 0.95–0.97 → borderline, request OTP
        - confidence 0.60–0.95 → authentic

        Fix #7: Confidence > 97% = replay attack (too perfect)
        """
        result = {
            'decision':         'REJECT',
            'confidence':       0.0,
            'mood_used':        mood,
            'checked_clusters': [],
            'replay_detected':  False,
            'otp_required':     False,
            'reason':           '',
            'enrollment_status': {}
        }

        # Check enrollment status
        for m, profile in self.profiles.items():
            result['enrollment_status'][m] = {
                'samples':  profile.sample_count,
                'ready':    profile.is_ready,
                'progress': profile.enrollment_progress
            }

        # Determine which clusters to check
        if mood_confidence >= 0.70:
            # High mood confidence → check specific cluster only
            clusters_to_check = [mood]
            result['reason'] = f'Checking {mood} cluster (confidence {mood_confidence:.0%})'
        else:
            # Low mood confidence → check all clusters (Fix #2)
            clusters_to_check = self.MOODS
            result['reason'] = f'Low mood confidence ({mood_confidence:.0%}) — checking all clusters'

        result['checked_clusters'] = clusters_to_check

        # Compute confidence against each relevant cluster
        best_confidence = 0.0
        best_mood = None

        for cluster_mood in clusters_to_check:
            profile = self.profiles[cluster_mood]
            if not profile.is_ready:
                continue

            conf = profile.compute_confidence(features)
            if conf > best_confidence:
                best_confidence = conf
                best_mood = cluster_mood

        result['confidence'] = best_confidence
        result['mood_used'] = best_mood or mood

        # ── Apply decision thresholds ──

        # Fix #6: Replay attack detection
        # Confidence > 97% = too perfect = replay attack
        if best_confidence >= THRESHOLD_REPLAY:
            result['decision'] = 'REJECT'
            result['replay_detected'] = True
            result['reason'] = (
                f'Replay attack detected — confidence {best_confidence:.1%} '
                f'exceeds human maximum (97%). No human types with perfect consistency.'
            )
            return result

        # Impostor — too different
        if best_confidence < THRESHOLD_ACCEPT:
            result['decision'] = 'REJECT'
            result['reason'] = (
                f'Confidence {best_confidence:.1%} below threshold ({THRESHOLD_ACCEPT:.0%}). '
                f'Rhythm does not match enrolled profiles.'
            )
            return result

        # Borderline — request OTP
        if best_confidence >= THRESHOLD_OTP:
            result['decision'] = 'OTP_REQUIRED'
            result['otp_required'] = True
            result['reason'] = (
                f'Borderline confidence {best_confidence:.1%}. '
                f'OTP verification required as additional factor.'
            )
            return result

        # Authentic
        result['decision'] = 'ACCEPT'
        result['reason'] = (
            f'Authenticated via {best_mood} profile '
            f'with confidence {best_confidence:.1%}'
        )
        return result

    def authenticate_all_clusters(self, features: np.ndarray) -> Dict:
        """Check against all clusters — for low mood confidence scenarios."""
        from models.mood_classifier import MOODS
        results = {}
        for mood in MOODS:
            profile = self.profiles[mood]
            if profile.is_ready:
                results[mood] = profile.compute_confidence(features)
        return results

    def get_enrollment_summary(self) -> Dict:
        """Returns enrollment status for all mood profiles."""
        summary = {}
        for mood, profile in self.profiles.items():
            summary[mood] = {
                'samples':    profile.sample_count,
                'ready':      profile.is_ready,
                'progress':   profile.enrollment_progress,
                'min_needed': MIN_ENROLLMENT_SAMPLES,
                'recommended': RECOMMENDED_SAMPLES,
            }
        return summary

    def total_enrolled_samples(self) -> int:
        return sum(p.sample_count for p in self.profiles.values())

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'profiles': self.profiles}, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.profiles = data['profiles']
        return True
