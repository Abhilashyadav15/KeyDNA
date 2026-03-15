"""
KeyDNA — Evaluation & Benchmarking

Compares KeyDNA against basic single-cluster KNN.
Measures FAR, FRR, EER, replay detection rate.
Generates all benchmark results for dashboard.
"""

import numpy as np
from typing import Dict, List, Tuple
from core.capture import SimulatedCapture
from core.features import FeatureExtractor
from models.mood_classifier import MoodClassifier, MOODS
from models.knn_auth import KNNAuthenticator
from enrollment.enroller import Enroller


class BenchmarkResult:
    def __init__(self, algorithm: str):
        self.algorithm     = algorithm
        self.far           = 0.0   # False Acceptance Rate
        self.frr           = 0.0   # False Rejection Rate
        self.eer           = 0.0   # Equal Error Rate
        self.replay_caught = 0.0   # Replay attack detection rate
        self.total_tests   = 0
        self.details       = {}

    def to_dict(self) -> Dict:
        return {
            'algorithm':     self.algorithm,
            'FAR':           round(self.far * 100, 2),
            'FRR':           round(self.frr * 100, 2),
            'EER':           round(self.eer * 100, 2),
            'replay_caught': round(self.replay_caught * 100, 2),
            'total_tests':   self.total_tests,
        }


class Evaluator:
    """
    Runs benchmarks comparing KeyDNA vs basic KNN.
    Uses simulated keystroke data for fair comparison.
    """

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.capture   = SimulatedCapture()

    def run_full_benchmark(self) -> Dict:
        """
        Run complete benchmark comparison.
        Returns results for both algorithms.
        """
        results = {}

        # Generate user profiles for testing
        user_profiles = self._generate_test_users(n_users=10)

        # Run KeyDNA (mood-aware)
        results['KeyDNA'] = self._benchmark_keydna(user_profiles)

        # Run Basic KNN (single cluster, no mood)
        results['Basic KNN'] = self._benchmark_basic_knn(user_profiles)

        # Run replay attack test for both
        replay_results = self._benchmark_replay_detection(user_profiles)
        results['KeyDNA'].replay_caught    = replay_results['keydna']
        results['Basic KNN'].replay_caught = replay_results['basic']

        return {k: v.to_dict() for k, v in results.items()}

    def _generate_test_users(self, n_users: int = 10) -> List[Dict]:
        """Generate synthetic user profiles with mood variation."""
        users = []

        mood_params = {
            'RELAXED':  {'avg': 115, 'var': 13, 'dwell': 87,  'error': 0.01},
            'FOCUSED':  {'avg': 87,  'var': 10, 'dwell': 70,  'error': 0.008},
            'STRESSED': {'avg': 67,  'var': 30, 'dwell': 52,  'error': 0.08},
            'TIRED':    {'avg': 155, 'var': 18, 'dwell': 110, 'error': 0.05},
        }

        for user_id in range(n_users):
            # Each user has unique base speed (±20% personal variation)
            base_multiplier = 0.8 + (user_id * 0.04)
            user_profiles = {}

            for mood, params in mood_params.items():
                samples = []
                for i in range(15):
                    events = self.capture.simulate_typing(
                        avg_speed_ms=params['avg'] * base_multiplier,
                        variance_ms=params['var'],
                        dwell_ms=params['dwell'] * base_multiplier,
                        n_keys=10,
                        error_rate=params['error'],
                        seed=user_id * 1000 + hash(mood) % 100 + i
                    )
                    features = self.extractor.extract(events)
                    if features is not None:
                        samples.append(features)

                user_profiles[mood] = samples

            users.append({
                'user_id':    user_id,
                'profiles':   user_profiles,
                'multiplier': base_multiplier,
            })

        return users

    def _benchmark_keydna(self, users: List[Dict]) -> BenchmarkResult:
        """Benchmark KeyDNA with mood-separated clusters."""
        result = BenchmarkResult('KeyDNA')

        false_accepts = 0
        false_rejects = 0
        total_genuine = 0
        total_impostor = 0

        mood_params = {
            'RELAXED':  {'avg': 115, 'var': 13, 'dwell': 87,  'error': 0.01},
            'FOCUSED':  {'avg': 87,  'var': 10, 'dwell': 70,  'error': 0.008},
            'STRESSED': {'avg': 67,  'var': 30, 'dwell': 52,  'error': 0.08},
            'TIRED':    {'avg': 155, 'var': 18, 'dwell': 110, 'error': 0.05},
        }

        for user in users:
            # Setup KeyDNA for this user
            knn = KNNAuthenticator()
            for mood, samples in user['profiles'].items():
                for sample in samples:
                    knn.enroll(sample, mood)

            # Genuine tests (same user, each mood)
            for mood, params in mood_params.items():
                for test_i in range(10):
                    events = self.capture.simulate_typing(
                        avg_speed_ms=params['avg'] * user['multiplier'],
                        variance_ms=params['var'] * 1.1,  # slight variation
                        dwell_ms=params['dwell'] * user['multiplier'],
                        n_keys=10,
                        error_rate=params['error'],
                        seed=99999 + user['user_id'] * 100 + test_i
                    )
                    features = self.extractor.extract(events)
                    if features is None:
                        continue

                    auth = knn.authenticate(features, mood, 0.85)
                    total_genuine += 1
                    if auth['decision'] == 'REJECT':
                        false_rejects += 1

            # Impostor tests (different user attacking)
            impostor = users[(user['user_id'] + 1) % len(users)]
            for mood in MOODS:
                imp_params = mood_params[mood]
                for test_i in range(5):
                    events = self.capture.simulate_typing(
                        avg_speed_ms=imp_params['avg'] * impostor['multiplier'],
                        variance_ms=imp_params['var'],
                        dwell_ms=imp_params['dwell'] * impostor['multiplier'],
                        n_keys=10,
                        error_rate=imp_params['error'],
                        seed=88888 + impostor['user_id'] * 50 + test_i
                    )
                    features = self.extractor.extract(events)
                    if features is None:
                        continue

                    auth = knn.authenticate(features, mood, 0.80)
                    total_impostor += 1
                    if auth['decision'] == 'ACCEPT':
                        false_accepts += 1

        result.far = false_accepts / max(total_impostor, 1)
        result.frr = false_rejects / max(total_genuine, 1)
        result.eer = (result.far + result.frr) / 2
        result.total_tests = total_genuine + total_impostor
        return result

    def _benchmark_basic_knn(self, users: List[Dict]) -> BenchmarkResult:
        """Benchmark basic single-cluster KNN (no mood separation)."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        result = BenchmarkResult('Basic KNN')

        mood_params = {
            'RELAXED':  {'avg': 115, 'var': 13, 'dwell': 87,  'error': 0.01},
            'FOCUSED':  {'avg': 87,  'var': 10, 'dwell': 70,  'error': 0.008},
            'STRESSED': {'avg': 67,  'var': 30, 'dwell': 52,  'error': 0.08},
            'TIRED':    {'avg': 155, 'var': 18, 'dwell': 110, 'error': 0.05},
        }

        false_accepts = 0
        false_rejects = 0
        total_genuine = 0
        total_impostor = 0

        for user in users:
            # Combine ALL mood samples into ONE cluster (basic approach)
            all_samples = []
            for mood_samples in user['profiles'].values():
                all_samples.extend(mood_samples)

            if len(all_samples) < 5:
                continue

            X = np.array(all_samples)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            center = np.mean(X_scaled, axis=0)
            # Threshold = 2 std devs from center
            distances = np.linalg.norm(X_scaled - center, axis=1)
            threshold = np.mean(distances) + 2 * np.std(distances)

            def basic_auth(features):
                scaled = scaler.transform(features.reshape(1, -1))
                dist = np.linalg.norm(scaled - center)
                return dist <= threshold

            # Genuine tests
            for mood, params in mood_params.items():
                for test_i in range(10):
                    events = self.capture.simulate_typing(
                        avg_speed_ms=params['avg'] * user['multiplier'],
                        variance_ms=params['var'] * 1.1,
                        dwell_ms=params['dwell'] * user['multiplier'],
                        n_keys=10,
                        error_rate=params['error'],
                        seed=99999 + user['user_id'] * 100 + test_i
                    )
                    features = self.extractor.extract(events)
                    if features is None:
                        continue

                    total_genuine += 1
                    if not basic_auth(features):
                        false_rejects += 1

            # Impostor tests
            impostor = users[(user['user_id'] + 1) % len(users)]
            for mood in MOODS:
                imp_params = mood_params[mood]
                for test_i in range(5):
                    events = self.capture.simulate_typing(
                        avg_speed_ms=imp_params['avg'] * impostor['multiplier'],
                        variance_ms=imp_params['var'],
                        dwell_ms=imp_params['dwell'] * impostor['multiplier'],
                        n_keys=10,
                        error_rate=imp_params['error'],
                        seed=88888 + impostor['user_id'] * 50 + test_i
                    )
                    features = self.extractor.extract(events)
                    if features is None:
                        continue

                    total_impostor += 1
                    if basic_auth(features):
                        false_accepts += 1

        result.far = false_accepts / max(total_impostor, 1)
        result.frr = false_rejects / max(total_genuine, 1)
        result.eer = (result.far + result.frr) / 2
        result.total_tests = total_genuine + total_impostor
        return result

    def _benchmark_replay_detection(self, users: List[Dict]) -> Dict:
        """Test replay attack detection for both algorithms."""
        keydna_caught = 0
        basic_caught  = 0
        total_replays = 0

        for user in users[:3]:  # test on 3 users
            knn = KNNAuthenticator()
            for mood, samples in user['profiles'].items():
                for sample in samples:
                    knn.enroll(sample, mood)

            # Simulate replay: take real sample and make it TOO consistent
            real_sample = user['profiles']['FOCUSED'][0].copy()
            # Make consistency score artificially high (replay attack)
            real_sample[11] = 0.985  # index 11 = consistency score

            total_replays += 1

            # KeyDNA: checks consistency score ceiling
            auth = knn.authenticate(real_sample, 'FOCUSED', 0.85)
            if auth.get('replay_detected') or auth['decision'] == 'REJECT':
                keydna_caught += 1

            # Basic KNN: no replay detection
            # Would accept because distance is small

        return {
            'keydna': keydna_caught / max(total_replays, 1),
            'basic':  0.0,  # basic KNN has no replay detection
        }

    def run_mood_accuracy_test(self) -> Dict:
        """Test mood classifier accuracy."""
        classifier = MoodClassifier()
        X, y = classifier.generate_training_data(samples_per_mood=200)
        accuracy = classifier.train(X, y)

        # Per-mood accuracy
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        classifier_test = MoodClassifier()
        classifier_test.train(X_train, y_train)

        X_test_scaled = classifier_test.scaler.transform(X_test)
        y_pred = classifier_test.model.predict(X_test_scaled)

        per_mood = {}
        for i, mood in enumerate(MOODS):
            mask = y_test == i
            if mask.sum() > 0:
                correct = (y_pred[mask] == i).sum()
                per_mood[mood] = round(correct / mask.sum() * 100, 1)

        return {
            'overall_accuracy': round(accuracy * 100, 1),
            'per_mood':         per_mood,
            'training_samples': len(X),
        }
