"""
KeyDNA — Evaluation & Benchmarking (v2 — Unified Model)

Tests the unified authentication model:
  - FAR (False Acceptance Rate)
  - FRR (False Rejection Rate)
  - EER (Equal Error Rate)
  - Replay attack detection rate
  - Adaptive attack detection
  - Mimicry resistance (sequence features)

No mood classifier. No KNN. Hybrid distance scoring only.
"""

import numpy as np
from typing import Dict, List
from core.capture import SimulatedCapture
from core.features import FeatureExtractor
from models.auth_model import UnifiedAuthModel


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
    Runs benchmarks for the KeyDNA unified authentication model.
    Uses simulated keystroke data for reproducible evaluation.
    """

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.capture   = SimulatedCapture()

    def run_full_benchmark(self) -> Dict:
        """
        Run complete benchmark evaluation.
        Returns results for KeyDNA unified model.
        """
        results = {}

        # Generate test user profiles
        user_profiles = self._generate_test_users(n_users=10)

        # Run KeyDNA unified model benchmark
        results['KeyDNA Unified'] = self._benchmark_unified(user_profiles)

        # Run replay attack test
        replay_rate = self._benchmark_replay_detection(user_profiles)
        results['KeyDNA Unified'].replay_caught = replay_rate

        return {k: v.to_dict() for k, v in results.items()}

    def _generate_test_users(self, n_users: int = 10) -> List[Dict]:
        """
        Generate synthetic user profiles with natural variation.
        Each user has a unique base typing speed and style.
        """
        users = []

        for user_id in range(n_users):
            # Each user has unique base speed (±20% personal variation)
            base_multiplier = 0.8 + (user_id * 0.04)
            base_var    = 12 + (user_id % 5) * 2
            base_dwell  = 75 + (user_id % 4) * 5
            base_error  = 0.01 + (user_id % 3) * 0.01

            enrollment_samples = []
            test_samples       = []

            # Generate 10 enrollment + 10 test samples with natural variation
            for i in range(20):
                # Natural variation across attempts
                variation = 1.0 + np.random.RandomState(user_id * 100 + i).normal(0, 0.08)

                events = self.capture.simulate_typing(
                    avg_speed_ms=100 * base_multiplier * variation,
                    variance_ms=base_var * max(0.5, variation),
                    dwell_ms=base_dwell * base_multiplier * variation,
                    n_keys=10,
                    error_rate=base_error,
                    seed=user_id * 1000 + i
                )
                features = self.extractor.extract(events)
                if features is not None:
                    if i < 10:
                        enrollment_samples.append(features)
                    else:
                        test_samples.append(features)

            users.append({
                'user_id':    user_id,
                'enrollment': enrollment_samples,
                'test':       test_samples,
                'multiplier': base_multiplier,
            })

        return users

    def _benchmark_unified(self, users: List[Dict]) -> BenchmarkResult:
        """Benchmark KeyDNA unified model with proper EER via threshold sweep."""
        result = BenchmarkResult('KeyDNA Unified')

        genuine_scores:  List[float] = []
        impostor_scores: List[float] = []

        for user in users:
            # Build unified model from enrollment samples
            model = UnifiedAuthModel()
            for sample in user['enrollment']:
                model.enroll(sample)

            if not model.is_ready:
                continue

            # ── Genuine tests (same user, new samples) ──
            for test_sample in user['test']:
                X = test_sample.reshape(1, -1)
                score = float(model._model.decision_function(X)[0])
                genuine_scores.append(score)

            # ── Impostor tests (different user attacking) ──
            impostor = users[(user['user_id'] + 1) % len(users)]
            for imp_sample in impostor['test']:
                X = imp_sample.reshape(1, -1)
                score = float(model._model.decision_function(X)[0])
                impostor_scores.append(score)

        # FAR/FRR at current operational threshold (THRESHOLD_ACCEPT = 0.0)
        from config import THRESHOLD_ACCEPT
        result.far = sum(1 for s in impostor_scores if s > THRESHOLD_ACCEPT) / max(len(impostor_scores), 1)
        result.frr = sum(1 for s in genuine_scores if s <= THRESHOLD_ACCEPT) / max(len(genuine_scores), 1)

        # Proper EER: sweep thresholds to find FAR == FRR crossing
        result.eer = self._compute_eer(genuine_scores, impostor_scores)
        result.total_tests = len(genuine_scores) + len(impostor_scores)
        return result

    @staticmethod
    def _compute_eer(genuine: List[float], impostor: List[float]) -> float:
        """Compute Equal Error Rate by sweeping thresholds on the DET curve."""
        if not genuine or not impostor:
            return 0.0

        thresholds = sorted(set(genuine + impostor))
        best_diff  = float('inf')
        eer        = 0.0

        for t in thresholds:
            far = sum(1 for s in impostor if s > t) / len(impostor)
            frr = sum(1 for s in genuine  if s <= t) / len(genuine)
            diff = abs(far - frr)
            if diff < best_diff:
                best_diff = diff
                eer = (far + frr) / 2.0

        return eer

    def _benchmark_replay_detection(self, users: List[Dict]) -> float:
        """
        Test replay attack detection.
        Simulates replay by setting consistency score > 0.97.
        """
        caught = 0
        total  = 0

        for user in users[:5]:
            model = UnifiedAuthModel()
            for sample in user['enrollment']:
                model.enroll(sample)

            if not model.is_ready:
                continue

            # Simulate replay: copy a real sample, make consistency too high
            for real_sample in user['enrollment'][:3]:
                replayed = real_sample.copy()
                replayed[11] = 0.985  # consistency score = too perfect
                total += 1

                model.reset_attack_tracking()
                auth = model.authenticate(replayed)
                if auth.get('replay_detected') or auth['decision'] == 'REJECT':
                    caught += 1

        return caught / max(total, 1)

    def _benchmark_mimicry_resistance(self, users: List[Dict]) -> Dict:
        """
        Test resistance to human mimicry attempts.
        Simulates attacker who matches global features but not sequence.
        """
        mimicry_rejected = 0
        total_mimicry    = 0

        for user in users[:5]:
            model = UnifiedAuthModel()
            for sample in user['enrollment']:
                model.enroll(sample)

            if not model.is_ready:
                continue

            # Simulate mimicry: copy global features from user,
            # but sequence features are from attacker
            impostor = users[(user['user_id'] + 3) % len(users)]
            for imp_sample in impostor['test'][:3]:
                mimic = user['enrollment'][0].copy()
                # Copy global features (0-16) from genuine user
                # But sequence features (17-26) from impostor
                mimic[17:] = imp_sample[17:]
                total_mimicry += 1

                model.reset_attack_tracking()
                auth = model.authenticate(mimic)
                if auth['decision'] != 'ACCEPT':
                    mimicry_rejected += 1

        rejection_rate = mimicry_rejected / max(total_mimicry, 1)
        return {
            'mimicry_rejection_rate': round(rejection_rate * 100, 2),
            'total_tests':           total_mimicry,
        }

    def run_comprehensive_report(self) -> Dict:
        """
        Run all benchmarks and return comprehensive report.
        """
        users = self._generate_test_users(n_users=10)

        unified_result = self._benchmark_unified(users)
        replay_rate    = self._benchmark_replay_detection(users)
        mimicry_result = self._benchmark_mimicry_resistance(users)

        unified_result.replay_caught = replay_rate

        return {
            'authentication': unified_result.to_dict(),
            'replay_detection': {
                'detection_rate': round(replay_rate * 100, 2),
            },
            'mimicry_resistance': mimicry_result,
        }
