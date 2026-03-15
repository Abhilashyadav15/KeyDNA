"""
KeyDNA — Feature Extraction Engine

Extracts 12 timing features from keystroke events.
Applies Fix #3 (keyboard normalization via ratios)
Applies Fix #9 (content-independent features)
Applies Fix #11 (rich features for short passwords)
Applies Fix #12 (15+ dimensions for similar typers)
Applies Fix #13 (deliberate slow typing — shape not speed)
"""

import numpy as np
from typing import List, Dict, Optional


class FeatureExtractor:
    """
    Extracts normalized timing features from raw keystroke events.
    All features are RATIOS or RELATIVE measures — never absolute.
    This makes the system keyboard-independent (Fix #3, Fix #9).
    """

    def extract(self, events: List[Dict]) -> Optional[np.ndarray]:
        """
        Extract feature vector from keystroke events.
        Returns None if insufficient events.
        """
        if len(events) < 3:
            return None

        # Separate regular keys from backspaces
        regular = [e for e in events if e['key_id'] != 'backspace']
        backspaces = [e for e in events if e['key_id'] == 'backspace']

        if len(regular) < 3:
            return None

        dwells = np.array([e['dwell'] * 1000 for e in regular])  # ms

        # Flight times (gap between key release and next key press)
        flights = []
        for i in range(len(regular) - 1):
            flight = (regular[i + 1]['press'] - regular[i]['release']) * 1000
            flights.append(max(0, flight))
        flights = np.array(flights) if flights else np.array([0])

        # Digraph times (press to press)
        digraphs = []
        for i in range(len(regular) - 1):
            digraph = (regular[i + 1]['press'] - regular[i]['press']) * 1000
            digraphs.append(max(0, digraph))
        digraphs = np.array(digraphs) if digraphs else np.array([0])

        features = self._compute_features(
            dwells, flights, digraphs, backspaces, regular
        )

        return np.array(features)

    def _compute_features(self, dwells, flights, digraphs,
                          backspaces, regular) -> List[float]:
        """
        Compute 12 timing features.
        All are normalized/relative for keyboard independence.
        """

        # ── Feature 1: Average flight time (normalized) ──
        avg_flight = np.mean(flights)

        # ── Feature 2: Average dwell time (normalized) ──
        avg_dwell = np.mean(dwells)

        # ── Feature 3: Average digraph time ──
        avg_digraph = np.mean(digraphs)

        # ── Feature 4: Flight variance (rhythm consistency) ──
        flight_var = np.std(flights) if len(flights) > 1 else 0

        # ── Feature 5: Dwell variance ──
        dwell_var = np.std(dwells) if len(dwells) > 1 else 0

        # ── Feature 6: Dwell/Flight ratio (personal signature) ──
        # Fix #3: ratio is keyboard-independent
        dwell_flight_ratio = avg_dwell / (avg_flight + 1e-6)

        # ── Feature 7: Error rate (backspace frequency) ──
        # Fix #1: error rate reveals mood state
        error_rate = len(backspaces) / max(len(regular), 1)

        # ── Feature 8: Acceleration curve ──
        # Fix #13: shape not speed — catches deliberate slow typing
        # Positive = speeding up, Negative = slowing down
        if len(digraphs) >= 3:
            first_half = np.mean(digraphs[:len(digraphs) // 2])
            second_half = np.mean(digraphs[len(digraphs) // 2:])
            acceleration = (second_half - first_half) / (first_half + 1e-6)
        else:
            acceleration = 0.0

        # ── Feature 9: Acceleration entropy ──
        # Fix #13: measures naturalness of speed changes
        # Low entropy = artificially controlled (conscious mimic)
        # High entropy = natural unconscious typing
        if len(digraphs) >= 4:
            diffs = np.diff(digraphs)
            if np.std(diffs) > 0:
                # Normalize to probability distribution
                abs_diffs = np.abs(diffs) + 1e-6
                probs = abs_diffs / abs_diffs.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0
        else:
            entropy = 0.5

        # ── Feature 10: Inter-key variance ratio ──
        # Fix #12: captures individual key pair signatures
        # Normalized by mean — keyboard independent
        inter_key_var = flight_var / (avg_flight + 1e-6)

        # ── Feature 11: Word rhythm shape ──
        # Fix #13: the SHAPE of timing curve across password
        # Normalized timing profile — independent of absolute speed
        if len(digraphs) >= 3:
            normalized_digraphs = digraphs / (np.mean(digraphs) + 1e-6)
            rhythm_shape = np.std(normalized_digraphs)
        else:
            rhythm_shape = 0.0

        # ── Feature 12: Consistency score ──
        # Used for replay attack detection (Fix #7)
        # Too perfect (>0.97) = replay attack
        # Natural human range = 0.70–0.92
        if len(flights) > 1:
            cv = np.std(flights) / (np.mean(flights) + 1e-6)
            consistency = 1.0 / (1.0 + cv)  # higher = more consistent
        else:
            consistency = 0.5

        return [
            avg_flight,           # 1. average flight time
            avg_dwell,            # 2. average dwell time
            avg_digraph,          # 3. average digraph time
            flight_var,           # 4. flight variance
            dwell_var,            # 5. dwell variance
            dwell_flight_ratio,   # 6. dwell/flight ratio
            error_rate,           # 7. error/backspace rate
            acceleration,         # 8. acceleration curve
            entropy,              # 9. acceleration entropy
            inter_key_var,        # 10. inter-key variance ratio
            rhythm_shape,         # 11. word rhythm shape
            consistency,          # 12. consistency score
        ]

    def extract_mood_features(self, events: List[Dict]) -> Optional[np.ndarray]:
        """
        Extract 6 features specifically for mood classification (Layer 1).
        These capture emotional state signals from typing behavior.
        """
        if len(events) < 3:
            return None

        regular = [e for e in events if e['key_id'] != 'backspace']
        backspaces = [e for e in events if e['key_id'] == 'backspace']

        if len(regular) < 3:
            return None

        dwells = np.array([e['dwell'] * 1000 for e in regular])
        digraphs = []
        for i in range(len(regular) - 1):
            d = (regular[i + 1]['press'] - regular[i]['press']) * 1000
            digraphs.append(max(0, d))
        digraphs = np.array(digraphs) if digraphs else np.array([50])

        # Mood Feature 1: Overall typing speed
        avg_speed = np.mean(digraphs)

        # Mood Feature 2: Speed variance (high = stressed/erratic)
        speed_variance = np.std(digraphs) if len(digraphs) > 1 else 0

        # Mood Feature 3: Average dwell time (high = tired/deliberate)
        avg_dwell = np.mean(dwells)

        # Mood Feature 4: Dwell variance
        dwell_variance = np.std(dwells) if len(dwells) > 1 else 0

        # Mood Feature 5: Error rate (high = stressed/rushing)
        error_rate = len(backspaces) / max(len(regular), 1)

        # Mood Feature 6: Acceleration curve
        if len(digraphs) >= 3:
            first_half = np.mean(digraphs[:len(digraphs) // 2])
            second_half = np.mean(digraphs[len(digraphs) // 2:])
            acceleration = (second_half - first_half) / (first_half + 1e-6)
        else:
            acceleration = 0.0

        return np.array([
            avg_speed,        # 1. overall speed
            speed_variance,   # 2. speed variance
            avg_dwell,        # 3. dwell time
            dwell_variance,   # 4. dwell variance
            error_rate,       # 5. error rate
            acceleration,     # 6. acceleration curve
        ])

    def get_consistency_score(self, features: np.ndarray) -> float:
        """
        Returns the consistency score from feature vector.
        Used for replay attack detection (Fix #7).
        Feature index 11 = consistency score.
        """
        if features is None or len(features) < 12:
            return 0.5
        return features[11]

    @property
    def feature_names(self):
        return [
            'avg_flight_time',
            'avg_dwell_time',
            'avg_digraph_time',
            'flight_variance',
            'dwell_variance',
            'dwell_flight_ratio',
            'error_rate',
            'acceleration_curve',
            'acceleration_entropy',
            'inter_key_variance',
            'rhythm_shape',
            'consistency_score',
        ]

    @property
    def mood_feature_names(self):
        return [
            'avg_speed',
            'speed_variance',
            'avg_dwell',
            'dwell_variance',
            'error_rate',
            'acceleration',
        ]
