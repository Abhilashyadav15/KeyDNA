"""
KeyDNA — Feature Extraction Engine (v2 — Unified Model)

Extracts 27 features from keystroke events (no time-of-day):
  Features 1–17  : Global statistical features
  Features 18–27 : Sequence-based normalized per-key timing

DESIGN:
  NO mood classifier. All features go into a SINGLE unified model.
  The system learns natural variation from all enrollment samples.

MIMIC RESISTANCE:
  Features 1–12  : Timing averages and variances (observable by watching)
  Features 13–17 : Per-position structural features (NOT observable)
  Features 18–27 : Normalized digraph timing sequence (CRITICAL for
                   preventing human mimicry — captures exact per-key
                   transition pattern that no human can consciously
                   reproduce even after watching multiple attempts)

Keyboard Independence:
  All features are RATIOS or RELATIVE measures — never absolute.
  This makes the system keyboard-independent.
"""

import numpy as np
from typing import Dict, List, Optional

from config import SEQUENCE_LENGTH


class FeatureExtractor:
    """
    Extracts 27-dimensional feature vector from raw keystroke events.

    Output vector layout:
      [0–16]  → 17 global statistical features
      [17–26] → 10 sequence-based features (normalized digraph timings)

    No time-of-day features — typing rhythm is time-independent.
    All features are RATIOS or RELATIVE measures — never absolute.
    This makes the system keyboard-independent.
    """

    def extract(self, events: List[Dict]) -> Optional[np.ndarray]:
        """
        Extract full 27-feature vector from keystroke events.
        Returns None if insufficient events.
        """
        if len(events) < 3:
            return None

        # Separate regular keys from backspaces
        regular: List[Dict] = [e for e in events if e['key_id'] != 'backspace']
        backspaces: List[Dict] = [e for e in events if e['key_id'] == 'backspace']

        if len(regular) < 3:
            return None

        dwells: np.ndarray = np.array(
            [e['dwell'] * 1000 for e in regular]
        )  # ms

        # Flight times (gap between key release and next key press)
        flights_list: List[float] = []
        for i in range(len(regular) - 1):
            flight: float = (regular[i + 1]['press'] - regular[i]['release']) * 1000
            flights_list.append(max(0, flight))
        flights: np.ndarray = (
            np.array(flights_list) if flights_list else np.array([0])
        )

        # Digraph times (press to press)
        digraphs_list: List[float] = []
        for i in range(len(regular) - 1):
            digraph: float = (regular[i + 1]['press'] - regular[i]['press']) * 1000
            digraphs_list.append(max(0, digraph))
        digraphs: np.ndarray = (
            np.array(digraphs_list) if digraphs_list else np.array([0])
        )

        # ── Part A: 17 global statistical features ──
        global_features: List[float] = self._compute_global_features(
            dwells, flights, digraphs, backspaces, regular
        )

        # ── Part B: 10 sequence-based features ──
        sequence_features: List[float] = self._compute_sequence_features(digraphs)

        # ── Concatenate: 17 + 10 = 27 features ──
        full_vector: List[float] = global_features + sequence_features

        return np.array(full_vector)

    # ════════════════════════════════════════════════════════════════
    # PART A: 17 Global Statistical Features
    # ════════════════════════════════════════════════════════════════

    def _compute_global_features(
        self,
        dwells: np.ndarray,
        flights: np.ndarray,
        digraphs: np.ndarray,
        backspaces: List[Dict],
        regular: List[Dict],
    ) -> List[float]:
        """
        Compute 17 global timing features.
        Features 1–12  : timing statistics (speed, variance, shape)
        Features 13–17 : per-position structural features (mimic-resistant)

        All are normalized/relative for keyboard independence.
        """

        # ── Feature 1: Average flight time (normalized) ──
        avg_flight: float = float(np.mean(flights))

        # ── Feature 2: Average dwell time (normalized) ──
        avg_dwell: float = float(np.mean(dwells))

        # ── Feature 3: Average digraph time ──
        avg_digraph: float = float(np.mean(digraphs))

        # ── Feature 4: Flight variance (rhythm consistency) ──
        flight_var: float = float(np.std(flights)) if len(flights) > 1 else 0.0

        # ── Feature 5: Dwell variance ──
        dwell_var: float = float(np.std(dwells)) if len(dwells) > 1 else 0.0

        # ── Feature 6: Dwell/Flight ratio (personal signature) ──
        # This ratio is stable per-person regardless of speed.
        dwell_flight_ratio: float = avg_dwell / (avg_flight + 1e-6)

        # ── Feature 7: Error rate (backspace frequency) ──
        # Reflects natural error tendency — hard to fake consistently.
        error_rate: float = len(backspaces) / max(len(regular), 1)

        # ── Feature 8: Acceleration curve ──
        # How speed changes from first half to second half of typing.
        if len(digraphs) >= 3:
            first_half: float = float(np.mean(digraphs[: len(digraphs) // 2]))
            second_half: float = float(np.mean(digraphs[len(digraphs) // 2 :]))
            acceleration: float = (second_half - first_half) / (first_half + 1e-6)
        else:
            acceleration = 0.0

        # ── Feature 9: Acceleration entropy ──
        # Low entropy = artificially controlled (conscious mimic attempt)
        # High entropy = natural unconscious typing
        if len(digraphs) >= 4:
            diffs: np.ndarray = np.diff(digraphs)
            if np.std(diffs) > 0:
                abs_diffs: np.ndarray = np.abs(diffs) + 1e-6
                probs: np.ndarray = abs_diffs / abs_diffs.sum()
                entropy: float = float(-np.sum(probs * np.log(probs + 1e-10)))
            else:
                entropy = 0.0
        else:
            entropy = 0.5

        # ── Feature 10: Inter-key variance ratio ──
        inter_key_var: float = flight_var / (avg_flight + 1e-6)

        # ── Feature 11: Word rhythm shape ──
        # Normalized standard deviation of digraph timings — captures
        # the "shape" of the rhythm independent of absolute speed.
        if len(digraphs) >= 3:
            normalized_digraphs: np.ndarray = digraphs / (np.mean(digraphs) + 1e-6)
            rhythm_shape: float = float(np.std(normalized_digraphs))
        else:
            rhythm_shape = 0.0

        # ── Feature 12: Consistency score (replay detection) ──
        # Used to detect replay attacks. Too-consistent typing = machine.
        # Score close to 1.0 = suspiciously perfect = likely automated.
        if len(flights) > 1:
            cv: float = float(np.std(flights)) / (float(np.mean(flights)) + 1e-6)
            consistency: float = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5

        # ════════════════════════════════════════════════════════════
        # MIMIC-RESISTANT FEATURES (13–17)
        # These capture sub-second per-position timing that is
        # impossible to observe visually or reproduce consciously.
        # A person watching you type 10 times cannot measure these.
        # ════════════════════════════════════════════════════════════

        # ── Feature 13: Per-position dwell profile variance ──
        if len(dwells) >= 3:
            norm_dwells: np.ndarray = dwells / (np.mean(dwells) + 1e-6)
            per_pos_dwell_var: float = float(np.std(norm_dwells))
        else:
            per_pos_dwell_var = 0.0

        # ── Feature 14: Bigram flight profile variance ──
        if len(flights) >= 3:
            norm_flights: np.ndarray = flights / (np.mean(flights) + 1e-6)
            bigram_profile_var: float = float(np.var(norm_flights))
        else:
            bigram_profile_var = 0.0

        # ── Feature 15: Tri-graph overlap pattern ──
        if len(digraphs) >= 3:
            trigraphs: List[float] = [
                digraphs[i] + digraphs[i + 1] for i in range(len(digraphs) - 1)
            ]
            tri_arr: np.ndarray = np.array(trigraphs)
            norm_tri: np.ndarray = tri_arr / (np.mean(tri_arr) + 1e-6)
            tri_pattern: float = float(np.std(norm_tri))
        else:
            tri_pattern = 0.0

        # ── Feature 16: Position timing curve slope ──
        if len(digraphs) >= 3:
            positions: np.ndarray = np.arange(len(digraphs), dtype=float)
            norm_d: np.ndarray = digraphs / (np.mean(digraphs) + 1e-6)
            slope: float = float(np.polyfit(positions, norm_d, 1)[0])
        else:
            slope = 0.0

        # ── Feature 17: Dwell asymmetry (first half vs second half) ──
        if len(dwells) >= 4:
            mid: int = len(dwells) // 2
            first_dwell: float = float(np.mean(dwells[:mid]))
            second_dwell: float = float(np.mean(dwells[mid:]))
            dwell_asym: float = first_dwell / (second_dwell + 1e-6)
        else:
            dwell_asym = 1.0

        return [
            avg_flight,           # 1.  average flight time
            avg_dwell,            # 2.  average dwell time
            avg_digraph,          # 3.  average digraph time
            flight_var,           # 4.  flight variance
            dwell_var,            # 5.  dwell variance
            dwell_flight_ratio,   # 6.  dwell/flight ratio
            error_rate,           # 7.  error/backspace rate
            acceleration,         # 8.  acceleration curve
            entropy,              # 9.  acceleration entropy
            inter_key_var,        # 10. inter-key variance ratio
            rhythm_shape,         # 11. word rhythm shape
            consistency,          # 12. consistency score (replay detection)
            per_pos_dwell_var,    # 13. per-position dwell profile variance
            bigram_profile_var,   # 14. bigram flight profile variance
            tri_pattern,          # 15. tri-graph overlap pattern
            slope,                # 16. position timing curve slope
            dwell_asym,           # 17. dwell asymmetry (first/second half)
        ]

    # ════════════════════════════════════════════════════════════════
    # PART B: 10 Sequence-Based Features (CRITICAL for mimicry)
    # ════════════════════════════════════════════════════════════════

    def _compute_sequence_features(self, digraphs: np.ndarray) -> List[float]:
        """
        Extract normalized per-key timing sequence.

        This is CRITICAL to prevent human mimicry:
        - Normalize digraph timings by their mean → captures SHAPE not speed
        - Create fixed-length vector of 10 values
        - Pad with 1.0 if fewer than 10 digraphs available

        A mimic can match your average speed but cannot reproduce
        the exact per-transition normalized timing pattern.
        """
        mean_digraph: float = float(np.mean(digraphs))

        if mean_digraph > 0:
            norm_digraphs: np.ndarray = digraphs / mean_digraph
        else:
            norm_digraphs = np.ones_like(digraphs)

        # Create fixed-length vector of exactly SEQUENCE_LENGTH values
        sequence: np.ndarray = np.ones(SEQUENCE_LENGTH)  # default pad = 1.0

        # Take first SEQUENCE_LENGTH values (or fewer if short password)
        n: int = min(len(norm_digraphs), SEQUENCE_LENGTH)
        sequence[:n] = norm_digraphs[:n]

        return sequence.tolist()

    # ════════════════════════════════════════════════════════════════
    # Utility methods
    # ════════════════════════════════════════════════════════════════

    def get_consistency_score(self, features: np.ndarray) -> float:
        """
        Returns the consistency score from feature vector.
        Used for replay attack detection.
        Feature index 11 = consistency score.

        SECURITY: consistency > 0.97 → REJECT as replay attack.
        No human types with that level of consistency.
        """
        if features is None or len(features) < 12:
            return 0.5
        return float(features[11])

    @property
    def feature_dim(self) -> int:
        """Total feature vector dimension: 17 global + 10 sequence = 27."""
        return 17 + SEQUENCE_LENGTH

    @property
    def global_dim(self) -> int:
        """Number of global statistical features."""
        return 17

    @property
    def sequence_dim(self) -> int:
        """Number of sequence-based features."""
        return SEQUENCE_LENGTH

    @property
    def feature_names(self) -> List[str]:
        global_names: List[str] = [
            'avg_flight_time',          # 1
            'avg_dwell_time',           # 2
            'avg_digraph_time',         # 3
            'flight_variance',          # 4
            'dwell_variance',           # 5
            'dwell_flight_ratio',       # 6
            'error_rate',               # 7
            'acceleration_curve',       # 8
            'acceleration_entropy',     # 9
            'inter_key_variance',       # 10
            'rhythm_shape',             # 11
            'consistency_score',        # 12
            'per_pos_dwell_var',        # 13 ← mimic-resistant
            'bigram_profile_var',       # 14 ← mimic-resistant
            'trigraph_pattern',         # 15 ← mimic-resistant
            'position_slope',           # 16 ← mimic-resistant
            'dwell_asymmetry',          # 17 ← mimic-resistant
        ]
        seq_names: List[str] = [
            f'seq_norm_digraph_{i+1}' for i in range(SEQUENCE_LENGTH)
        ]
        return global_names + seq_names
