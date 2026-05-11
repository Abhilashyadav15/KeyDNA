"""
KeyDNA — Enrollment System (v2 — Unified Model)

ARCHITECTURE:
  NO mood classifier. NO per-mood enrollment.
  Single unified enrollment — all samples go into one pool.

ENROLLMENT:
  Collect 10 typing samples from the user.
  Each sample → extract 27 features → store in unified model.
  No mood classification. No mood prompts.
  Whatever the user types IS their pattern.

REJECTION ONLY FOR:
  - Autofill / paste (machine inserted, not real typing)
  - Zero events (nothing was typed)
  - Fewer than 3 keystrokes
"""

import time
import numpy as np
from typing import Dict, List, Optional

from core.features import FeatureExtractor
from config import AUTOFILL_THRESHOLD_MS, MIN_KEYSTROKE_EVENTS, ENROLLMENT_SAMPLES


class EnrollmentSession:
    """
    Collects real keystroke samples for enrollment.

    Unified model — NO mood classification.
    Does exactly one thing: take events → extract 27 features → store.
    No mood prompts. No speed checks. No quality judgment.
    Whatever the user types IS their pattern.
    """

    def __init__(self, target_samples: int = ENROLLMENT_SAMPLES) -> None:
        self.target_samples: int = target_samples
        self.collected_samples: List[np.ndarray] = []
        self.extractor: FeatureExtractor = FeatureExtractor()
        self.start_time: float = time.time()

    def process_attempt(self, events: List[Dict]) -> Dict:
        """
        Accept one typing attempt.

        Only two rejection reasons:
          1. No events at all / too few keystrokes
          2. Autofill / paste detected (not real typing)

        Everything else is accepted and stored.
        """
        result: Dict = {
            'accepted': False,
            'features': None,
            'feedback': '',
            'progress': self.progress,
        }

        # Nothing typed
        if not events or len(events) < MIN_KEYSTROKE_EVENTS:
            result['feedback'] = 'No keystrokes detected. Please type your password.'
            return result

        # Autofill / paste — not real typing
        if self._is_autofill(events):
            result['feedback'] = 'Paste or autofill detected. Please type manually.'
            return result

        # Extract 27 timing features
        features: Optional[np.ndarray] = self.extractor.extract(events)

        if features is None:
            result['feedback'] = 'Too few keystrokes captured. Try again.'
            return result

        # Store — no further checks
        self.collected_samples.append(features)
        result['accepted'] = True
        result['features'] = features
        result['progress'] = self.progress
        result['feedback'] = (
            f'Sample {len(self.collected_samples)}/{self.target_samples} saved.'
        )

        return result

    def _is_autofill(self, events: List[Dict]) -> bool:
        """Detect paste or password manager autofill."""
        if len(events) < 2:
            return False
        # Entire password completed in under 50ms = machine
        total_ms: float = (events[-1]['release'] - events[0]['press']) * 1000
        if total_ms < AUTOFILL_THRESHOLD_MS:
            return True
        # All gaps suspiciously uniform = machine
        gaps: List[float] = [
            (events[i + 1]['press'] - events[i]['release']) * 1000
            for i in range(len(events) - 1)
        ]
        if len(gaps) > 2 and np.std(gaps) < 2.0:
            return True
        return False

    @property
    def progress(self) -> float:
        """Enrollment progress (0.0–1.0)."""
        return min(1.0, len(self.collected_samples) / self.target_samples)

    @property
    def is_complete(self) -> bool:
        """True if enough samples have been collected."""
        return len(self.collected_samples) >= self.target_samples

    @property
    def sample_count(self) -> int:
        """Number of samples collected so far."""
        return len(self.collected_samples)
