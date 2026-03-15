"""
KeyDNA — Enrollment System

Enrollment = pure data collection.
No classification. No validation. No mood checks.
Just: capture keystrokes → extract 12 timing features → store.

Mood classifier runs ONLY during login, never during enrollment.
The only things rejected here:
  - Autofill / paste (machine inserted, not real typing)
  - Zero events (nothing was typed)
"""

import time
import numpy as np
from typing import Optional, Dict, List
from core.capture import SimulatedCapture
from core.features import FeatureExtractor
from models.mood_classifier import MOODS


AUTOFILL_THRESHOLD_MS = 50   # total password < 50ms = autofill
MIN_EVENTS            = 3    # minimum keystrokes needed to extract features


class EnrollmentSession:
    """
    Collects real keystroke samples for one mood during enrollment.

    Does exactly one thing: take events → extract 12 features → store.
    No mood classification. No speed checks. No quality judgment.
    Whatever the user types IS their pattern for that mood.
    """

    def __init__(self, mood: str, target_samples: int = 10):
        self.mood = mood
        self.target_samples = target_samples
        self.collected_samples: List[np.ndarray] = []
        self.extractor = FeatureExtractor()
        self.start_time = time.time()

    def process_attempt(self, events: list) -> Dict:
        """
        Accept one typing attempt.

        Only two rejection reasons:
          1. No events at all
          2. Autofill / paste detected (not real typing)

        Everything else is accepted and stored.
        """
        result = {
            'accepted':  False,
            'features':  None,
            'feedback':  '',
            'progress':  self.progress,
        }

        # Nothing typed
        if not events or len(events) < MIN_EVENTS:
            result['feedback'] = 'No keystrokes detected. Please type your password.'
            return result

        # Autofill / paste — not real typing
        if self._is_autofill(events):
            result['feedback'] = 'Paste or autofill detected. Please type manually.'
            return result

        # Extract 12 timing features — no mood features needed here
        features = self.extractor.extract(events)

        if features is None:
            result['feedback'] = 'Too few keystrokes captured. Try again.'
            return result

        # Store — no further checks
        self.collected_samples.append(features)
        result['accepted'] = True
        result['features'] = features
        result['progress'] = self.progress
        result['feedback'] = f'Sample {len(self.collected_samples)}/{self.target_samples} saved.'

        return result

    def _is_autofill(self, events: list) -> bool:
        """Detect paste or password manager autofill."""
        if len(events) < 2:
            return False
        # Entire password completed in under 50ms = machine
        total_ms = (events[-1]['release'] - events[0]['press']) * 1000
        if total_ms < AUTOFILL_THRESHOLD_MS:
            return True
        # All gaps suspiciously uniform = machine
        gaps = [(events[i+1]['press'] - events[i]['release']) * 1000
                for i in range(len(events) - 1)]
        if len(gaps) > 2 and np.std(gaps) < 2.0:
            return True
        return False

    def _get_progress_feedback(self) -> str:
        n = len(self.collected_samples)
        target = self.target_samples
        if n < 5:
            return f'Sample {n}/{target} collected. Keep going!'
        elif n < 10:
            return f'Sample {n}/{target} — System learning your rhythm.'
        elif n < target:
            return f'Sample {n}/{target} — Almost there!'
        else:
            return f'Enrollment complete for {self.mood} mood! ✓'

    @property
    def progress(self) -> float:
        return min(1.0, len(self.collected_samples) / self.target_samples)

    @property
    def is_complete(self) -> bool:
        return len(self.collected_samples) >= self.target_samples

    @property
    def sample_count(self) -> int:
        return len(self.collected_samples)


class Enroller:
    """
    Manages complete multi-mood enrollment process.
    Guides user through enrolling in all 4 mood states.
    """

    MOOD_INSTRUCTIONS = {
        'RELAXED': (
            'You are in RELAXED mode.\n'
            'Type your password as you would on a quiet weekend morning.\n'
            'No pressure. Take your time. Type naturally and comfortably.'
        ),
        'FOCUSED': (
            'You are in FOCUSED mode.\n'
            'Type your password as you would during concentrated work.\n'
            'Efficient and deliberate. Like during a productive work session.'
        ),
        'STRESSED': (
            'You are in STRESSED mode.\n'
            'Type your password as you would before an important deadline.\n'
            'Slightly rushed. A bit of urgency. Type faster than normal.'
        ),
        'TIRED': (
            'You are in TIRED mode.\n'
            'Type your password as you would late at night when tired.\n'
            'Slower and more deliberate. Take your time between keys.'
        ),
    }

    def __init__(self, samples_per_mood: int = 10):
        self.samples_per_mood = samples_per_mood
        self.sessions: Dict[str, EnrollmentSession] = {}
        self.extractor = FeatureExtractor()

    def start_mood_session(self, mood: str) -> EnrollmentSession:
        """Start enrollment session for a specific mood."""
        session = EnrollmentSession(mood, self.samples_per_mood)
        self.sessions[mood] = session
        return session

    def get_all_samples(self) -> Dict[str, List[np.ndarray]]:
        """Return all collected samples organized by mood."""
        return {
            mood: session.collected_samples
            for mood, session in self.sessions.items()
        }

    def get_enrollment_status(self) -> Dict:
        """Overall enrollment status across all moods."""
        status = {}
        for mood in MOODS:
            if mood in self.sessions:
                session = self.sessions[mood]
                status[mood] = {
                    'samples':   session.sample_count,
                    'complete':  session.is_complete,
                    'progress':  session.progress,
                }
            else:
                status[mood] = {
                    'samples':  0,
                    'complete': False,
                    'progress': 0.0,
                }
        return status

    def is_fully_enrolled(self) -> bool:
        """True if at least 2 moods are fully enrolled."""
        complete_count = sum(
            1 for mood in MOODS
            if mood in self.sessions and self.sessions[mood].is_complete
        )
        return complete_count >= 2

    def can_authenticate(self) -> bool:
        """True if at least 1 mood has minimum samples."""
        for mood in MOODS:
            if mood in self.sessions:
                if self.sessions[mood].sample_count >= 5:
                    return True
        return False

    def simulate_enrollment(self) -> Dict[str, List[np.ndarray]]:
        """
        Simulate enrollment for demo/testing.
        Generates realistic samples for each mood.
        Fix #4: Progressive enrollment demonstrated.
        """
        capture = SimulatedCapture()
        all_samples = {}

        mood_params = {
            'RELAXED':  {'avg': 115, 'var': 13, 'dwell': 87,  'error': 0.01},
            'FOCUSED':  {'avg': 87,  'var': 10, 'dwell': 70,  'error': 0.008},
            'STRESSED': {'avg': 67,  'var': 30, 'dwell': 52,  'error': 0.08},
            'TIRED':    {'avg': 155, 'var': 18, 'dwell': 110, 'error': 0.05},
        }

        for mood, params in mood_params.items():
            samples = []
            session = self.start_mood_session(mood)

            for i in range(self.samples_per_mood):
                events = capture.simulate_typing(
                    avg_speed_ms=params['avg'],
                    variance_ms=params['var'],
                    dwell_ms=params['dwell'],
                    n_keys=10,
                    error_rate=params['error'],
                    seed=hash(mood + str(i)) % 10000
                )
                features = self.extractor.extract(events)
                if features is not None:
                    session.collected_samples.append(features)
                    samples.append(features)

            all_samples[mood] = samples

        return all_samples


# ══════════════════════════════════════════════════════
# Fallback Enrollment Helper (added to existing Enroller)
# ══════════════════════════════════════════════════════

def simulate_fallback_enrollment():
    """
    Simulate a complete fallback enrollment for demo/testing.
    Returns a FallbackEnrollment object ready to use.
    """
    from authentication.fallback import FallbackEnrollment

    fe = FallbackEnrollment()

    # Demo PIN
    fe.set_pin("847291")

    # Demo: choose questions in order [2, 0, 4]
    # Q2 = favourite food, Q0 = favourite teacher, Q4 = mother's maiden name
    fe.set_questions(
        order=[2, 0, 4],
        answers=["pizza", "mr sharma", "patel"]
    )

    return fe
