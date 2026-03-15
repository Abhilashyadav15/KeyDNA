"""
KeyDNA — Main Authentication Pipeline

FINAL ARCHITECTURE:
─────────────────────────────────────────────────────────────────
KeyDNA operates ONLY at application level.
It does NOT handle OS-level phone lock screens.

  Phone lock screen → OS handles it (Face ID / PIN / Pattern)
                      KeyDNA never touches this layer

  App login (phone already open) → KeyDNA handles it
  PC login                       → KeyDNA handles it
  Web login                      → KeyDNA handles it

─────────────────────────────────────────────────────────────────
PRIMARY AUTH:
  Password + Typing Rhythm (KeyDNA)

FALLBACK (triggered after 3 failed rhythm attempts):
  Step 1: Backup PIN
  Step 2: Choose Q1 from 5 questions → answer
  Step 3: Choose Q2 from remaining 4 → answer
  Step 4: Choose Q3 from remaining 3 → answer
  Must be chosen in the SAME ORDER as enrollment.
  Wrong order = rejected.

─────────────────────────────────────────────────────────────────
OTP: completely removed from all contexts.
     OTP fails when attacker has the device.
     PIN + ordered questions are device-independent knowledge.
─────────────────────────────────────────────────────────────────
"""

import time
from typing import Dict, Optional
from core.features import FeatureExtractor
from models.mood_classifier import MoodClassifier
from models.knn_auth import KNNAuthenticator
from authentication.fallback import FallbackEnrollment, FallbackSession


# ── Contexts ──
CONTEXT_APP_MOBILE  = 'APP_MOBILE'   # phone already unlocked, app inside
CONTEXT_APP_DESKTOP = 'APP_DESKTOP'  # PC / laptop app or login screen

# ── Thresholds ──
BACKOFF_SECONDS          = [0, 30, 120, 600, 3600]
MAX_ATTEMPTS_BEFORE_FALL = 3


class AuthenticationResult:
    """Result from one rhythm authentication attempt."""

    def __init__(self, decision, confidence, mood, mood_confidence,
                 reason, replay_detected=False,
                 fallback_required=False,
                 failed_attempts=0, context=CONTEXT_APP_DESKTOP):
        self.decision          = decision
        self.confidence        = confidence
        self.mood              = mood
        self.mood_confidence   = mood_confidence
        self.reason            = reason
        self.replay_detected   = replay_detected
        self.fallback_required = fallback_required   # PIN + questions
        self.failed_attempts   = failed_attempts
        self.context           = context
        self.timestamp         = time.time()

    @property
    def accepted(self):
        return self.decision == 'ACCEPT'

    @property
    def needs_fallback(self):
        return self.decision == 'FALLBACK_REQUIRED'

    def to_dict(self):
        return {
            'decision':          self.decision,
            'confidence':        round(self.confidence, 4),
            'mood':              self.mood,
            'mood_confidence':   round(self.mood_confidence, 4),
            'reason':            self.reason,
            'replay_detected':   self.replay_detected,
            'fallback_required': self.fallback_required,
            'failed_attempts':   self.failed_attempts,
            'context':           self.context,
            'accepted':          self.accepted,
            'needs_fallback':    self.needs_fallback,
        }


class Authenticator:
    """
    Main KeyDNA authentication pipeline.

    Scope:
      ✓ App-level authentication (phone open or PC)
      ✗ NOT phone OS lock screen (OS owns that layer)

    Fallback when rhythm fails 3×:
      PIN → 3 ordered security questions
      Same fallback for both mobile and desktop contexts.
      No OTP anywhere — OTP fails when attacker has the device.
    """

    def __init__(self, mood_classifier: MoodClassifier,
                 knn_auth: KNNAuthenticator,
                 fallback_enrollment: Optional[FallbackEnrollment] = None,
                 context: str = CONTEXT_APP_DESKTOP,
                 device_type: str = 'physical'):

        self.mood_classifier      = mood_classifier
        self.knn_auth             = knn_auth
        self.fallback_enrollment  = fallback_enrollment
        self.extractor            = FeatureExtractor()
        self.context              = context
        self.device_type          = device_type

        self.failed_attempts      = 0
        self.last_attempt_time    = 0.0
        self.attempt_history      = []
        self.active_fallback_session: Optional[FallbackSession] = None

    # ══════════════════════════════════════
    # Primary: Rhythm Authentication
    # ══════════════════════════════════════
    def authenticate(self, events: list) -> AuthenticationResult:
        """
        Primary authentication.
        Step 1: backoff check
        Step 2: feature extraction
        Step 3: Layer 1 mood classification
        Step 4: replay detection
        Step 5: Layer 2 KNN authentication
        Step 6: fallback if 3× failed
        """

        # Step 1: Backoff
        backoff = self._check_backoff()
        if backoff:
            return backoff

        # Step 2: Features
        features      = self.extractor.extract(events)
        mood_features = self.extractor.extract_mood_features(events)

        if features is None or mood_features is None:
            return self._make_result('REJECT', 0.0, 'UNKNOWN', 0.0,
                'Insufficient keystroke data. Type your full password.')

        # Step 3: Mood classification (Layer 1)
        mood, mood_confidence = self.mood_classifier.predict(mood_features)

        # Step 4: Replay detection
        consistency = self.extractor.get_consistency_score(features)
        if consistency >= 0.97:
            self._record_failed_attempt()
            return self._make_result(
                'REJECT', consistency, mood, mood_confidence,
                f'Replay attack detected. Consistency {consistency:.3f} '
                f'exceeds human maximum (0.97).',
                replay_detected=True)

        # Step 5: KNN authentication (Layer 2)
        auth_result = self.knn_auth.authenticate(features, mood, mood_confidence)
        decision    = auth_result['decision']
        confidence  = auth_result['confidence']
        reason      = auth_result['reason']

        # Step 6: Fallback logic
        if decision in ('REJECT', 'OTP_REQUIRED'):
            self._record_failed_attempt()
            if self.failed_attempts >= MAX_ATTEMPTS_BEFORE_FALL:
                return self._trigger_fallback(confidence, mood, mood_confidence)

        elif decision == 'ACCEPT':
            self._record_success()

        result = self._make_result(
            decision, confidence, mood, mood_confidence, reason,
            replay_detected=auth_result.get('replay_detected', False))
        result.failed_attempts = self.failed_attempts

        self.attempt_history.append({
            'timestamp':  result.timestamp,
            'decision':   decision,
            'confidence': confidence,
            'mood':       mood,
        })
        return result

    # ══════════════════════════════════════
    # Fallback Trigger
    # ══════════════════════════════════════
    def _trigger_fallback(self, confidence, mood, mood_confidence) -> AuthenticationResult:
        """
        Rhythm failed 3 times.
        Start PIN + security question fallback session.
        Same for both APP_MOBILE and APP_DESKTOP.
        """
        if self.fallback_enrollment and self.fallback_enrollment.is_complete():
            self.active_fallback_session = FallbackSession(self.fallback_enrollment)
            reason = (
                f'Typing rhythm not recognized after {self.failed_attempts} attempts. '
                f'Fallback: enter your backup PIN, then answer 3 security '
                f'questions in the order you set during enrollment.'
            )
        else:
            reason = (
                f'Typing rhythm not recognized after {self.failed_attempts} attempts. '
                f'No fallback enrolled. Please contact support or re-enroll.'
            )

        result = self._make_result(
            'FALLBACK_REQUIRED', confidence, mood, mood_confidence,
            reason, fallback_required=True)
        result.failed_attempts = self.failed_attempts
        return result

    # ══════════════════════════════════════
    # Fallback Step: PIN
    # ══════════════════════════════════════
    def verify_fallback_pin(self, pin: str) -> Dict:
        """Step 1 of fallback: verify backup PIN."""
        if not self.active_fallback_session:
            return {'success': False,
                    'message': 'No active fallback session.'}
        return self.active_fallback_session.verify_pin(pin)

    # ══════════════════════════════════════
    # Fallback Step: Security Questions
    # ══════════════════════════════════════
    def verify_fallback_question(self, question_index: int,
                                  answer: str) -> Dict:
        """Steps 2–4 of fallback: verify one security question."""
        if not self.active_fallback_session:
            return {'success': False,
                    'message': 'No active fallback session.'}

        result = self.active_fallback_session.verify_question(
            question_index, answer)

        if result.get('authenticated'):
            self._record_success()
            self.active_fallback_session = None

        return result

    # ══════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════
    def _check_backoff(self) -> Optional[AuthenticationResult]:
        if self.failed_attempts == 0:
            return None
        idx      = min(self.failed_attempts, len(BACKOFF_SECONDS) - 1)
        required = BACKOFF_SECONDS[idx]
        elapsed  = time.time() - self.last_attempt_time
        if elapsed < required:
            remaining = int(required - elapsed)
            return self._make_result(
                'REJECT', 0.0, 'UNKNOWN', 0.0,
                f'Too many failed attempts. Wait {remaining}s. '
                f'({self.failed_attempts} of {MAX_ATTEMPTS_BEFORE_FALL} '
                f'before fallback is offered.)')
        return None

    def _record_failed_attempt(self):
        self.failed_attempts   += 1
        self.last_attempt_time  = time.time()

    def _record_success(self):
        self.failed_attempts   = 0
        self.last_attempt_time = 0.0

    def _make_result(self, decision, confidence, mood, mood_confidence,
                     reason, replay_detected=False,
                     fallback_required=False) -> AuthenticationResult:
        return AuthenticationResult(
            decision=decision, confidence=confidence,
            mood=mood, mood_confidence=mood_confidence,
            reason=reason, replay_detected=replay_detected,
            fallback_required=fallback_required,
            failed_attempts=self.failed_attempts,
            context=self.context)

    def reset_attempts(self):
        """Call after successful fallback authentication."""
        self.failed_attempts          = 0
        self.last_attempt_time        = 0.0
        self.active_fallback_session  = None

    def get_summary(self) -> Dict:
        return {
            'context':         self.context,
            'device_type':     self.device_type,
            'failed_attempts': self.failed_attempts,
            'total_attempts':  len(self.attempt_history),
            'fallback_enrolled': (self.fallback_enrollment is not None and
                                  self.fallback_enrollment.is_complete()),
        }
