"""
KeyDNA — Fallback Authentication System

Triggered when typing rhythm fails 3 times.
Security questions fallback:
  Step 1: Choose 3 security questions IN ORDER from 5, answer each

Works ONLY when the app/screen is already accessible.
KeyDNA does NOT handle OS-level phone lock screens.
That is the OS's job (Apple/Google).

Architecture:
  Phone lock screen  → OS handles it (Face ID / PIN / Pattern)
  App login          → KeyDNA handles it (rhythm + this fallback)
  PC login           → KeyDNA handles it (rhythm + this fallback)
"""

import time
from typing import Dict, List, Optional, Tuple

from authentication.recovery import RecoveryEnrollment, RecoverySession
from core.security import hash_answer, verify_answer
from config import (
    SECURITY_QUESTIONS,
    TOTAL_QUESTIONS,
    QUESTIONS_TO_CHOOSE,
    MAX_QUESTION_ATTEMPTS,
)


# ══════════════════════════════════════════════════════
# Enrollment
# ══════════════════════════════════════════════════════

class FallbackEnrollment:
    """
    Collects ordered security question answers
    and trusted contact email during initial enrollment.
    """

    def __init__(self) -> None:
        self.question_order: Optional[List[int]] = None    # e.g. [2, 0, 4]
        self.answer_hashes: Optional[List[str]] = None     # salted PBKDF2 hashes
        self.enrolled_at: Optional[float] = None
        self.recovery: RecoveryEnrollment = RecoveryEnrollment()

    # ── Step 1: Questions ──
    def set_questions(
        self, order: List[int], answers: List[str]
    ) -> Tuple[bool, str]:
        """
        Store the chosen question order and answers.
        order   = list of 3 question indices, e.g. [2, 0, 4]
        answers = list of 3 answers in same order
        """
        if len(order) != QUESTIONS_TO_CHOOSE:
            return False, f"Must choose exactly {QUESTIONS_TO_CHOOSE} questions."
        if len(set(order)) != QUESTIONS_TO_CHOOSE:
            return False, "Cannot choose the same question twice."
        if any(i < 0 or i >= TOTAL_QUESTIONS for i in order):
            return False, f"Question index must be 0–{TOTAL_QUESTIONS - 1}."
        if len(answers) != QUESTIONS_TO_CHOOSE:
            return False, "Must provide one answer per question."
        for i, ans in enumerate(answers):
            if len(ans.strip()) < 2:
                return False, f"Answer {i+1} is too short."

        self.question_order = order
        self.answer_hashes = [hash_answer(a) for a in answers]
        self.enrolled_at = time.time()
        return True, "Security questions enrolled."

    def is_complete(self) -> bool:
        """Questions are required. Recovery contact is optional."""
        return (
            self.question_order is not None and self.answer_hashes is not None
        )

    def has_recovery(self) -> bool:
        """True only if a trusted contact was enrolled for OTP recovery."""
        return self.recovery.is_complete()

    def to_dict(self) -> Dict:
        """Serialize to dict for storage."""
        return {
            'question_order': self.question_order,
            'answer_hashes': self.answer_hashes,
            'enrolled_at': self.enrolled_at,
            'recovery': self.recovery.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FallbackEnrollment':
        """Reconstruct from stored dict."""
        obj: FallbackEnrollment = cls()
        obj.question_order = data.get('question_order')
        obj.answer_hashes = data.get('answer_hashes')
        obj.enrolled_at = data.get('enrolled_at')
        if 'recovery' in data and data['recovery']:
            obj.recovery = RecoveryEnrollment.from_dict(data['recovery'])
        return obj


# ══════════════════════════════════════════════════════
# Runtime Fallback Verification
# ══════════════════════════════════════════════════════

class FallbackSession:
    """
    Manages one live fallback session.
    Called when rhythm fails 3 times.

    Flow:
      Stage 1: verify Q1 (user picks from all 5)
      Stage 2: verify Q2 (user picks from remaining 4)
      Stage 3: verify Q3 (user picks from remaining 3)
      → AUTHENTICATED or REJECTED
    """

    STAGE_Q1: str = 'Q1'
    STAGE_Q2: str = 'Q2'
    STAGE_Q3: str = 'Q3'
    STAGE_RECOVERY: str = 'RECOVERY'
    STAGE_DONE: str = 'DONE'
    STAGE_FAIL: str = 'FAIL'

    def __init__(
        self,
        enrollment: FallbackEnrollment,
        smtp_config: Optional[Dict] = None,
    ) -> None:
        self.enrollment: FallbackEnrollment = enrollment
        self.stage: str = self.STAGE_Q1
        self.answered_indices: List[int] = []
        self.wrong_order_attempts: int = 0
        self.wrong_answer_attempts: int = 0
        self.started_at: float = time.time()
        self.locked_until: float = 0.0
        self._recovery_session: Optional[RecoverySession] = None
        self._smtp_config: Optional[Dict] = smtp_config

    # ── Question Verification ──
    def verify_question(self, question_index: int, answer: str) -> Dict:
        """
        Verify one security question answer.
        question_index = which question the user picked (0–4)
        answer         = what they typed
        """
        stage_to_q_pos: Dict[str, int] = {
            self.STAGE_Q1: 0,
            self.STAGE_Q2: 1,
            self.STAGE_Q3: 2,
        }

        if self.stage not in stage_to_q_pos:
            return self._status(False, "Not in a question stage.")

        q_pos: int = stage_to_q_pos[self.stage]

        # Must be an available question (not already answered)
        available: List[int] = self._available_indices()
        if question_index not in available:
            return self._status(False, "Invalid question selection.")

        # Must match the enrolled order
        expected_index: int = self.enrollment.question_order[q_pos]
        if question_index != expected_index:
            self.wrong_order_attempts += 1
            if self.wrong_order_attempts >= MAX_QUESTION_ATTEMPTS:
                self.stage = self.STAGE_RECOVERY
                return self._status(
                    False,
                    "Wrong question order too many times. "
                    "A recovery OTP will be sent to your trusted contact.",
                )
            return self._status(
                False,
                f"Wrong question chosen. Pick in the same order as enrollment. "
                f"{MAX_QUESTION_ATTEMPTS - self.wrong_order_attempts} attempt(s) left.",
            )

        # Check answer — uses PBKDF2 salted hash verification
        if not verify_answer(answer, self.enrollment.answer_hashes[q_pos]):
            self.wrong_answer_attempts += 1
            if self.wrong_answer_attempts >= MAX_QUESTION_ATTEMPTS:
                self.stage = self.STAGE_RECOVERY
                return self._status(
                    False,
                    "Too many wrong answers. "
                    "A recovery OTP will be sent to your trusted contact.",
                )
            remaining: int = MAX_QUESTION_ATTEMPTS - self.wrong_answer_attempts
            return self._status(
                False, f"Wrong answer. {remaining} attempt(s) remaining."
            )

        # Correct — reset per-question counters before advancing
        self.wrong_order_attempts = 0
        self.wrong_answer_attempts = 0
        self.answered_indices.append(question_index)

        # Advance stage
        next_stages: Dict[str, str] = {
            self.STAGE_Q1: self.STAGE_Q2,
            self.STAGE_Q2: self.STAGE_Q3,
            self.STAGE_Q3: self.STAGE_DONE,
        }
        self.stage = next_stages[self.stage]

        if self.stage == self.STAGE_DONE:
            return self._status(
                True,
                "All questions verified. Authentication successful.",
                next_stage=self.STAGE_DONE,
                authenticated=True,
            )

        return self._status(
            True,
            "Correct! Choose your next question.",
            next_stage=self.stage,
            available_questions=self._available_questions(),
        )

    # ── Recovery: Parental OTP ──

    def request_recovery_otp(self, username: str = "the user") -> Dict:
        """
        Send recovery OTP to trusted contact's email.
        Can only be called when stage == STAGE_RECOVERY.
        """
        if self.stage != self.STAGE_RECOVERY:
            return self._status(
                False,
                "Recovery OTP is only available after all other options are exhausted.",
            )

        if not self.enrollment.has_recovery():
            self.stage = self.STAGE_FAIL
            return self._status(
                False,
                "No trusted contact enrolled. Account is fully locked. "
                "Contact support for manual reset.",
            )

        if self._recovery_session is None:
            self._recovery_session = RecoverySession(
                self.enrollment.recovery, smtp_config=self._smtp_config
            )

        result: Dict = self._recovery_session.send_otp(username)

        status: Dict = self._status(result['success'], result['message'])
        status['time_remaining'] = self._recovery_session.time_remaining()
        return status

    def verify_recovery_otp(self, otp: str) -> Dict:
        """
        Verify OTP entered by user (shared verbally by trusted contact).
        On success, stage moves to DONE and re-enrollment is triggered.
        """
        if self.stage != self.STAGE_RECOVERY:
            return self._status(False, "Not in recovery stage.")

        if self._recovery_session is None:
            return self._status(
                False, "No OTP has been sent yet. Request one first."
            )

        result: Dict = self._recovery_session.verify_otp(otp)

        if result['authenticated']:
            self.stage = self.STAGE_DONE
            return self._status(
                True,
                "Recovery successful. Please re-enroll your rhythm and credentials.",
                next_stage=self.STAGE_DONE,
                authenticated=True,
            )

        if result['failed']:
            self.stage = self.STAGE_FAIL
            return self._status(
                False,
                "Recovery failed. Please contact support for a manual reset.",
            )

        status = self._status(False, result['message'])
        status['attempts_left'] = result['attempts_left']
        if self._recovery_session:
            status['time_remaining'] = self._recovery_session.time_remaining()
        return status

    # ── Helpers ──
    def _available_indices(self) -> List[int]:
        """Question indices not yet answered."""
        return [i for i in range(TOTAL_QUESTIONS) if i not in self.answered_indices]

    def _available_questions(self) -> List[Dict]:
        """Full question objects for available indices."""
        return [
            {'index': i, 'question': SECURITY_QUESTIONS[i]}
            for i in self._available_indices()
        ]

    def _status(
        self,
        success: bool,
        message: str,
        next_stage: Optional[str] = None,
        available_questions: Optional[List] = None,
        authenticated: bool = False,
    ) -> Dict:
        return {
            'success': success,
            'message': message,
            'stage': self.stage,
            'next_stage': next_stage,
            'available_questions': available_questions or [],
            'authenticated': authenticated,
            'failed': self.stage == self.STAGE_FAIL,
            'recovery_available': self.stage == self.STAGE_RECOVERY,
            'questions_answered': len(self.answered_indices),
            'questions_remaining': QUESTIONS_TO_CHOOSE - len(self.answered_indices),
        }

    @property
    def is_done(self) -> bool:
        """True if all questions answered successfully."""
        return self.stage == self.STAGE_DONE

    @property
    def is_failed(self) -> bool:
        """True if fallback has fully failed."""
        return self.stage == self.STAGE_FAIL

    @property
    def needs_recovery(self) -> bool:
        """True if OTP recovery is needed."""
        return self.stage == self.STAGE_RECOVERY

    def get_progress(self) -> Dict:
        """Return progress information."""
        return {
            'stage': self.stage,
            'questions_answered': len(self.answered_indices),
            'total_questions': QUESTIONS_TO_CHOOSE,
            'progress_pct': len(self.answered_indices) / QUESTIONS_TO_CHOOSE * 100,
        }
