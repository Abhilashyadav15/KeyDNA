"""
KeyDNA — Fallback Authentication System

Triggered when typing rhythm fails 3 times.
Two-step fallback:
  Step 1: Backup PIN
  Step 2: Choose 3 security questions IN ORDER from 5, answer each

Works ONLY when the app/screen is already accessible.
KeyDNA does NOT handle OS-level phone lock screens.
That is the OS's job (Apple/Google).

Architecture:
  Phone lock screen  → OS handles it (Face ID / PIN / Pattern)
  App login          → KeyDNA handles it (rhythm + this fallback)
  PC login           → KeyDNA handles it (rhythm + this fallback)
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple


# ── 5 Security Questions (shown all at once, user picks 3 in order) ──
SECURITY_QUESTIONS = [
    "What is your favourite teacher's name?",
    "What was your first pet's name?",
    "What is your favourite food?",
    "What is your childhood nickname?",
    "What is your mother's maiden name?",
]

TOTAL_QUESTIONS      = len(SECURITY_QUESTIONS)   # 5
QUESTIONS_TO_CHOOSE  = 3                          # user picks 3
MAX_PIN_ATTEMPTS     = 3
MAX_QUESTION_ATTEMPTS= 3


def _hash(value: str) -> str:
    """Normalize and hash a string. Case-insensitive, strip whitespace."""
    normalized = value.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


# ══════════════════════════════════════════════════════
# Enrollment
# ══════════════════════════════════════════════════════

class FallbackEnrollment:
    """
    Collects backup PIN and ordered security question answers
    during initial enrollment.
    """

    def __init__(self):
        self.pin_hash:        Optional[str]       = None
        self.question_order:  Optional[List[int]] = None   # e.g. [2, 0, 4]
        self.answer_hashes:   Optional[List[str]] = None   # hashed answers in order
        self.enrolled_at:     Optional[float]     = None

    # ── Step 1: PIN ──
    def set_pin(self, pin: str) -> Tuple[bool, str]:
        """
        Validate and store backup PIN.
        Rules: 6–12 digits, not all same digit.
        """
        pin = pin.strip()
        if not pin.isdigit():
            return False, "PIN must contain digits only."
        if not (6 <= len(pin) <= 12):
            return False, "PIN must be 6 to 12 digits."
        if len(set(pin)) == 1:
            return False, "PIN cannot be all the same digit (e.g. 111111)."
        self.pin_hash = _hash(pin)
        return True, "PIN set successfully."

    # ── Step 2: Questions ──
    def set_questions(self, order: List[int],
                      answers: List[str]) -> Tuple[bool, str]:
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
        self.answer_hashes  = [_hash(a) for a in answers]
        self.enrolled_at    = time.time()
        return True, "Security questions enrolled."

    def is_complete(self) -> bool:
        return (self.pin_hash is not None and
                self.question_order is not None and
                self.answer_hashes is not None)

    def to_dict(self) -> Dict:
        return {
            'pin_hash':       self.pin_hash,
            'question_order': self.question_order,
            'answer_hashes':  self.answer_hashes,
            'enrolled_at':    self.enrolled_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FallbackEnrollment':
        obj = cls()
        obj.pin_hash        = data.get('pin_hash')
        obj.question_order  = data.get('question_order')
        obj.answer_hashes   = data.get('answer_hashes')
        obj.enrolled_at     = data.get('enrolled_at')
        return obj


# ══════════════════════════════════════════════════════
# Runtime Fallback Verification
# ══════════════════════════════════════════════════════

class FallbackSession:
    """
    Manages one live fallback session.
    Called when rhythm fails 3 times.

    Flow:
      Stage 1: verify PIN
      Stage 2: verify Q1 (user picks from all 5)
      Stage 3: verify Q2 (user picks from remaining 4)
      Stage 4: verify Q3 (user picks from remaining 3)
      → AUTHENTICATED or REJECTED
    """

    STAGE_PIN  = 'PIN'
    STAGE_Q1   = 'Q1'
    STAGE_Q2   = 'Q2'
    STAGE_Q3   = 'Q3'
    STAGE_DONE = 'DONE'
    STAGE_FAIL = 'FAIL'

    def __init__(self, enrollment: FallbackEnrollment):
        self.enrollment       = enrollment
        self.stage            = self.STAGE_PIN
        self.answered_indices: List[int] = []   # questions answered so far
        self.pin_attempts     = 0
        self.question_attempts= 0
        self.started_at       = time.time()
        self.locked_until     = 0.0

    # ── PIN Verification ──
    def verify_pin(self, pin: str) -> Dict:
        """Verify backup PIN. Returns status dict."""
        if self.stage != self.STAGE_PIN:
            return self._status(False, "Not in PIN stage.")

        if time.time() < self.locked_until:
            wait = int(self.locked_until - time.time())
            return self._status(False, f"Too many attempts. Wait {wait}s.")

        self.pin_attempts += 1

        if _hash(pin) == self.enrollment.pin_hash:
            self.stage = self.STAGE_Q1
            return self._status(True,
                "PIN correct. Now choose your first security question.",
                next_stage=self.STAGE_Q1,
                available_questions=self._available_questions())
        else:
            if self.pin_attempts >= MAX_PIN_ATTEMPTS:
                self.stage       = self.STAGE_FAIL
                self.locked_until= time.time() + 300   # 5 min lockout
                return self._status(False,
                    "Too many wrong PINs. Locked for 5 minutes.")
            remaining = MAX_PIN_ATTEMPTS - self.pin_attempts
            return self._status(False,
                f"Wrong PIN. {remaining} attempt(s) remaining.")

    # ── Question Verification ──
    def verify_question(self, question_index: int, answer: str) -> Dict:
        """
        Verify one security question answer.
        question_index = which question the user picked (0–4)
        answer         = what they typed
        """
        stage_to_q_pos = {self.STAGE_Q1: 0,
                          self.STAGE_Q2: 1,
                          self.STAGE_Q3: 2}

        if self.stage not in stage_to_q_pos:
            return self._status(False, "Not in a question stage.")

        q_pos = stage_to_q_pos[self.stage]   # 0, 1, or 2

        # Must be an available question (not already answered)
        available = self._available_indices()
        if question_index not in available:
            return self._status(False, "Invalid question selection.")

        # Must match the enrolled order
        expected_index = self.enrollment.question_order[q_pos]
        if question_index != expected_index:
            self.question_attempts += 1
            if self.question_attempts >= MAX_QUESTION_ATTEMPTS:
                self.stage = self.STAGE_FAIL
                return self._status(False,
                    "Wrong question order. Fallback failed. "
                    "You must choose questions in the same order as enrollment.")
            return self._status(False,
                f"Wrong question chosen. Pick in the same order as enrollment. "
                f"{MAX_QUESTION_ATTEMPTS - self.question_attempts} attempt(s) left.")

        # Check answer
        if _hash(answer) != self.enrollment.answer_hashes[q_pos]:
            self.question_attempts += 1
            if self.question_attempts >= MAX_QUESTION_ATTEMPTS:
                self.stage = self.STAGE_FAIL
                return self._status(False,
                    "Too many wrong answers. Fallback failed.")
            remaining = MAX_QUESTION_ATTEMPTS - self.question_attempts
            return self._status(False,
                f"Wrong answer. {remaining} attempt(s) remaining.")

        # Correct
        self.answered_indices.append(question_index)

        # Advance stage
        next_stages = {
            self.STAGE_Q1: self.STAGE_Q2,
            self.STAGE_Q2: self.STAGE_Q3,
            self.STAGE_Q3: self.STAGE_DONE,
        }
        self.stage = next_stages[self.stage]

        if self.stage == self.STAGE_DONE:
            return self._status(True,
                "All questions verified. Authentication successful.",
                next_stage=self.STAGE_DONE,
                authenticated=True)

        return self._status(True,
            f"Correct! Choose your next question.",
            next_stage=self.stage,
            available_questions=self._available_questions())

    # ── Helpers ──
    def _available_indices(self) -> List[int]:
        """Question indices not yet answered."""
        return [i for i in range(TOTAL_QUESTIONS)
                if i not in self.answered_indices]

    def _available_questions(self) -> List[Dict]:
        """Full question objects for available indices."""
        return [{'index': i, 'question': SECURITY_QUESTIONS[i]}
                for i in self._available_indices()]

    def _status(self, success: bool, message: str,
                next_stage: str = None,
                available_questions: List = None,
                authenticated: bool = False) -> Dict:
        return {
            'success':             success,
            'message':             message,
            'stage':               self.stage,
            'next_stage':          next_stage,
            'available_questions': available_questions or [],
            'authenticated':       authenticated,
            'failed':              self.stage == self.STAGE_FAIL,
            'questions_answered':  len(self.answered_indices),
            'questions_remaining': QUESTIONS_TO_CHOOSE - len(self.answered_indices),
        }

    @property
    def is_done(self) -> bool:
        return self.stage == self.STAGE_DONE

    @property
    def is_failed(self) -> bool:
        return self.stage == self.STAGE_FAIL

    def get_progress(self) -> Dict:
        return {
            'stage':              self.stage,
            'questions_answered': len(self.answered_indices),
            'total_questions':    QUESTIONS_TO_CHOOSE,
            'progress_pct':       len(self.answered_indices) / QUESTIONS_TO_CHOOSE * 100,
        }
