"""
KeyDNA — Centralized Configuration

All project-wide constants and parameters in one place.
Import from here instead of defining inline.
"""

import os

# ═══════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════════════

PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(PROJECT_DIR, "data")
DATA_FILE: str = os.path.join(DATA_DIR, "user_profile.json")  # plain JSON
THEME_FILE: str = os.path.join(DATA_DIR, "theme.txt")

# ═══════════════════════════════════════════════════════════════════════
# ENROLLMENT
# ═══════════════════════════════════════════════════════════════════════

ENROLLMENT_SAMPLES: int = 10       # typing samples required during enrollment
MIN_ENROLLMENT_SAMPLES: int = 5    # minimum before auth is enabled
MAX_ENROLLMENT_SAMPLES: int = 10   # target enrollment samples

# ═══════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════

MAX_ATTEMPTS: int = 3              # failed attempts before fallback
SESSION_TIMEOUT: int = 300         # seconds before auto sign-out

# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

SEQUENCE_LENGTH: int = 10          # fixed-length normalized digraph sequence
GLOBAL_DIM: int = 17               # features [0:17] — global statistical
SEQUENCE_DIM: int = SEQUENCE_LENGTH  # features [17:27] — sequence-based
TOTAL_DIM: int = GLOBAL_DIM + SEQUENCE_DIM  # 27

# ═══════════════════════════════════════════════════════════════════════
# ONE-CLASS SVM PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

SVM_KERNEL: str = "rbf"
SVM_GAMMA: str = "scale"
SVM_NU: float = 0.18              # captures 82% of natural typing boundary
THRESHOLD_ACCEPT: float = -0.05   # score >= this → ACCEPT
THRESHOLD_RETRY: float = -0.15   # score >= this → RETRY, below → REJECT

# ═══════════════════════════════════════════════════════════════════════
# ANTI-ATTACK
# ═══════════════════════════════════════════════════════════════════════

REPLAY_CONSISTENCY_THRESHOLD: float = 0.97
ADAPTIVE_ATTACK_WINDOW: int = 3
AUTOFILL_THRESHOLD_MS: float = 50.0
MIN_KEYSTROKE_EVENTS: int = 3

# ═══════════════════════════════════════════════════════════════════════
# FALLBACK & RECOVERY
# ═══════════════════════════════════════════════════════════════════════

SECURITY_QUESTIONS: list = [
    "What is your favourite teacher's name?",
    "What was your first pet's name?",
    "What is your favourite food?",
    "What is your childhood nickname?",
    "What is your mother's maiden name?",
]

TOTAL_QUESTIONS: int = len(SECURITY_QUESTIONS)
QUESTIONS_TO_CHOOSE: int = 3
MAX_QUESTION_ATTEMPTS: int = 3

# Recovery
OTP_LENGTH: int = 6
OTP_EXPIRY_SECONDS: int = 600
MAX_OTP_ATTEMPTS: int = 3
MAX_REQUESTS_PER_HR: int = 3
LOCKOUT_SECONDS: int = 3600

# ═══════════════════════════════════════════════════════════════════════
# UI THEME COLORS
# ═══════════════════════════════════════════════════════════════════════

C_BG: str = "#080c14"
C_CARD: str = "#0e1420"
C_BORDER: str = "#1a2436"
C_ACCENT: str = "#00d4ff"
C_TEXT: str = "#e8eaf0"
C_SUB: str = "#94a3b8"
C_GREEN: str = "#10b981"
C_RED: str = "#ef4444"
C_YELLOW: str = "#f59e0b"
C_PURPLE: str = "#a855f7"
C_DIM: str = "#2d3748"          # muted section labels / subtitles

# Light mode overrides
C_BG_LIGHT: str = "#f0f2f5"
C_CARD_LIGHT: str = "#ffffff"
C_BORDER_LIGHT: str = "#d1d5db"
C_TEXT_LIGHT: str = "#1f2937"
C_SUB_LIGHT: str = "#6b7280"

# ═══════════════════════════════════════════════════════════════════════
# WINDOW
# ═══════════════════════════════════════════════════════════════════════

WINDOW_MIN_WIDTH: int = 460
WINDOW_MIN_HEIGHT: int = 580
WINDOW_DEFAULT_GEOMETRY: str = "460x580"
