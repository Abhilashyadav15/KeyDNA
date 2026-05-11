"""
KeyDNA — Security Utilities (Simplified)

Password hashing  : SHA-256
Answer hashing    : SHA-256
Profile storage   : Plain JSON (no encryption)

Focus is on the ML biometric model, not enterprise-grade crypto.
"""

import hashlib
import json
import os
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# PASSWORD HASHING (SHA-256)
# ═══════════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    normalized: str = password.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against its SHA-256 hash."""
    if not stored_hash:
        return False
    normalized: str = password.strip()
    return hashlib.sha256(normalized.encode()).hexdigest() == stored_hash


def needs_rehash(stored_hash: str) -> bool:
    """Always False — SHA-256 hashes never need migration."""
    return False


# ═══════════════════════════════════════════════════════════════════════
# SECURITY ANSWER HASHING (SHA-256)
# ═══════════════════════════════════════════════════════════════════════

def hash_answer(answer: str) -> str:
    """Hash a security question answer using SHA-256 (case-insensitive)."""
    normalized: str = answer.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def verify_answer(answer: str, stored: str) -> bool:
    """Verify a security question answer against its SHA-256 hash."""
    if not stored:
        return False
    normalized: str = answer.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest() == stored


# ═══════════════════════════════════════════════════════════════════════
# PROFILE STORAGE (Plain JSON)
# ═══════════════════════════════════════════════════════════════════════

def save_encrypted_profile(data: dict, filepath: str) -> None:
    """Save profile as plain JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_encrypted_profile(filepath: str) -> dict:
    """
    Load profile from JSON file.
    Returns empty dict if file not found or unreadable.
    """
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return {}
