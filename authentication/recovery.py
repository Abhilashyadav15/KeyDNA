"""
KeyDNA — Parental Email Recovery System

Triggered ONLY when the full fallback chain fails:
  Rhythm (3x) → Security Questions (3x) → HERE

Sends a 6-digit OTP to a pre-registered trusted contact's email.
The trusted contact shares it verbally/physically with the user.
The user enters it to unlock and re-enroll.

Why this works (unlike regular OTP):
  Regular OTP: attacker has the phone → reads the SMS themselves.
  Parental OTP: goes to a DIFFERENT person on a DIFFERENT device.
  Attacker cannot intercept it without involving the trusted person.

Security rules:
  - OTP is 6 digits, valid for 10 minutes only
  - Maximum 3 attempts per OTP
  - Maximum 3 recovery requests per hour (rate limit)
  - OTP invalidated immediately after first correct use
  - Email subject never contains the OTP (prevents preview leaks)
"""

import hashlib
import os
import random
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

from config import (
    OTP_LENGTH,
    OTP_EXPIRY_SECONDS,
    MAX_OTP_ATTEMPTS,
    MAX_REQUESTS_PER_HR,
    LOCKOUT_SECONDS,
)


def _hash(value: str) -> str:
    """SHA-256 hash, normalized (lowercase + stripped)."""
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()


def _generate_otp() -> str:
    """Generate a cryptographically random 6-digit OTP."""
    return ''.join(
        [str(random.SystemRandom().randint(0, 9)) for _ in range(OTP_LENGTH)]
    )


# ══════════════════════════════════════════════════════
# Enrollment  (called once during initial setup)
# ══════════════════════════════════════════════════════

class RecoveryEnrollment:
    """
    Stores the trusted contact's name and email during enrollment.
    Email is stored as plaintext for sending — but we never log it.
    Name is stored so the UI can show "OTP sent to Mom" etc.
    """

    def __init__(self) -> None:
        self.trusted_name: Optional[str] = None    # e.g. "Mom"
        self.trusted_email: Optional[str] = None   # e.g. "mom@example.com"
        self.enrolled_at: Optional[float] = None

    def set_trusted_contact(
        self, name: str, email: str
    ) -> Tuple[bool, str]:
        """
        Validate and store trusted contact details.
        Rules:
          - Name: 2–50 characters
          - Email: basic format check (contains @ and .)
        """
        name = name.strip()
        email = email.strip().lower()

        if len(name) < 2 or len(name) > 50:
            return False, "Trusted contact name must be 2–50 characters."

        if '@' not in email or '.' not in email.split('@')[-1]:
            return False, "Invalid email address format."

        if len(email) > 254:  # RFC 5321 max
            return False, "Email address is too long."

        self.trusted_name = name
        self.trusted_email = email
        self.enrolled_at = time.time()
        return True, f"Trusted contact '{name}' enrolled successfully."

    def is_complete(self) -> bool:
        """True if trusted contact details are set."""
        return self.trusted_name is not None and self.trusted_email is not None

    def masked_email(self) -> str:
        """
        Returns partially masked email for UI display.
        e.g. mom@example.com → m**@e******.com
        """
        if not self.trusted_email:
            return "not set"
        local, domain = self.trusted_email.split('@', 1)
        domain_parts: List[str] = domain.rsplit('.', 1)
        masked_local: str = local[0] + '**'
        masked_domain: str = domain_parts[0][0] + (
            '*' * (len(domain_parts[0]) - 1)
        )
        return f"{masked_local}@{masked_domain}.{domain_parts[1]}"

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            'trusted_name': self.trusted_name,
            'trusted_email': self.trusted_email,
            'enrolled_at': self.enrolled_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RecoveryEnrollment':
        """Reconstruct from dict."""
        obj: RecoveryEnrollment = cls()
        obj.trusted_name = data.get('trusted_name')
        obj.trusted_email = data.get('trusted_email')
        obj.enrolled_at = data.get('enrolled_at')
        return obj


# ══════════════════════════════════════════════════════
# Runtime Recovery Session
# ══════════════════════════════════════════════════════

class RecoverySession:
    """
    Manages one live recovery session.

    Flow:
      1. send_otp()      → emails OTP to trusted contact
      2. verify_otp()    → user enters OTP (told by trusted contact)
      3. On success → caller triggers re-enrollment
    """

    STAGE_READY: str = 'READY'
    STAGE_SENT: str = 'SENT'
    STAGE_EXPIRED: str = 'EXPIRED'
    STAGE_SUCCESS: str = 'SUCCESS'
    STAGE_FAIL: str = 'FAIL'

    def __init__(
        self,
        enrollment: RecoveryEnrollment,
        smtp_config: Optional[Dict] = None,
    ) -> None:
        self.enrollment: RecoveryEnrollment = enrollment
        self.smtp_config: Optional[Dict] = smtp_config or self._config_from_env()
        self.stage: str = self.STAGE_READY
        self._otp: Optional[str] = None
        self._otp_hash: Optional[str] = None
        self._otp_expiry: float = 0.0
        self._attempts: int = 0
        self._request_times: List[float] = []
        self.started_at: float = time.time()

    # ── Step 1: Send OTP ──

    def send_otp(self, username: str = "the user") -> Dict:
        """
        Generate OTP and email it to the trusted contact.
        username: shown in the email body so the contact knows
                  whose account needs recovery.
        """
        if self.stage == self.STAGE_SUCCESS:
            return self._status(False, "Recovery already completed.")

        if self.stage == self.STAGE_FAIL:
            return self._status(
                False, "Too many wrong OTP entries. Restart the app."
            )

        # ── Rate limit: max 3 requests per hour ──
        now: float = time.time()
        self._request_times = [
            t for t in self._request_times if now - t < LOCKOUT_SECONDS
        ]
        if len(self._request_times) >= MAX_REQUESTS_PER_HR:
            wait_min: int = (
                int((LOCKOUT_SECONDS - (now - self._request_times[0])) / 60) + 1
            )
            return self._status(
                False,
                f"Too many recovery requests. Please wait {wait_min} minute(s).",
            )

        if not self.enrollment.is_complete():
            return self._status(
                False, "No trusted contact enrolled. Cannot send recovery OTP."
            )

        # Generate OTP
        otp: str = _generate_otp()
        self._otp_hash = _hash(otp)
        self._otp_expiry = now + OTP_EXPIRY_SECONDS
        self._attempts = 0
        self._otp = otp

        # Send email
        sent: bool
        message: str
        sent, message = self._send_email(otp, username)
        self._otp = None  # clear plaintext immediately

        if not sent:
            return self._status(
                False, f"Failed to send recovery email: {message}"
            )

        self._request_times.append(now)
        self.stage = self.STAGE_SENT

        masked: str = self.enrollment.masked_email()
        return self._status(
            True,
            f"Recovery OTP sent to {self.enrollment.trusted_name} "
            f"({masked}). Ask them to share it with you. "
            f"OTP is valid for 10 minutes.",
        )

    # ── Step 2: Verify OTP ──

    def verify_otp(self, entered_otp: str) -> Dict:
        """Verify the OTP entered by the user."""
        if self.stage != self.STAGE_SENT:
            return self._status(False, "No OTP has been sent yet.")

        # Expiry check
        if time.time() > self._otp_expiry:
            self.stage = self.STAGE_EXPIRED
            self._otp_hash = None
            return self._status(
                False,
                "OTP has expired (10 minute limit). "
                "Click 'Send New OTP' to request another.",
            )

        self._attempts += 1

        if _hash(entered_otp.strip()) == self._otp_hash:
            # Correct — invalidate immediately
            self._otp_hash = None
            self._otp_expiry = 0.0
            self.stage = self.STAGE_SUCCESS
            return self._status(
                True,
                "OTP verified. You can now re-enroll your rhythm and credentials.",
                authenticated=True,
            )
        else:
            remaining: int = MAX_OTP_ATTEMPTS - self._attempts
            if remaining <= 0:
                self.stage = self.STAGE_FAIL
                self._otp_hash = None
                return self._status(
                    False,
                    "Too many wrong OTP entries. Recovery failed. "
                    "Please contact support or request a new session.",
                )
            return self._status(
                False, f"Wrong OTP. {remaining} attempt(s) remaining."
            )

    # ── Email Sending ──

    def _send_email(self, otp: str, username: str) -> Tuple[bool, str]:
        """Send OTP email to the trusted contact."""
        cfg: Optional[Dict] = self.smtp_config
        if not cfg:
            return False, "SMTP not configured."

        contact_name: str = self.enrollment.trusted_name
        to_email: str = self.enrollment.trusted_email

        subject: str = "KeyDNA Account Recovery Request"

        body: str = f"""Hello {contact_name},

Someone is attempting to recover a KeyDNA account that listed you as a trusted contact.

If this is {username} asking you for help, please share the following code with them directly (in person or by phone):

    Recovery Code: {otp}

This code is valid for 10 minutes only.

If you do not recognise this request, please ignore this email. No action is needed on your part.

— KeyDNA Security
"""

        try:
            msg: MIMEMultipart = MIMEMultipart()
            msg['From'] = cfg.get('user', 'noreply@keydna.app')
            msg['To'] = to_email
            msg['Subject'] = subject  # OTP is NOT in the subject line
            msg.attach(MIMEText(body, 'plain'))

            server: smtplib.SMTP
            if cfg.get('use_tls', True):
                server = smtplib.SMTP(cfg['host'], cfg.get('port', 587))
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(cfg['host'], cfg.get('port', 465))

            server.login(cfg['user'], cfg['password'])
            server.sendmail(cfg['user'], to_email, msg.as_string())
            server.quit()
            return True, "Sent."

        except smtplib.SMTPAuthenticationError:
            return False, "SMTP authentication failed. Check email credentials."
        except smtplib.SMTPConnectError:
            return False, "Could not connect to SMTP server."
        except Exception as e:
            return False, str(e)

    def _config_from_env(self) -> Optional[Dict]:
        """Load SMTP config from environment variables if available."""
        host: Optional[str] = os.environ.get('KEYDNA_SMTP_HOST')
        user: Optional[str] = os.environ.get('KEYDNA_SMTP_USER')
        pwd: Optional[str] = os.environ.get('KEYDNA_SMTP_PASSWORD')

        if not all([host, user, pwd]):
            return None

        return {
            'host': host,
            'port': int(os.environ.get('KEYDNA_SMTP_PORT', 587)),
            'user': user,
            'password': pwd,
            'use_tls': True,
        }

    # ── Helpers ──

    def _status(
        self,
        success: bool,
        message: str,
        authenticated: bool = False,
    ) -> Dict:
        return {
            'success': success,
            'message': message,
            'stage': self.stage,
            'authenticated': authenticated,
            'failed': self.stage == self.STAGE_FAIL,
            'expired': self.stage == self.STAGE_EXPIRED,
            'attempts_used': self._attempts,
            'attempts_left': max(0, MAX_OTP_ATTEMPTS - self._attempts),
        }

    @property
    def is_success(self) -> bool:
        """True if OTP was verified successfully."""
        return self.stage == self.STAGE_SUCCESS

    @property
    def is_failed(self) -> bool:
        """True if recovery has failed."""
        return self.stage == self.STAGE_FAIL

    def time_remaining(self) -> int:
        """Seconds until OTP expires. 0 if not sent or expired."""
        if self.stage != self.STAGE_SENT:
            return 0
        return max(0, int(self._otp_expiry - time.time()))
