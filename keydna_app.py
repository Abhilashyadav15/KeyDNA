"""
KeyDNA — Desktop App (v2 — Unified Model)
Real keyboard capture via pynput bound to Tkinter Entry widgets.

ARCHITECTURE:
  NO mood classifier. Single unified model.
  27-feature extraction. One-Class SVM scoring.
  Strict thresholds. Information hiding.

ENROLLMENT:
  10 typing samples → all stored in single unified model.
  No mood prompts. Type your password naturally.

AUTHENTICATION:
  Password (bcrypt) + Typing Rhythm (SVM)
  Score > 0.00 ACCEPT | > -0.01 RETRY | <= -0.01 REJECT

SECURITY:
  - bcrypt password hashing (salted, slow by design)
  - PBKDF2 salted security question answers
  - Fernet AES-encrypted profile storage
  - Never show confidence or score to user
  - Generic "Authentication failed" on all rejections
  - Adaptive attack detection
  - Replay detection
  - 3 attempts then fallback
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Tuple

# ── Dependency checks ──────────────────────────────────────────────────
missing = []
try:
    import customtkinter as ctk
except ImportError:
    missing.append("customtkinter")

try:
    from pynput import keyboard as pynput_kb
except ImportError:
    missing.append("pynput")

try:
    import numpy as np
except ImportError:
    missing.append("numpy")


if missing:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror(
        "KeyDNA — Missing Dependencies",
        f"Please install:\n\npip install {' '.join(missing)}\n\n"
        "KeyDNA requires real keyboard capture (pynput) and cannot "
        "run without it.")
    sys.exit(1)

# ── Backend imports ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.features import FeatureExtractor
from core.security import (
    hash_password, verify_password,
    save_encrypted_profile, load_encrypted_profile,
)
from models.auth_model import UnifiedAuthModel
from enrollment.enroller import EnrollmentSession
from authentication.fallback import FallbackEnrollment, FallbackSession
from config import (
    ENROLLMENT_SAMPLES, MAX_ATTEMPTS, SESSION_TIMEOUT, DATA_FILE,
    SECURITY_QUESTIONS, QUESTIONS_TO_CHOOSE, THEME_FILE, DATA_DIR,
    C_BG, C_CARD, C_BORDER, C_ACCENT, C_TEXT, C_SUB,
    C_GREEN, C_RED, C_YELLOW, C_PURPLE, C_DIM,
    WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT, WINDOW_DEFAULT_GEOMETRY,
)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def save_profile(p: dict) -> None:
    """Save profile as encrypted binary file."""
    save_encrypted_profile(p, DATA_FILE)

def load_profile() -> dict:
    """Load profile from encrypted file (with legacy JSON fallback)."""
    return load_encrypted_profile(DATA_FILE)

def rebuild_model(model_data: dict) -> UnifiedAuthModel:
    """Rebuild UnifiedAuthModel from saved profile data."""
    model: UnifiedAuthModel = UnifiedAuthModel()
    if model_data:
        samples_list = model_data if isinstance(model_data, list) else model_data.get('samples', [])
        model.load_samples(samples_list)
    return model

def pw_strength(pw: str) -> Tuple[int, str, str]:
    """Return (score, label, color) for password strength."""
    if not pw: return 0, "", C_BORDER
    s: int = 0
    if len(pw) >= 8:                              s += 1
    if any(c.isdigit() for c in pw):              s += 1
    if any(c in "!@#$%^&*_-+=" for c in pw):     s += 1
    labels: list = ["", "Weak", "Medium", "Strong"]
    colors: list = [C_BORDER, C_RED, C_YELLOW, C_GREEN]
    return s, labels[s], colors[s]

def _load_theme() -> str:
    """Load saved theme preference. Defaults to dark."""
    try:
        if os.path.exists(THEME_FILE):
            with open(THEME_FILE, "r") as f:
                theme = f.read().strip()
                if theme in ("dark", "light"):
                    return theme
    except Exception:
        pass
    return "dark"

def _save_theme(theme: str) -> None:
    """Persist theme preference."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(THEME_FILE, "w") as f:
        f.write(theme)


# ═══════════════════════════════════════════════════════════════════════
# REAL KEYSTROKE CAPTURE  (pynput bound to Tkinter Entry)
# ═══════════════════════════════════════════════════════════════════════

class KeystrokeCapture:
    """
    Captures real keyboard timing from a Tkinter Entry widget.
    Binds directly to the widget's KeyPress / KeyRelease events.
    Privacy: key content anonymized, only timing stored.
    """

    def __init__(self):
        self._press_times  = {}
        self._events       = []
        self._lock         = threading.Lock()
        self._key_counter  = 0
        self._key_map      = {}
        self.active        = False

    def attach(self, widget):
        """Bind to a Tkinter Entry widget."""
        widget.bind("<KeyPress>",   self._on_press,   add="+")
        widget.bind("<KeyRelease>", self._on_release, add="+")
        self.active = True

    def _anon_key(self, keysym: str) -> str:
        """Anonymize key to position id."""
        if keysym in ("BackSpace", "Delete"):
            return "backspace"
        with self._lock:
            if keysym not in self._key_map:
                self._key_counter += 1
                self._key_map[keysym] = f"key_{self._key_counter}"
            return self._key_map[keysym]

    def _on_press(self, event):
        key_id = self._anon_key(event.keysym)
        with self._lock:
            self._press_times[key_id] = time.perf_counter()

    def _on_release(self, event):
        key_id       = self._anon_key(event.keysym)
        release_time = time.perf_counter()
        with self._lock:
            press_time = self._press_times.pop(key_id, None)
        if press_time is None:
            return
        dwell = release_time - press_time
        if dwell < 0 or dwell > 2.0:
            return
        with self._lock:
            self._events.append({
                "press":   press_time,
                "release": release_time,
                "dwell":   dwell,
                "key_id":  key_id,
            })

    def get_events(self):
        with self._lock:
            return list(self._events)

    def reset(self):
        with self._lock:
            self._events.clear()
            self._press_times.clear()
            self._key_map.clear()
            self._key_counter = 0

    def get_stats(self):
        evs     = self.get_events()
        regular = [e for e in evs if e["key_id"] != "backspace"]
        if len(regular) < 2:
            return {"keys": len(regular), "avg_ms": 0, "ready": False}
        flights = []
        for i in range(len(regular) - 1):
            f = (regular[i+1]["press"] - regular[i]["release"]) * 1000
            if f >= 0: flights.append(f)
        avg = round(sum(flights)/len(flights), 1) if flights else 0
        return {"keys": len(regular), "avg_ms": avg,
                "ready": len(regular) >= 4}


# ═══════════════════════════════════════════════════════════════════════
# BASE FRAME  (shared by all screens)
# ═══════════════════════════════════════════════════════════════════════

class BaseFrame(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, fg_color=C_BG, **kwargs)
        self.app = app
        self.grid(row=0, column=0, sticky="nsew")

    def show(self):
        self.tkraise()
        self.on_show()

    def on_show(self):
        pass

    # ── Reusable widgets ──

    def make_logo(self, parent):
        ctk.CTkLabel(parent, text="🔐", font=ctk.CTkFont(size=40)).pack(pady=(0,4))
        ctk.CTkLabel(parent, text="KEYDNA",
                     font=ctk.CTkFont(family="Courier", size=22, weight="bold"),
                     text_color=C_TEXT).pack()
        ctk.CTkLabel(parent, text="your rhythm is your key",
                     font=ctk.CTkFont(size=11), text_color=C_DIM).pack(pady=(2,0))

    def make_card(self, parent, **kwargs):
        return ctk.CTkFrame(parent, fg_color=C_CARD,
                            border_color=C_BORDER, border_width=1,
                            corner_radius=14, **kwargs)

    def make_label(self, parent, text, size=13, color=C_TEXT, bold=False, **kw):
        font = ctk.CTkFont(size=size, weight="bold" if bold else "normal")
        return ctk.CTkLabel(parent, text=text, font=font,
                            text_color=color, **kw)

    def make_entry(self, parent, placeholder="", show="", **kw):
        return ctk.CTkEntry(parent,
                            placeholder_text=placeholder,
                            show=show,
                            fg_color="#0a0f1a",
                            border_color=C_BORDER,
                            text_color=C_TEXT,
                            placeholder_text_color=C_DIM,
                            font=ctk.CTkFont(family="Courier", size=13),
                            height=40, **kw)

    def make_btn(self, parent, text, command, color=C_ACCENT,
                 text_color="#080c14", **kw):
        return ctk.CTkButton(parent, text=text, command=command,
                             fg_color=color, hover_color="#33ddff",
                             text_color=text_color,
                             font=ctk.CTkFont(size=13, weight="bold"),
                             height=42, corner_radius=8, **kw)

    def make_link_btn(self, parent, text, command):
        return ctk.CTkButton(parent, text=text, command=command,
                             fg_color="transparent",
                             hover_color=C_BORDER,
                             text_color="#4a90d9",
                             font=ctk.CTkFont(size=12),
                             height=30, corner_radius=6)

    def msg(self, parent, text, kind="info"):
        colors = {"ok":   (C_GREEN,  "#0a2218"),
                  "err":  (C_RED,    "#1a0808"),
                  "warn": (C_YELLOW, "#1a1200"),
                  "info": (C_ACCENT, "#0a1628")}
        border, bg = colors.get(kind, colors["info"])
        f = ctk.CTkFrame(parent, fg_color=bg,
                         border_color=border, border_width=1,
                         corner_radius=8)
        f.pack(fill="x", pady=4)
        ctk.CTkLabel(f, text=text, font=ctk.CTkFont(size=12),
                     text_color=C_TEXT, wraplength=380,
                     justify="left").pack(padx=12, pady=8)
        return f

    def clear_msg(self, widget):
        if widget and widget.winfo_exists():
            widget.destroy()

    def step_dots(self, parent, total, current):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(pady=(0, 10))
        for i in range(total):
            color = C_GREEN   if i < current - 1 else \
                    C_ACCENT  if i == current - 1 else \
                    C_BORDER
            ctk.CTkLabel(f, text="●", font=ctk.CTkFont(size=10),
                         text_color=color).pack(side="left", padx=3)

    def prog_bar(self, parent, value: float, color=C_ACCENT):
        outer = ctk.CTkFrame(parent, fg_color=C_BORDER,
                             height=4, corner_radius=2)
        outer.pack(fill="x", pady=(0, 12))
        outer.pack_propagate(False)
        pct = max(0, min(1, value))
        if pct > 0:
            inner = ctk.CTkFrame(outer, fg_color=color,
                                 height=4, corner_radius=2)
            inner.place(relx=0, rely=0, relwidth=pct, relheight=1)
        return outer

    def attempt_dots(self, parent, used, total=3):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(anchor="w", pady=(4, 0))
        for i in range(total):
            color = C_RED if i < used else C_BORDER
            ctk.CTkLabel(f, text="●", font=ctk.CTkFont(size=11),
                         text_color=color).pack(side="left", padx=2)
        return f


# ═══════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════════

class LoginScreen(BaseFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._capture        = KeystrokeCapture()
        self._msg_widget     = None
        self._dots_frame     = None
        self._auth_model     = None
        self._auto_submit_id = None
        self._build()

    def _build(self):
        wrap = ctk.CTkFrame(self, fg_color=C_BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.88)

        self.make_logo(wrap)
        ctk.CTkLabel(wrap, text="", height=16).pack()

        card = self.make_card(wrap)
        card.pack(fill="x", pady=6)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=20)

        self.make_label(inner, "SIGN IN", size=10,
                        color=C_DIM).pack(anchor="w", pady=(0,8))

        self._pw_entry = self.make_entry(inner, placeholder="Enter your password",
                                         show="●")
        self._pw_entry.pack(fill="x", pady=(0, 6))
        self._capture.attach(self._pw_entry)
        self._pw_entry.bind("<KeyRelease>", self._auto_submit, add="+")
        self._pw_entry.bind("<Return>", lambda e: self._on_enter())

        self._msg_area = ctk.CTkFrame(inner, fg_color="transparent")
        self._msg_area.pack(fill="x")

        self.make_btn(inner, "ENTER", self._on_enter).pack(fill="x", pady=(8,0))

        sep = ctk.CTkFrame(inner, fg_color=C_BORDER, height=1)
        sep.pack(fill="x", pady=12)

        self._dots_area = ctk.CTkFrame(inner, fg_color="transparent")
        self._dots_area.pack(anchor="w")

        links = ctk.CTkFrame(inner, fg_color="transparent")
        links.pack(fill="x", pady=(8,0))
        self.make_link_btn(links, "Forgot password?",
                           self._on_forgot).pack(side="left")
        self.make_link_btn(links, "New enrollment →",
                           self._on_enroll).pack(side="right")

    def on_show(self):
        # Cancel any pending auto-submit from previous session
        if self._auto_submit_id:
            self.after_cancel(self._auto_submit_id)
            self._auto_submit_id = None

        self._capture.reset()
        self._pw_entry.delete(0, "end")
        self._clear_msg()
        self._refresh_dots()
        
        # Initialize model once per login screen session to maintain recent_scores history
        profile = self.app.profile
        model_data = profile.get("model_data")
        if model_data:
            self._auth_model = rebuild_model(model_data)
        else:
            self._auth_model = None

    def _refresh_dots(self):
        for w in self._dots_area.winfo_children():
            w.destroy()
        fa = self.app.failed_attempts
        for i in range(MAX_ATTEMPTS):
            color = C_RED if i < fa else C_BORDER
            ctk.CTkLabel(self._dots_area, text="●",
                         font=ctk.CTkFont(size=11),
                         text_color=color).pack(side="left", padx=2)

    def _show_msg(self, text, kind="err"):
        self._clear_msg()
        self._msg_widget = self.msg(self._msg_area, text, kind)

    def _clear_msg(self):
        for w in self._msg_area.winfo_children():
            w.destroy()

    def _auto_submit(self, event=None):
        """Check on every keystroke — password hash acts as the gate."""
        if self._auto_submit_id:
            self.after_cancel(self._auto_submit_id)
        self._auto_submit_id = self.after(50, self._do_auto_submit)

    def _do_auto_submit(self):
        """
        SHA-256 hash gate auto-submit.
        - On every keystroke, check SHA-256(typed) vs stored hash (< 1ms)
        - Hash doesn't match → user still typing → do nothing
        - Hash matches → password complete → check rhythm → login or reject
        """
        self._auto_submit_id = None

        if not self._auth_model:
            return

        password = self._pw_entry.get()
        if not password:
            return

        profile = self.app.profile
        if not profile.get("password_hash"):
            return

        # ── Hash gate: only proceeds when EXACT password is typed ──
        if not verify_password(password, profile.get("password_hash")):
            return  # partial/wrong password → do nothing, wait for more typing

        # ── Password complete! Check rhythm immediately ──
        events = self._capture.get_events()
        if len(events) < 4:
            return

        extractor = FeatureExtractor()
        features  = extractor.extract(events)

        if features is None:
            return

        # Replay detection
        if extractor.get_consistency_score(features) >= 0.97:
            self.app.failed_attempts += 1
            self._refresh_dots()
            self._show_msg("Authentication failed.", "err")
            self._pw_entry.delete(0, "end")
            self._capture.reset()
            return

        result = self._auth_model.authenticate(features)

        if result["decision"] == "ACCEPT":
            # Password + rhythm matched → auto-login!
            self.app.failed_attempts = 0
            self.app.auth_verified = True
            self.app.show("success")
        else:
            # Password correct but rhythm rejected → immediate reject
            self.app.failed_attempts += 1
            fa = self.app.failed_attempts
            self._refresh_dots()
            if fa >= MAX_ATTEMPTS:
                self.app.failed_attempts = 0
                self._go_fallback()
            else:
                self._show_msg("Authentication failed.", "err")
            self._pw_entry.delete(0, "end")
            self._capture.reset()


    def _on_enter(self):
        events   = self._capture.get_events()
        password = self._pw_entry.get()
        profile  = self.app.profile

        if not profile.get("password_hash"):
            self._show_msg("No account found. Click New Enrollment.", "err")
            return
        if not password:
            self._show_msg("Please enter your password.", "err")
            return

        if not verify_password(password, profile.get("password_hash")):
            self.app.failed_attempts += 1
            fa = self.app.failed_attempts
            self._refresh_dots()
            if fa >= MAX_ATTEMPTS:
                self.app.failed_attempts = 0
                self._go_fallback()
            else:
                # SECURITY: Generic failure message
                self._show_msg("Authentication failed.", "err")
            self._pw_entry.delete(0, "end")
            self._capture.reset()
            return

        # Verify rhythm
        model_data = profile.get("model_data")
        extractor  = FeatureExtractor()

        if model_data and len(events) >= 4:
            features = extractor.extract(events)

            if features is not None:
                # Replay detection
                consistency = extractor.get_consistency_score(features)
                if consistency >= 0.97:
                    self.app.failed_attempts += 1
                    self._refresh_dots()
                    # SECURITY: Generic failure message — no details
                    self._show_msg("Authentication failed.", "err")
                    self._pw_entry.delete(0, "end")
                    self._capture.reset()
                    return

                if self._auth_model is None and model_data:
                    self._auth_model = rebuild_model(model_data)

                if self._auth_model is not None:
                    result = self._auth_model.authenticate(features)

                    if result["decision"] == "ACCEPT":
                        self.app.failed_attempts = 0
                        self.app.auth_verified = True
                        self.app.show("success")
                        return
                    else:
                        # REJECT or RETRY — treat both as failed attempt
                        self.app.failed_attempts += 1
                        fa = self.app.failed_attempts
                        self._refresh_dots()
                        if fa >= MAX_ATTEMPTS:
                            self.app.failed_attempts = 0
                            self._go_fallback()
                        else:
                            self._show_msg("Authentication failed.", "err")
                        self._pw_entry.delete(0, "end")
                        self._capture.reset()
                        return

                # Model couldn't be built — reject (should not happen normally)
                self.app.failed_attempts += 1
                fa = self.app.failed_attempts
                self._refresh_dots()
                if fa >= MAX_ATTEMPTS:
                    self.app.failed_attempts = 0
                    self._go_fallback()
                else:
                    self._show_msg("Authentication failed.", "err")
                self._pw_entry.delete(0, "end")
                self._capture.reset()
                return

        # No rhythm enrolled or not enough keystroke events
        # If model is enrolled but events < 4, still reject (don't allow password-only bypass)
        model_data = profile.get("model_data")
        if model_data:
            self.app.failed_attempts += 1
            fa = self.app.failed_attempts
            self._refresh_dots()
            if fa >= MAX_ATTEMPTS:
                self.app.failed_attempts = 0
                self._go_fallback()
            else:
                self._show_msg("Authentication failed. Type your full password.", "err")
            self._pw_entry.delete(0, "end")
            self._capture.reset()
            return

        # Truly no rhythm enrolled — password alone (first-time setup only)
        self.app.failed_attempts = 0
        self.app.auth_verified   = False
        self.app.show("success")

    def _go_fallback(self):
        profile = self.app.profile
        if profile.get("fallback"):
            self.app.fallback_session = FallbackSession(
                FallbackEnrollment.from_dict(profile["fallback"]))
            self.app.show("fallback")
        else:
            self._show_msg(
                "Too many failed attempts. No fallback enrolled. "
                "Please contact support.", "err")

    def _on_forgot(self):
        if not self.app.profile.get("password_hash"):
            self._show_msg("No account found. Please enroll first.", "err")
            return
        self._go_fallback()

    def _on_enroll(self):
        if self.app.profile.get("password_hash"):
            if not getattr(self, "_overwrite_warned", False):
                self._show_msg(
                    "⚠ Account already exists. Click again to overwrite.", "warn")
                self._overwrite_warned = True
                return
        self._overwrite_warned = False
        self.app.show("enroll")


# ═══════════════════════════════════════════════════════════════════════
# ENROLL SCREEN  (Unified — no mood classification)
# ═══════════════════════════════════════════════════════════════════════

class EnrollScreen(BaseFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._step           = 1
        self._password       = ""
        self._fe             = FallbackEnrollment()
        self._samples        = []     # collected feature vectors
        self._enroll_session = None
        self._capture        = KeystrokeCapture()
        self._q_order        = []
        self._q_answers      = []
        self._content_frame  = None
        self._rhythm_only    = False   # skip questions when re-enrolling rhythm
        self._build_shell()

    def _build_shell(self):
        wrap = ctk.CTkScrollableFrame(self, fg_color=C_BG,
                                      scrollbar_button_color=C_BORDER)
        wrap.place(relx=0.5, rely=0.5, anchor="center",
                   relwidth=0.9, relheight=0.95)
        self._wrap = wrap

        self.make_logo(wrap)
        ctk.CTkLabel(wrap, text="", height=8).pack()

        # Step dots + progress
        self._dots_row = ctk.CTkFrame(wrap, fg_color="transparent")
        self._dots_row.pack()
        self._prog_outer = ctk.CTkFrame(wrap, fg_color=C_BORDER,
                                        height=4, corner_radius=2)
        self._prog_outer.pack(fill="x", pady=(4, 10))
        self._prog_inner = ctk.CTkFrame(self._prog_outer, fg_color=C_ACCENT,
                                        height=4, corner_radius=2)

        # Card
        self._card = self.make_card(wrap)
        self._card.pack(fill="x", pady=4)

        self._back_btn = self.make_link_btn(wrap, "← Back to login",
                                            lambda: self.app.show("login"))
        self._back_btn.pack(pady=8)

    def on_show(self):
        self._step          = 1
        self._password      = ""
        self._fe            = FallbackEnrollment()
        self._samples       = []
        self._q_order       = []
        self._q_answers     = []
        self._trusted_name  = ""
        self._trusted_email = ""
        self._rhythm_only   = False
        self._enroll_session = None
        self._render()

    def _render(self):
        TOTAL_STEPS = 4

        # Update dots
        for w in self._dots_row.winfo_children(): w.destroy()
        for i in range(TOTAL_STEPS):
            color = C_GREEN  if i < self._step-1 else \
                    C_ACCENT if i == self._step-1 else C_BORDER
            ctk.CTkLabel(self._dots_row, text="●",
                         font=ctk.CTkFont(size=10),
                         text_color=color).pack(side="left", padx=3)

        # Update progress bar
        pct = (self._step - 1) / TOTAL_STEPS
        self._prog_inner.place(relx=0, rely=0,
                               relwidth=max(0.01, pct), relheight=1)

        # Clear card
        for w in self._card.winfo_children(): w.destroy()

        inner = ctk.CTkFrame(self._card, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=20)

        builders = {1: self._build_step1,
                    2: self._build_step2,
                    3: self._build_step3,
                    4: self._build_step4}
        builders[self._step](inner)

    # ── Step 1: Password ────────────────────────────────────────────
    def _build_step1(self, p):
        self.make_label(p, "STEP 1 OF 4 — SET YOUR PASSWORD",
                        size=10, color=C_DIM).pack(anchor="w", pady=(0,10))

        self.make_label(p, "New password").pack(anchor="w")
        self._e_pw1 = self.make_entry(p, show="●")
        self._e_pw1.pack(fill="x", pady=(2,8))

        self.make_label(p, "Confirm password").pack(anchor="w")
        self._e_pw2 = self.make_entry(p, show="●")
        self._e_pw2.pack(fill="x", pady=(2,8))

        # Strength bar
        self._str_bar = ctk.CTkProgressBar(p, height=4,
                                           fg_color=C_BORDER,
                                           progress_color=C_BORDER)
        self._str_bar.set(0)
        self._str_bar.pack(fill="x", pady=(0,2))
        self._str_lbl = self.make_label(p, "", size=11, color=C_SUB)
        self._str_lbl.pack(anchor="w")

        self._e_pw1.bind("<KeyRelease>", self._update_strength)

        self._msg1 = ctk.CTkFrame(p, fg_color="transparent")
        self._msg1.pack(fill="x")

        self.make_btn(p, "CONTINUE", self._step1_next).pack(fill="x", pady=(10,0))

    def _update_strength(self, event=None):
        pw = self._e_pw1.get()
        sc, sl, col = pw_strength(pw)
        self._str_bar.configure(progress_color=col)
        self._str_bar.set(sc / 3)
        self._str_lbl.configure(text=sl, text_color=col)

    def _step1_next(self):
        for w in self._msg1.winfo_children(): w.destroy()
        pw1 = self._e_pw1.get()
        pw2 = self._e_pw2.get()
        if not pw1 or not pw2:
            self.msg(self._msg1, "Fill both fields.", "err"); return
        if len(pw1) < 6:
            self.msg(self._msg1, "At least 6 characters.", "err"); return
        if pw1 != pw2:
            self.msg(self._msg1, "Passwords don't match.", "err"); return
        self._password = pw1
        self._step = 2
        self._render()

    # ── Step 2: Unified Rhythm Enrollment ────────────────────────────
    def _build_step2(self, p):
        n = len(self._samples)
        remaining = ENROLLMENT_SAMPLES - n

        if n >= ENROLLMENT_SAMPLES:
            self._build_step2_done(p)
            return

        # ── Header ──
        self.make_label(p,
            f"STEP 2 OF 4 — RHYTHM ENROLLMENT",
            size=10, color=C_DIM).pack(anchor="w", pady=(0,10))

        self.make_label(p,
            "Type your password naturally. Each attempt teaches the\n"
            "system your unique typing rhythm. No rushing needed.",
            size=12, color=C_SUB).pack(anchor="w", pady=(0,14))

        # ── BIG COUNTDOWN NUMBER ──
        count_frame = ctk.CTkFrame(p, fg_color=C_CARD,
                                   border_color=C_ACCENT, border_width=2,
                                   corner_radius=16)
        count_frame.pack(fill="x", pady=(0, 12))

        inner_cf = ctk.CTkFrame(count_frame, fg_color="transparent")
        inner_cf.pack(pady=16)

        self._countdown_num = ctk.CTkLabel(
            inner_cf,
            text=str(remaining),
            font=ctk.CTkFont(family="Courier", size=64, weight="bold"),
            text_color=C_ACCENT)
        self._countdown_num.pack()

        self._countdown_sub = ctk.CTkLabel(
            inner_cf,
            text=f"sample{'s' if remaining != 1 else ''} remaining",
            font=ctk.CTkFont(size=12),
            text_color=C_SUB)
        self._countdown_sub.pack()

        # ── Password hint ──
        hint = "●" * len(self._password) if self._password else "your password"
        self.make_label(
            p,
            f"Type  {hint}  and press  Enter",
            size=12, color=C_SUB).pack(anchor="w", pady=(0, 6))

        # ── Password entry ──
        self._e_type = self.make_entry(
            p,
            placeholder="Type your password... then press Enter",
            show="●")
        self._e_type.pack(fill="x", pady=(0, 6))
        self._capture.reset()
        self._capture.attach(self._e_type)
        self._e_type.focus()

        # Pressing Enter captures the sample
        self._e_type.bind("<Return>", lambda e: self._capture_sample())

        # ── Feedback message area ──
        self._msg2 = ctk.CTkFrame(p, fg_color="transparent")
        self._msg2.pack(fill="x")

        # ── Enroll session ──
        if self._enroll_session is None:
            self._enroll_session = EnrollmentSession(ENROLLMENT_SAMPLES)

    def _capture_sample(self):
        """Called on Enter key press — captures one sample."""
        for w in self._msg2.winfo_children(): w.destroy()

        events = self._capture.get_events()
        self._capture.reset()
        self._e_type.delete(0, "end")
        self._e_type.focus()

        if len(events) < 4:
            self.msg(self._msg2,
                     "Too few keystrokes. Type your full password.", "err")
            return

        extractor = FeatureExtractor()
        features  = extractor.extract(events)

        if features is None:
            self.msg(self._msg2, "Could not extract features. Try again.", "err")
            return

        # Autofill / paste detection
        total_ms = (events[-1]["release"] - events[0]["press"]) * 1000
        cons     = extractor.get_consistency_score(features)
        if total_ms < 50 or cons >= 0.99:
            self.msg(self._msg2,
                     "Autofill or paste detected. Please type manually.", "warn")
            return

        # ── Store sample ──
        self._samples.append(features)

        n         = len(self._samples)
        remaining = ENROLLMENT_SAMPLES - n

        # ── Update countdown ──
        if remaining > 0:
            self._countdown_num.configure(
                text=str(remaining),
                text_color=C_ACCENT if remaining > 3 else C_YELLOW)
            self._countdown_sub.configure(
                text=f"sample{'s' if remaining != 1 else ''} remaining")
            self.msg(self._msg2, f"✓  Sample {n} captured", "ok")
        else:
            # All done — flash green then move on
            self._countdown_num.configure(text="✓", text_color=C_GREEN)
            self._countdown_sub.configure(
                text="Enrollment complete!", text_color=C_GREEN)
            self.msg(self._msg2,
                     f"All {ENROLLMENT_SAMPLES} samples captured ✓", "ok")
            self.after(1200, self._finish_rhythm)

    def _finish_rhythm(self):
        self._step = 2  # will show done view
        self._render()

    def _build_step2_done(self, p):
        self.make_label(p, "STEP 2 OF 4 — RHYTHM ENROLLMENT",
                        size=10, color=C_DIM).pack(anchor="w", pady=(0,10))
        self.msg(p, f"✓ {len(self._samples)} rhythm samples captured.", "ok")

        self.make_label(
            p,
            "Building your personal authentication model\n"
            "from real captured timing data...",
            size=11, color=C_SUB).pack(pady=(10,0))

        self.make_btn(p, "BUILD MODEL & CONTINUE", self._build_model).pack(
            fill="x", pady=(10,0))

    def _build_model(self):
        if not self._samples or len(self._samples) < 5:
            messagebox.showerror("KeyDNA",
                "Not enough samples. Please provide at least 5.")
            return

        # Build unified model
        model = UnifiedAuthModel()
        for s in self._samples:
            model.enroll(s)

        self.app.auth_model   = model
        self.app.model_data   = model.get_samples_serializable()

        # If rhythm-only re-enrollment, skip questions/contact — save directly
        if self._rhythm_only:
            self._rhythm_only = False
            self._do_save()
            return

        self._step = 3
        self._render()

    # ── Step 3: Backup PIN + Security Questions ──────────────────────
    def _build_step3(self, p):
        q_num = len(self._q_order) + 1

        if q_num > QUESTIONS_TO_CHOOSE:
            self._build_step3_confirm(p)
            return

        # Question sub-step
        self.make_label(p, "STEP 3 OF 4 — SECURITY QUESTIONS",
                        size=10, color=C_DIM).pack(anchor="w", pady=(0,6))
        self.make_label(
            p,
            f"Choose question {q_num} of {QUESTIONS_TO_CHOOSE} and answer it.\n"
            "Remember this order — required during fallback login.",
            size=12, color=C_SUB).pack(anchor="w", pady=(0,10))

        used      = list(self._q_order)
        remaining = [(i, q) for i, q in enumerate(SECURITY_QUESTIONS)
                     if i not in used]

        self._q_var = tk.StringVar(value=f"Q{remaining[0][0]+1}: {remaining[0][1]}")
        for i, q in remaining:
            ctk.CTkRadioButton(
                p, text=f"Q{i+1}: {q}",
                variable=self._q_var,
                value=f"Q{i+1}: {q}",
                font=ctk.CTkFont(size=12),
                text_color=C_TEXT,
                fg_color=C_ACCENT,
                hover_color="#33ddff").pack(anchor="w", pady=3)

        self.make_label(p, "Your answer").pack(anchor="w", pady=(10,0))
        self._e_ans = self.make_entry(p, placeholder="Type your answer...")
        self._e_ans.pack(fill="x", pady=(2,8))

        self._msg3q = ctk.CTkFrame(p, fg_color="transparent")
        self._msg3q.pack(fill="x")

        btn_lbl = "ADD QUESTION" if q_num < QUESTIONS_TO_CHOOSE else "DONE"
        self.make_btn(p, btn_lbl, self._step3_add_question).pack(
            fill="x", pady=(8,0))

        self._q_remaining = remaining

    def _step3_add_question(self):
        for w in self._msg3q.winfo_children(): w.destroy()
        sel_label = self._q_var.get()
        sel_idx   = next(i for i, q in self._q_remaining
                         if f"Q{i+1}: {q}" == sel_label)
        ans = self._e_ans.get().strip()
        if len(ans) < 2:
            self.msg(self._msg3q, "Answer too short.", "err"); return
        self._q_order.append(sel_idx)
        self._q_answers.append(ans)
        self._render()

    def _build_step3_confirm(self, p):
        self.make_label(p, "STEP 3 OF 4 — CONFIRM",
                        size=10, color=C_DIM).pack(anchor="w", pady=(0,10))
        self.make_label(p, "All 3 questions chosen ✓",
                        size=13, color=C_GREEN, bold=True).pack(anchor="w", pady=(0,8))

        for i, (qi, a) in enumerate(zip(self._q_order, self._q_answers)):
            row = ctk.CTkFrame(p, fg_color=C_CARD,
                               border_color=C_BORDER, border_width=1,
                               corner_radius=8)
            row.pack(fill="x", pady=3)
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)
            ctk.CTkLabel(inner, text=f"#{i+1}",
                         font=ctk.CTkFont(family="Courier", size=12),
                         text_color=C_ACCENT).pack(side="left", padx=(0,8))
            ctk.CTkLabel(inner, text=SECURITY_QUESTIONS[qi],
                         font=ctk.CTkFont(size=11),
                         text_color=C_SUB).pack(side="left")
            ctk.CTkLabel(inner, text=f"→ {a}",
                         font=ctk.CTkFont(size=11),
                         text_color=C_GREEN).pack(side="right")

        self._msg3c = ctk.CTkFrame(p, fg_color="transparent")
        self._msg3c.pack(fill="x", pady=6)

        self.make_btn(p, "SAVE & CONTINUE", self._save_questions).pack(
            fill="x", pady=(8,0))

    def _save_questions(self):
        for w in self._msg3c.winfo_children(): w.destroy()
        ok, msg = self._fe.set_questions(self._q_order, self._q_answers)
        if not ok:
            self.msg(self._msg3c, msg, "err"); return
        self._step = 4
        self._render()

    # ── Step 4: Trusted Contact ─────────────────────────────────────
    def _build_step4(self, p):
        self.make_label(p, "STEP 4 OF 4 — TRUSTED CONTACT",
                        size=10, color=C_DIM).pack(anchor="w", pady=(0,10))
        self.make_label(
            p,
            "Last-resort recovery: if you forget your security\n"
            "questions, a one-time code will be sent to this person.\n"
            "They share it with you verbally to unlock your account.",
            size=12, color=C_SUB).pack(anchor="w", pady=(0,12))

        self.make_label(p, "Contact name (e.g. Mom, Dad, Friend)").pack(anchor="w")
        self._e_tname = self.make_entry(p, placeholder="e.g. Mom")
        self._e_tname.pack(fill="x", pady=(2,8))

        self.make_label(p, "Their email address").pack(anchor="w")
        self._e_temail = self.make_entry(p, placeholder="e.g. mom@example.com")
        self._e_temail.pack(fill="x", pady=(2,8))

        self._msg4 = ctk.CTkFrame(p, fg_color="transparent")
        self._msg4.pack(fill="x")

        self.make_btn(p, "SAVE & ACTIVATE", self._finish_enrollment).pack(
            fill="x", pady=(10,0))
        self.make_link_btn(p, "Skip (not recommended)",
                           self._finish_enrollment_no_recovery).pack(pady=(6,0))

    def _finish_enrollment(self):
        for w in self._msg4.winfo_children(): w.destroy()
        name  = self._e_tname.get().strip()
        email = self._e_temail.get().strip()
        ok, msg = self._fe.recovery.set_trusted_contact(name, email)
        if not ok:
            self.msg(self._msg4, msg, "err"); return
        self._do_save()

    def _finish_enrollment_no_recovery(self):
        """Allow skipping trusted contact — recovery won't be available."""
        self._do_save()

    def _do_save(self):
        model_data = getattr(self.app, "model_data", [])
        existing   = self.app.profile or {}

        fallback_dict = self._fe.to_dict() if self._fe.is_complete() \
                        else existing.get("fallback", {})

        pw_hash = hash_password(self._password) if self._password \
                  else existing.get("password_hash", "")

        profile = {
            "password_hash": pw_hash,
            "fallback":      fallback_dict,
            "model_data":    model_data,
            "model_enrolled": bool(model_data),
            "enrolled_at":   time.time(),
        }
        save_profile(profile)
        self.app.profile = profile
        self.app.show("login")


# ═══════════════════════════════════════════════════════════════════════
# FALLBACK SCREEN
# ═══════════════════════════════════════════════════════════════════════

class FallbackScreen(BaseFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._build_shell()

    def _build_shell(self):
        self._wrap = ctk.CTkScrollableFrame(self, fg_color=C_BG,
                                            scrollbar_button_color=C_BORDER)
        self._wrap.place(relx=0.5, rely=0.5, anchor="center",
                         relwidth=0.9, relheight=0.95)
        self.make_logo(self._wrap)
        ctk.CTkLabel(self._wrap, text="", height=8).pack()
        self._dots_row   = ctk.CTkFrame(self._wrap, fg_color="transparent")
        self._dots_row.pack()
        self._prog_outer = ctk.CTkFrame(self._wrap, fg_color=C_BORDER,
                                        height=4, corner_radius=2)
        self._prog_outer.pack(fill="x", pady=(4,10))
        self._prog_inner = ctk.CTkFrame(self._prog_outer, fg_color=C_ACCENT,
                                        height=4, corner_radius=2)
        self._card = self.make_card(self._wrap)
        self._card.pack(fill="x", pady=4)
        self.make_btn(self._wrap, "Back to login page",
                           self._back).pack(pady=8)

    def on_show(self):
        self._render()

    def _render(self):
        session = self.app.fallback_session
        if not session:
            self.app.show("login"); return

        stage   = session.stage
        cur_map = {FallbackSession.STAGE_Q1:0,
                   FallbackSession.STAGE_Q2:1, FallbackSession.STAGE_Q3:2,
                   FallbackSession.STAGE_RECOVERY:3, FallbackSession.STAGE_DONE:3}
        cur = cur_map.get(stage, 0)

        # Dots
        for w in self._dots_row.winfo_children(): w.destroy()
        for i in range(4):
            color = C_GREEN  if i < cur else \
                    C_ACCENT if i == cur else C_BORDER
            ctk.CTkLabel(self._dots_row, text="●",
                         font=ctk.CTkFont(size=10),
                         text_color=color).pack(side="left", padx=3)

        # Progress
        self._prog_inner.place(relx=0, rely=0,
                               relwidth=max(0.01, cur/3), relheight=1)

        # Card content
        for w in self._card.winfo_children(): w.destroy()
        inner = ctk.CTkFrame(self._card, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=20)

        if stage in (FallbackSession.STAGE_Q1,
                       FallbackSession.STAGE_Q2,
                       FallbackSession.STAGE_Q3):
            self._build_question(inner, stage)
        elif stage == FallbackSession.STAGE_RECOVERY:
            self._build_recovery(inner)
        elif stage == FallbackSession.STAGE_FAIL:
            self._build_fail(inner)

    def _build_question(self, p, stage):
        q_num   = {FallbackSession.STAGE_Q1:1,
                   FallbackSession.STAGE_Q2:2,
                   FallbackSession.STAGE_Q3:3}[stage]
        ordinal = ["first","second","third"][q_num-1]

        self.make_label(
            p,
            f"FALLBACK STEP {q_num} OF 4 — QUESTION {q_num}",
            size=10, color=C_DIM).pack(anchor="w", pady=(0,6))
        self.make_label(
            p,
            f"Choose the question you picked {ordinal} during enrollment,\n"
            f"then type your answer.",
            size=12, color=C_SUB).pack(anchor="w", pady=(0,10))

        available = self.app.fallback_session._available_questions()
        self._q_var = tk.StringVar(
            value=f"Q{available[0]['index']+1}: {available[0]['question']}")

        for q in available:
            ctk.CTkRadioButton(
                p,
                text=f"Q{q['index']+1}: {q['question']}",
                variable=self._q_var,
                value=f"Q{q['index']+1}: {q['question']}",
                font=ctk.CTkFont(size=12),
                text_color=C_TEXT,
                fg_color=C_ACCENT,
                hover_color="#33ddff").pack(anchor="w", pady=3)

        self.make_label(p, "Answer").pack(anchor="w", pady=(10,0))
        self._e_qans = self.make_entry(p, placeholder="Type your answer...")
        self._e_qans.pack(fill="x", pady=(2,8))
        self._e_qans.focus()

        self._q_available = available
        self._fb_msg2 = ctk.CTkFrame(p, fg_color="transparent")
        self._fb_msg2.pack(fill="x")

        self.make_btn(p, "VERIFY", self._verify_question).pack(
            fill="x", pady=(8,0))

    def _verify_question(self):
        for w in self._fb_msg2.winfo_children(): w.destroy()
        sel_label = self._q_var.get()
        sel_idx   = next(
            q["index"] for q in self._q_available
            if f"Q{q['index']+1}: {q['question']}" == sel_label)

        r = self.app.fallback_session.verify_question(
            sel_idx, self._e_qans.get())

        if r.get("authenticated"):
            self.app.fallback_session = None
            self.app.needs_reenroll   = True
            self.app.auth_verified    = False
            self.app.show("success")
        elif r["success"]:
            self._render()
        elif r.get("recovery_available"):
            self._render()
        elif r.get("failed"):
            self.app.fallback_session = None
            self.app.show("login")
        else:
            self.msg(self._fb_msg2, r["message"], "err")

    def _build_recovery(self, p):
        """Parental OTP recovery screen."""
        session = self.app.fallback_session
        has_contact = session.enrollment.has_recovery()

        self.make_label(p, "LAST RESORT — TRUSTED CONTACT RECOVERY",
                        size=10, color=C_DIM).pack(anchor="w", pady=(0,10))

        if not has_contact:
            self.msg(p,
                "No trusted contact was enrolled for this account.\n"
                "Account is fully locked. Contact support for a manual reset.",
                "err")
            self.make_btn(p, "BACK TO LOGIN",
                          lambda: self.app.show("login")).pack(fill="x", pady=(12,0))
            return

        contact = session.enrollment.recovery
        self.make_label(
            p,
            f"A one-time recovery code will be sent to {contact.trusted_name}\n"
            f"({contact.masked_email()}).\n"
            f"Ask them to share the code with you. Valid for 10 minutes.",
            size=12, color=C_SUB).pack(anchor="w", pady=(0,12))

        self._rec_msg = ctk.CTkFrame(p, fg_color="transparent")
        self._rec_msg.pack(fill="x")

        self._rec_otp_frame = ctk.CTkFrame(p, fg_color="transparent")
        self._rec_otp_frame.pack(fill="x")

        self.make_btn(p, f"SEND CODE TO {contact.trusted_name.upper()}",
                      self._send_recovery_otp).pack(fill="x", pady=(10,4))

    def _send_recovery_otp(self):
        for w in self._rec_msg.winfo_children(): w.destroy()
        for w in self._rec_otp_frame.winfo_children(): w.destroy()

        r = self.app.fallback_session.request_recovery_otp()
        if r["success"]:
            self.msg(self._rec_msg, r["message"], "ok")
            self.make_label(self._rec_otp_frame,
                            "Enter the code shared by your trusted contact:"
                            ).pack(anchor="w", pady=(12,0))
            self._e_otp = self.make_entry(self._rec_otp_frame,
                                          placeholder="6-digit code")
            self._e_otp.pack(fill="x", pady=(2,8))
            self._otp_msg = ctk.CTkFrame(self._rec_otp_frame, fg_color="transparent")
            self._otp_msg.pack(fill="x")
            self.make_btn(self._rec_otp_frame, "VERIFY CODE",
                          self._verify_recovery_otp).pack(fill="x", pady=(8,0))
        else:
            self.msg(self._rec_msg, r["message"], "err")

    def _verify_recovery_otp(self):
        for w in self._otp_msg.winfo_children(): w.destroy()
        r = self.app.fallback_session.verify_recovery_otp(self._e_otp.get())

        if r.get("authenticated"):
            self.app.fallback_session = None
            self.app.needs_reenroll   = True
            self.app.auth_verified    = False
            self.app.show("success")
        elif r.get("failed"):
            self.app.fallback_session = None
            self._render()
        elif r.get("expired"):
            self.msg(self._otp_msg, r["message"], "warn")
        else:
            self.msg(self._otp_msg, r["message"], "err")

    def _build_fail(self, p):
        self.msg(p, "Fallback failed. Contact support to reset your account.", "err")
        self.make_btn(p, "BACK TO LOGIN",
                      lambda: self.app.show("login")).pack(fill="x", pady=(12,0))

    def _back(self):
        """Go back to login and reset attempts so user gets another 3 tries."""
        self.app.fallback_session = None
        self.app.failed_attempts  = 0
        self.app.show("login")


# ═══════════════════════════════════════════════════════════════════════
# SUCCESS SCREEN
# ═══════════════════════════════════════════════════════════════════════

class SuccessScreen(BaseFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._timer_id      = None
        self._reenroll_row  = None
        self._build()

    def _build(self):
        wrap = ctk.CTkFrame(self, fg_color=C_BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.88)

        ctk.CTkLabel(wrap, text="✅",
                     font=ctk.CTkFont(size=52)).pack(pady=(0,8))
        ctk.CTkLabel(wrap, text="You are in.",
                     font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=C_GREEN).pack()
        ctk.CTkLabel(wrap, text="authentication successful",
                     font=ctk.CTkFont(family="Courier", size=11),
                     text_color=C_DIM).pack(pady=(4,0))

        # SECURITY: No confidence or score shown — only verified status
        self._status_lbl = ctk.CTkLabel(
            wrap, text="",
            font=ctk.CTkFont(family="Courier", size=12),
            text_color=C_GREEN)
        self._status_lbl.pack(pady=(12,0))

        self._timeout_lbl = ctk.CTkLabel(
            wrap, text="",
            font=ctk.CTkFont(family="Courier", size=11),
            text_color=C_SUB)
        self._timeout_lbl.pack(pady=(4,0))

        self._reenroll_card = ctk.CTkFrame(wrap, fg_color="transparent")
        self._reenroll_card.pack(fill="x", pady=12)

        ctk.CTkLabel(wrap, text="", height=8).pack()

        # ── Re-enroll rhythm button ──
        self.make_btn(wrap, "RE-ENROLL RHYTHM", self._manual_reenroll,
                      color="#1a2a1a",
                      text_color="#66bb6a").pack(fill="x", pady=(0,4))
        ctk.CTkLabel(wrap,
                     text="Re-capture your typing rhythm (keeps questions & contact)",
                     font=ctk.CTkFont(size=10),
                     text_color=C_SUB).pack(pady=(0,8))

        self.make_btn(wrap, "SIGN OUT", self._sign_out,
                      color=C_BORDER, text_color=C_TEXT).pack(
                      fill="x", pady=4)

    def on_show(self):
        # SECURITY: Only show verified/not-verified — no score/confidence
        if self.app.auth_verified:
            self._status_lbl.configure(
                text="Rhythm verified ✓",
                text_color=C_GREEN)
        else:
            self._status_lbl.configure(text="")

        # Re-enroll prompt
        for w in self._reenroll_card.winfo_children(): w.destroy()
        if self.app.needs_reenroll:
            f = ctk.CTkFrame(self._reenroll_card,
                             fg_color="#1a1200",
                             border_color=C_YELLOW, border_width=1,
                             corner_radius=10)
            f.pack(fill="x")
            ctk.CTkLabel(f,
                text="⚠  Rhythm failed during login. Re-enroll to avoid fallback next time.",
                font=ctk.CTkFont(size=12), text_color=C_YELLOW,
                wraplength=360).pack(padx=14, pady=(10,6))
            row = ctk.CTkFrame(f, fg_color="transparent")
            row.pack(pady=(0,10))
            self.make_btn(row, "RE-ENROLL RHYTHM",
                          self._reenroll, color=C_YELLOW,
                          text_color="#080c14").pack(side="left", padx=4)
            self.make_link_btn(row, "Skip for now",
                               self._skip_reenroll).pack(side="left")

        # Session timeout countdown
        self.app.session_start = time.time()
        self._tick()

    def _tick(self):
        if self._timer_id:
            self.after_cancel(self._timer_id)
        elapsed   = time.time() - self.app.session_start
        remaining = SESSION_TIMEOUT - int(elapsed)
        if remaining <= 0:
            self._sign_out()
            return
        mins = remaining // 60
        secs = remaining % 60
        self._timeout_lbl.configure(
            text=f"Session expires in  {mins:02d}:{secs:02d}")
        self._timer_id = self.after(1000, self._tick)

    def _reenroll(self):
        """Called after rhythm failed at login — forced re-enroll."""
        self.app.needs_reenroll = False
        if self._timer_id: self.after_cancel(self._timer_id)

        # Set up state directly — bypass on_show() to keep _step = 2
        enroll                 = self.app.frames["enroll"]
        enroll._step           = 2
        enroll._samples        = []
        enroll._enroll_session = None
        enroll._rhythm_only    = True
        enroll._password       = ""
        enroll._capture        = KeystrokeCapture()
        enroll.tkraise()
        enroll._render()

    def _manual_reenroll(self):
        """
        Manual re-enroll triggered by user from success screen.
        Clears only model data from the saved profile.
        Keeps password, security questions, and trusted contact intact.
        Jumps straight to Step 2 (rhythm capture).
        """
        if self._timer_id: self.after_cancel(self._timer_id)

        # Clear model data from saved profile — keep everything else
        profile = self.app.profile or {}
        profile["model_data"]    = []
        profile["model_enrolled"] = False
        save_profile(profile)
        self.app.profile    = profile
        self.app.auth_model = None
        self.app.model_data = []

        # Set up state directly — bypass on_show() to keep _step = 2
        enroll                 = self.app.frames["enroll"]
        enroll._step           = 2
        enroll._samples        = []
        enroll._enroll_session = None
        enroll._rhythm_only    = True
        enroll._password       = ""
        enroll._capture        = KeystrokeCapture()
        enroll._fe = FallbackEnrollment.from_dict(
            profile.get("fallback", {}))
        self.app.needs_reenroll = False
        enroll.tkraise()
        enroll._render()

    def _skip_reenroll(self):
        self.app.needs_reenroll = False
        for w in self._reenroll_card.winfo_children(): w.destroy()

    def _sign_out(self):
        if self._timer_id: self.after_cancel(self._timer_id)
        self.app.failed_attempts  = 0
        self.app.auth_verified    = False
        self.app.needs_reenroll   = False
        self.app.fallback_session = None
        self.app.show("login")


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════

class KeyDNAApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.current_theme = _load_theme()
        ctk.set_appearance_mode(self.current_theme)
        ctk.set_default_color_theme("blue")

        self.title("KeyDNA — Behavioral Authentication")
        self.geometry(WINDOW_DEFAULT_GEOMETRY)
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resizable(True, True)
        self.configure(fg_color=C_BG)

        # Theme toggle button
        self.theme_btn = ctk.CTkButton(
            self, text="☀️" if self.current_theme == "dark" else "🌙",
            width=30, height=30, fg_color="transparent", hover_color=C_BORDER,
            command=self.toggle_theme
        )
        self.theme_btn.place(relx=0.95, rely=0.02, anchor="ne")

        # App state
        self.profile          = load_profile()
        self.failed_attempts  = 0
        self.auth_verified    = False
        self.needs_reenroll   = False
        self.fallback_session = None
        self.auth_model       = None
        self.model_data       = []
        self.session_start    = time.time()

        # NO mood classifier — unified model only

        # Grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Build all screens
        self.frames = {
            "login":    LoginScreen(self, self),
            "enroll":   EnrollScreen(self, self),
            "fallback": FallbackScreen(self, self),
            "success":  SuccessScreen(self, self),
        }

        self.show("login")

    def show(self, name: str):
        frame = self.frames[name]
        frame.show()

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        ctk.set_appearance_mode(self.current_theme)
        self.theme_btn.configure(text="☀️" if self.current_theme == "dark" else "🌙")
        _save_theme(self.current_theme)


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = KeyDNAApp()
    app.mainloop()
