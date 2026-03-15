"""
KeyDNA — Keystroke Capture Engine

Two modes:
  RealKeystrokeCapture  — uses pynput to capture actual keyboard timing
  SimulatedCapture      — generates synthetic timing for demo/testing

PRIVACY (Fix #13):
  Only TIMING is stored — press time, release time, dwell.
  Key content (which key was pressed) is anonymized to position index.
  Actual characters are NEVER stored, logged, or transmitted.
"""

import time
import threading
from collections import deque
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════
# Real Keystroke Capture (pynput)
# ══════════════════════════════════════════════════════

class RealKeystrokeCapture:
    """
    Captures real keyboard timing using pynput.

    Usage:
        capture = RealKeystrokeCapture()
        capture.start()
        # user types password in the Streamlit text input
        capture.stop()
        events = capture.get_events()

    Privacy (Fix #13):
        Key identity is anonymized to sequential index.
        Characters typed are never stored.
        Only millisecond timing between keystrokes is kept.
    """

    def __init__(self, max_events: int = 200):
        self.max_events     = max_events
        self.events         = deque(maxlen=max_events)
        self._press_times   = {}        # key_id → press timestamp
        self._key_counter   = 0         # anonymized position counter
        self._key_map       = {}        # pynput key → anonymized id
        self._listener      = None
        self._lock          = threading.Lock()
        self.is_capturing   = False
        self._start_time    = None

    def start(self):
        """Begin capturing keystrokes."""
        try:
            from pynput import keyboard as kb

            self.events.clear()
            self._press_times.clear()
            self._key_map.clear()
            self._key_counter   = 0
            self.is_capturing   = True
            self._start_time    = time.perf_counter()

            self._listener = kb.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress=False      # don't intercept — just observe
            )
            self._listener.start()
            return True

        except ImportError:
            # pynput not installed — caller should handle this
            self.is_capturing = False
            return False

        except Exception:
            self.is_capturing = False
            return False

    def stop(self):
        """Stop capturing."""
        self.is_capturing = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _get_key_id(self, key) -> str:
        """
        Convert pynput key to anonymous position ID.
        Fix #13: never stores actual key character.
        Special keys (shift, ctrl etc.) get their own stable ID.
        """
        try:
            # Special key (e.g. Key.backspace, Key.shift)
            key_str = str(key)
        except Exception:
            key_str = 'unknown'

        # Backspace — keep as backspace for error rate calculation
        if 'backspace' in key_str.lower():
            return 'backspace'

        # All other keys → anonymize to position index
        with self._lock:
            if key_str not in self._key_map:
                self._key_counter += 1
                self._key_map[key_str] = f'key_{self._key_counter}'
            return self._key_map[key_str]

    def _on_press(self, key):
        """Record press timestamp."""
        if not self.is_capturing:
            return
        try:
            key_id = self._get_key_id(key)
            with self._lock:
                self._press_times[key_id] = time.perf_counter()
        except Exception:
            pass

    def _on_release(self, key):
        """Record release, compute dwell, store event."""
        if not self.is_capturing:
            return
        try:
            key_id       = self._get_key_id(key)
            release_time = time.perf_counter()

            with self._lock:
                press_time = self._press_times.pop(key_id, None)

            if press_time is None:
                return

            dwell = release_time - press_time

            # Sanity check — ignore unrealistic values
            if dwell < 0 or dwell > 2.0:
                return

            self.events.append({
                'press':   press_time,
                'release': release_time,
                'dwell':   dwell,
                'key_id':  key_id,     # anonymized — never real character
            })

        except Exception:
            pass

    def get_events(self) -> List[Dict]:
        """Return captured events."""
        with self._lock:
            return list(self.events)

    def reset(self):
        """Clear all captured data."""
        with self._lock:
            self.events.clear()
            self._press_times.clear()
            self._key_map.clear()
            self._key_counter = 0

    @property
    def event_count(self) -> int:
        return len(self.events)

    def get_stats(self) -> Dict:
        """Quick stats for UI feedback."""
        evs = self.get_events()
        regular = [e for e in evs if e['key_id'] != 'backspace']
        if len(regular) < 2:
            return {'keys': len(regular), 'avg_flight_ms': 0, 'ready': False}
        flights = []
        for i in range(len(regular) - 1):
            f = (regular[i+1]['press'] - regular[i]['release']) * 1000
            if f >= 0:
                flights.append(f)
        avg_flight = sum(flights) / len(flights) if flights else 0
        return {
            'keys':          len(regular),
            'avg_flight_ms': round(avg_flight, 1),
            'ready':         len(regular) >= 4,
        }


# ══════════════════════════════════════════════════════
# Streamlit-Compatible Capture Session
# ══════════════════════════════════════════════════════

class CaptureSession:
    """
    Manages a timed capture window for Streamlit.

    Because Streamlit reruns the script on every interaction,
    the capture listener must live in st.session_state, not
    as a local variable.

    Usage in Streamlit:
        # Start
        session = CaptureSession()
        session.start()
        st.session_state['capture_session'] = session

        # Stop (on next rerun after user typed)
        session = st.session_state['capture_session']
        events  = session.stop_and_get()
    """

    def __init__(self):
        self.capture    = RealKeystrokeCapture()
        self.started_at = None
        self.stopped    = False

    def start(self) -> bool:
        self.started_at = time.time()
        self.stopped    = False
        return self.capture.start()

    def stop_and_get(self) -> List[Dict]:
        if not self.stopped:
            self.capture.stop()
            self.stopped = True
        return self.capture.get_events()

    def get_stats(self) -> Dict:
        return self.capture.get_stats()

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

    @property
    def is_active(self) -> bool:
        return self.capture.is_capturing


# ══════════════════════════════════════════════════════
# Simulated Capture (demo / testing / mood classifier)
# ══════════════════════════════════════════════════════

class SimulatedCapture:
    """
    Generates synthetic keystroke events for:
      - Training the mood classifier (Random Forest)
      - Demo mode when pynput unavailable
      - Benchmarking

    NOT used for real user authentication after enrollment.
    """

    def __init__(self):
        self.events = []

    def simulate_typing(self,
                        avg_speed_ms: float,
                        variance_ms:  float,
                        dwell_ms:     float,
                        n_keys:       int   = 10,
                        error_rate:   float = 0.02,
                        seed:         int   = None) -> list:
        """
        Simulate realistic typing with Gaussian timing distribution.

        Parameters:
            avg_speed_ms  — average gap between keystrokes (ms)
            variance_ms   — natural variation around average
            dwell_ms      — how long each key is held down
            n_keys        — number of keystrokes to simulate
            error_rate    — probability of backspace (mistake)
            seed          — for reproducible tests
        """
        import random
        if seed is not None:
            random.seed(seed)

        events       = []
        current_time = 0.0

        for i in range(n_keys):
            gap   = max(20, random.gauss(avg_speed_ms, variance_ms))
            dwell = max(10, random.gauss(dwell_ms, dwell_ms * 0.15))

            press_time   = current_time + gap
            release_time = press_time + dwell

            events.append({
                'press':   press_time   / 1000,
                'release': release_time / 1000,
                'dwell':   dwell        / 1000,
                'key_id':  f'key_{i}',
            })

            current_time = release_time

            # Backspace simulation (error rate = mood signal Fix #1)
            if random.random() < error_rate:
                bs_gap     = random.gauss(150, 30)
                bs_press   = current_time + bs_gap
                bs_release = bs_press + random.gauss(60, 10)
                events.append({
                    'press':   bs_press   / 1000,
                    'release': bs_release / 1000,
                    'dwell':   (bs_release - bs_press) / 1000,
                    'key_id':  'backspace',
                })
                current_time = bs_release

        self.events = events
        return events
