"""
KeyDNA — Simulated Keystroke Capture

Generates synthetic keystroke events for:
  - Training and benchmarking
  - Demo mode when pynput unavailable
  - Unit testing

NOT used for real user authentication.
Real capture is handled by KeystrokeCapture in keydna_app.py.

PRIVACY:
  Only TIMING is stored — press time, release time, dwell.
  Key content is anonymized to position index.
"""

from typing import Dict, List


class SimulatedCapture:
    """
    Generates synthetic keystroke events for testing and benchmarking.

    NOT used for real user authentication after enrollment.
    Real keyboard capture is handled by KeystrokeCapture in keydna_app.py.
    """

    def __init__(self) -> None:
        self.events: List[Dict] = []

    def simulate_typing(
        self,
        avg_speed_ms: float,
        variance_ms: float,
        dwell_ms: float,
        n_keys: int = 10,
        error_rate: float = 0.02,
        seed: int = None,
    ) -> List[Dict]:
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

        events: List[Dict] = []
        current_time: float = 0.0

        for i in range(n_keys):
            gap: float = max(20, random.gauss(avg_speed_ms, variance_ms))
            dwell: float = max(10, random.gauss(dwell_ms, dwell_ms * 0.15))

            press_time: float = current_time + gap
            release_time: float = press_time + dwell

            events.append({
                'press': press_time / 1000,
                'release': release_time / 1000,
                'dwell': dwell / 1000,
                'key_id': f'key_{i}',
            })

            current_time = release_time

            # Backspace simulation (error rate)
            if random.random() < error_rate:
                bs_gap: float = random.gauss(150, 30)
                bs_press: float = current_time + bs_gap
                bs_release: float = bs_press + random.gauss(60, 10)
                events.append({
                    'press': bs_press / 1000,
                    'release': bs_release / 1000,
                    'dwell': (bs_release - bs_press) / 1000,
                    'key_id': 'backspace',
                })
                current_time = bs_release

        self.events = events
        return events
