"""A thermostat for long GPU jobs: pause while the GPU is too hot.

Reads the temperature with nvidia-smi (no extra package, no settings changed).
``cool_down(limit)`` returns at once below ``limit`` °C; otherwise it waits,
with the GPU idle, until the temperature is ``hysteresis`` degrees lower. The
long jobs call it between batches or training steps. Without nvidia-smi it
does nothing.
"""

import shutil
import subprocess
import time

_SMI = shutil.which("nvidia-smi") or (shutil.which("/usr/lib/wsl/lib/nvidia-smi"))


def temperature():
    """Hottest GPU temperature in °C, or None when it cannot be read."""
    if not _SMI:
        return None
    try:
        out = subprocess.run([_SMI, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return max(int(value) for value in out.split())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class Thermostat:
    """Checks at most every ``every`` seconds, so it costs nothing per step."""

    def __init__(self, limit, hysteresis=6, every=20.0, log=print):
        self.limit, self.hysteresis, self.every, self.log = limit, hysteresis, every, log
        self.checked, self.paused_seconds = 0.0, 0.0

    def __call__(self):
        if not self.limit or time.monotonic() - self.checked < self.every:
            return 0.0
        self.checked = time.monotonic()
        current = temperature()
        if current is None or current < self.limit:
            return 0.0
        start = time.monotonic()
        self.log(f"GPU at {current} °C (limit {self.limit}); pausing until {self.limit - self.hysteresis} °C")
        while current is not None and current > self.limit - self.hysteresis:
            time.sleep(5)
            current = temperature()
        paused = time.monotonic() - start
        self.paused_seconds += paused
        self.log(f"GPU at {current} °C after {paused:.0f} s; resuming")
        self.checked = time.monotonic()
        return paused
