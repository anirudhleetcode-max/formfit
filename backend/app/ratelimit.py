"""Tiny in-memory sliding-window limiter for failed logins (single process only).

For several workers or servers this would move to Redis; for a single-user training app a
per-process limiter is enough to stop naive password guessing."""
import time
from collections import defaultdict, deque
from threading import Lock


class FailureLimiter:
    def __init__(self, max_failures: int, window_s: int):
        self.max, self.window = max_failures, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def _trim(self, q: deque, now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def blocked(self, key: str) -> int:
        """Seconds until the key may try again (0 = allowed)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.get(key)
            if not q:
                return 0
            self._trim(q, now)
            if len(q) < self.max:
                return 0
            return int(self.window - (now - q[0])) + 1

    def fail(self, key: str) -> None:
        with self._lock:
            q = self._hits[key]
            self._trim(q, time.monotonic())
            q.append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)
