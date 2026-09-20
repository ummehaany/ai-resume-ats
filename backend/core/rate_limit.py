"""Small in-memory rate limiter (no Redis or other infrastructure needed).

Limits are per server *process*: they reset when the server restarts and are not
shared between several workers/instances. Run a single worker (the provided
Dockerfile does) or put a shared limiter (e.g. Redis) in front if you scale out.
"""
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

Rule = Tuple[str, int, int]   # (key, max_hits, window_seconds)


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def try_acquire(self, rules: List[Rule], now: Optional[float] = None) -> Optional[int]:
        """Record one hit against every rule if ALL allow it.

        Returns None when allowed, otherwise the number of seconds to wait before retrying.
        A rule with max_hits <= 0 is disabled.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            retry_after = 0
            for key, max_hits, window in rules:
                if max_hits <= 0:
                    continue
                q = self._hits.get(key)
                if q is None:
                    continue
                while q and now - q[0] >= window:
                    q.popleft()
                if len(q) >= max_hits:
                    retry_after = max(retry_after, int(window - (now - q[0])) + 1)
            if retry_after:
                return retry_after

            for key, max_hits, window in rules:
                if max_hits <= 0:
                    continue
                self._hits.setdefault(key, deque()).append(now)
            self._prune(now)
            return None

    def _prune(self, now: float) -> None:
        if len(self._hits) < 5000:
            return
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] > 86400]:
            del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()
