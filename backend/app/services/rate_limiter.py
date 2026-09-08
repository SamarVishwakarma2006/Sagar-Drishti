import time
from collections import defaultdict, deque
from typing import Tuple, Dict
import threading


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter for SagarBot requests.
    Tracks requests per client identifier (IP address) and global requests.
    """

    def __init__(self, max_requests: int = 15, window_seconds: float = 60.0, global_max: int = 40):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.global_max = global_max
        self._client_history: Dict[str, deque] = defaultdict(deque)
        self._global_history: deque = deque()
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Checks whether the request from client_id is allowed under the rate limit.
        Returns:
            (allowed: bool, retry_after_seconds: int, remaining_requests: int)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Clean global history
            while self._global_history and self._global_history[0] <= window_start:
                self._global_history.popleft()

            # Check global rate limit
            if len(self._global_history) >= self.global_max:
                oldest = self._global_history[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest)))
                return False, retry_after, 0

            # Clean client-specific history
            client_queue = self._client_history[client_id]
            while client_queue and client_queue[0] <= window_start:
                client_queue.popleft()

            # Check client rate limit
            if len(client_queue) >= self.max_requests:
                oldest = client_queue[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest)))
                return False, retry_after, 0

            # Record this request
            client_queue.append(now)
            self._global_history.append(now)

            remaining = max(0, self.max_requests - len(client_queue))
            return True, 0, remaining

    def reset(self):
        """Resets all tracking history (useful for testing)."""
        with self._lock:
            self._client_history.clear()
            self._global_history.clear()


# Default global instance for SagarBot API
chat_rate_limiter = RateLimiter(max_requests=15, window_seconds=60.0, global_max=40)
