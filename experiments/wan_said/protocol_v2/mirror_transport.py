"""Mirror-only requests with process-wide pacing and shared 429 cooldowns."""
import json
import threading
import time
from email.utils import parsedate_to_datetime

import requests

from ..mirror_download import MIRROR, StrictRangeSession


def retry_after_seconds(value, now=None):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            timestamp = parsedate_to_datetime(value).timestamp()
            return max(0.0, timestamp - (time.time() if now is None else now))
        except (TypeError, ValueError, OverflowError):
            return None


class RequestGate:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.blocked_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(self.next_request, self.blocked_until) - now
                if delay <= 0:
                    self.next_request = now + self.interval
                    return
            time.sleep(delay)

    def cooldown(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


MIRROR_GATE = RequestGate()


class MirrorRangeSession(StrictRangeSession):
    def __init__(self, gate=None, rate_retries=5):
        super().__init__()
        self.trust_env = False
        self.gate = MIRROR_GATE if gate is None else gate
        self.rate_retries = rate_retries

    def request(self, method, url, **kwargs):
        if not url.startswith(MIRROR + '/'):
            raise ValueError('Only the configured domestic HF mirror is permitted')
        for attempt in range(self.rate_retries + 1):
            self.gate.wait()
            try:
                return super().request(method, url, **kwargs)
            except requests.HTTPError as error:
                response = error.response
                if response is None:
                    raise
                status = response.status_code
                delay = retry_after_seconds(response.headers.get('Retry-After'))
                response.close()
                if status != 429:
                    raise
                # Cool down all workers, not just the one that received the 429.
                delay = max(60.0, delay if delay is not None else min(900, 60 * 2**attempt))
                self.gate.cooldown(delay)
                print(json.dumps(dict(event='mirror_rate_limited', source=MIRROR,
                                      cooldown_seconds=delay, retry=attempt + 1)), flush=True)
                if attempt == self.rate_retries:
                    raise
