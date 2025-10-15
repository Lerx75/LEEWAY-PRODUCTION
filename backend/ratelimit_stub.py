# Placeholder rate limiting utilities.
# Future: switch to Redis-backed token bucket.
from time import time
from typing import Dict, Tuple

# In-memory counters: key -> (reset_epoch, count)
_WINDOW = 60
_LIMITS = {
    "auth": 30,   # login/signup per minute per IP
    "generic": 120,
}
_state: Dict[Tuple[str, str], Tuple[float, int]] = {}

def check_limit(kind: str, ident: str) -> bool:
    """Return True if allowed, False if over limit."""
    now = time()
    key = (kind, ident)
    reset, count = _state.get(key, (now + _WINDOW, 0))
    if now > reset:
        reset = now + _WINDOW
        count = 0
    limit = _LIMITS.get(kind, _LIMITS["generic"])
    if count + 1 > limit:
        _state[key] = (reset, count)
        return False
    _state[key] = (reset, count + 1)
    return True
