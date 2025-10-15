import os
from celery import Celery


def _redis_url() -> str:
    url = os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL")
    if url:
        return url
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    password = os.getenv("REDIS_PASSWORD")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


_broker = _redis_url()
_backend = os.getenv("CELERY_RESULT_BACKEND") or _broker

celery_app = Celery(
    "leeway",
    broker=_broker,
    backend=_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

try:
    import tasks  # noqa: F401
except ImportError:
    # In some environments the worker image may not include optional task modules;
    # failing silently keeps the worker boot process resilient.
    pass
