import asyncio
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from celery_app import celery_app


@lru_cache(maxsize=1)
def _get_runner():
    module = import_module("app")
    try:
        return module._vehicle_route_run_from_bytes
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Vehicle route runner not available in backend app") from exc


@celery_app.task(bind=True, name="vehicle_route_task")
def vehicle_route_task(self, job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the vehicle routing job in the background worker."""
    calls_path = job.get("calls_path")
    resources_path = job.get("resources_path")
    if not calls_path or not resources_path:
        raise ValueError("Job payload missing calls/resources paths")

    params = job.get("params", {})
    headers = job.get("headers") or {}

    with open(calls_path, "rb") as cf:
        calls_bytes = cf.read()
    with open(resources_path, "rb") as rf:
        resources_bytes = rf.read()

    result = asyncio.run(
        _get_runner()(
            calls_bytes=calls_bytes,
            resources_bytes=resources_bytes,
            params=params,
            headers=headers,
        )
    )

    if job.get("cleanup", True):
        work_dir = job.get("work_dir")
        for path in (calls_path, resources_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        if work_dir:
            try:
                Path(work_dir).rmdir()
            except Exception:
                pass

    return result
