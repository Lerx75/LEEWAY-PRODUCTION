import os
import math
import random
import hashlib
import hmac
import logging
from urllib.parse import urlparse
import io
import time
import json
import base64
import traceback
import asyncio
import tempfile
from datetime import datetime
from functools import lru_cache
from typing import Optional, List, Tuple, Dict, Any, Mapping, Union

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pandas as pd
import requests
import googlemaps
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from health import router as health_router
from project_store import build_store_from_env
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import duckdb as _duckdb  # type: ignore
    _DUCKDB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _duckdb = None  # type: ignore
    _DUCKDB_AVAILABLE = False

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore
    _HAS_ORTOOLS = True
except Exception:  # pragma: no cover - allow start without OR-Tools for tooling
    pywrapcp = None  # type: ignore
    routing_enums_pb2 = None  # type: ignore
    _HAS_ORTOOLS = False

try:
    from ortools.graph import pywrapgraph  # type: ignore
    _mcf_mod = pywrapgraph
    _HAS_FLOW = True
except Exception:  # pragma: no cover - optional dependency
    pywrapgraph = None  # type: ignore
    _mcf_mod = None
    _HAS_FLOW = False

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_QPS = float(os.getenv("GOOGLE_QPS", "5"))
GOOGLE_TIMEOUT = float(os.getenv("GOOGLE_TIMEOUT", "10"))
GOOGLE_RETRY_TIMEOUT = float(os.getenv("GOOGLE_RETRY_TIMEOUT", "60"))

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY must be configured")

try:
    gmaps = googlemaps.Client(
        key=GOOGLE_API_KEY,
        queries_per_second=max(1.0, GOOGLE_QPS),
        timeout=GOOGLE_TIMEOUT,
        retry_timeout=GOOGLE_RETRY_TIMEOUT,
    )
except Exception as exc:
    raise RuntimeError(f"Failed to initialise Google Maps client: {exc}") from exc

_cors_origins_env = os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ALLOW_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
if not _cors_origins:
    _cors_origins = ["*"]
if "*" in _cors_origins:
    _cors_origins = ["*"]

app = FastAPI(title="LeeWay API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
app.include_router(health_router)

logger = logging.getLogger("leeway.api")

# Try new OR-Tools API first, then legacy
_UPLOAD_FILE_TYPES = (UploadFile, StarletteUploadFile)

_PROJECT_SERVICE_TOKEN = os.getenv("PROJECT_SERVICE_TOKEN", "").strip()
_PROJECT_STORE = None
_PROJECT_STORE_ERROR_LOGGED = False


def _new_mcf():
    if not _HAS_FLOW or _mcf_mod is None:
        raise RuntimeError("Min-cost flow not available on this server")
    if hasattr(_mcf_mod, "SimpleMinCostFlow"):
        return _mcf_mod.SimpleMinCostFlow()
    if hasattr(_mcf_mod, "simple_min_cost_flow"):
        return _mcf_mod.simple_min_cost_flow()
    raise RuntimeError("Min-cost flow API unavailable")


def _get_project_store():
    global _PROJECT_STORE, _PROJECT_STORE_ERROR_LOGGED
    if _PROJECT_STORE is None:
        try:
            _PROJECT_STORE = build_store_from_env()
            _PROJECT_STORE_ERROR_LOGGED = False
        except Exception as exc:  # pragma: no cover - configuration issue
            if not _PROJECT_STORE_ERROR_LOGGED:
                print(f"[WARN] Project store unavailable: {exc}")
                _PROJECT_STORE_ERROR_LOGGED = True
            _PROJECT_STORE = None
    return _PROJECT_STORE


def _get_header_value(source: Union[Request, Mapping[str, Any], None], key: str) -> Optional[str]:
    if source is None:
        return None
    if isinstance(source, Request):
        return source.headers.get(key)
    if isinstance(source, Mapping):
        for variant in (key, key.lower(), key.upper()):
            if variant in source:
                value = source[variant]
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                if value is None:
                    continue
                return str(value)
    return None


def _service_token_valid(token: Optional[str]) -> bool:
    if not token or not _PROJECT_SERVICE_TOKEN:
        return False
    try:
        return hmac.compare_digest(str(token).strip(), _PROJECT_SERVICE_TOKEN)
    except Exception:
        return str(token).strip() == _PROJECT_SERVICE_TOKEN


def _require_project_access(request: Request):
    store = _get_project_store()
    if store is None:
        raise HTTPException(503, "Project storage unavailable")
    if not _PROJECT_SERVICE_TOKEN:
        raise HTTPException(503, "PROJECT_SERVICE_TOKEN not configured")
    token = _get_header_value(request, "X-Service-Token")
    if not _service_token_valid(token):
        raise HTTPException(403, "Invalid service token")
    user_id = (
        _get_header_value(request, "X-User-Id")
        or _get_header_value(request, "X-User-ID")
        or _get_header_value(request, "X-Project-User")
    )
    if not user_id or not str(user_id).strip():
        raise HTTPException(400, "X-User-Id header required")
    return store, str(user_id).strip()


def _maybe_save_project(
    headers_source: Union[Request, Mapping[str, Any], None],
    *,
    mode: str,
    rows: List[List[Any]],
    meta: Optional[Dict[str, Any]] = None,
    explicit_name: Optional[str] = None,
) -> Optional[int]:
    store = _get_project_store()
    if store is None or not _PROJECT_SERVICE_TOKEN:
        return None
    token = _get_header_value(headers_source, "X-Service-Token")
    if not _service_token_valid(token):
        return None
    user_id = (
        _get_header_value(headers_source, "X-User-Id")
        or _get_header_value(headers_source, "X-User-ID")
        or _get_header_value(headers_source, "X-Project-User")
    )
    if not user_id:
        return None
    user_id = str(user_id).strip()
    if not user_id:
        return None
    project_name = (explicit_name or _get_header_value(headers_source, "X-Project-Name") or "").strip()
    if not project_name:
        project_name = f"{mode}-{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    project_name = project_name[:200]
    try:
        return store.save_project(
            user_id=user_id,
            name=project_name,
            mode=mode,
            rows=rows,
            meta=dict(meta or {}),
        )
    except Exception as exc:  # pragma: no cover - storage failure fallback
        print(f"[WARN] Failed to save project for user {user_id}: {exc}")
        return None


@app.get("/projects")
async def list_projects(request: Request):
    store, user_id = _require_project_access(request)
    projects = store.list_projects(user_id=user_id)
    return {"projects": projects}


@app.get("/projects/{project_id}")
async def get_project_detail(project_id: int, request: Request):
    if project_id <= 0:
        raise HTTPException(400, "Invalid project id")
    store, user_id = _require_project_access(request)
    project = store.get_project(project_id=project_id, user_id=user_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, request: Request):
    if project_id <= 0:
        raise HTTPException(400, "Invalid project id")
    store, user_id = _require_project_access(request)
    deleted = store.delete_project(project_id=project_id, user_id=user_id)
    if not deleted:
        raise HTTPException(404, "Project not found")
    return {"deleted": True}


@app.post("/api/vehicle-route")
async def vehicle_route(request: Request):
    form = await request.form()
    try:
        print("[DEBUG] /api/vehicle-route form keys:", list(form.keys()))
    except Exception:
        pass
    calls_file = form.get("callsFile")
    resources_file = form.get("resourcesFile")
    if not isinstance(calls_file, _UPLOAD_FILE_TYPES):
        raise HTTPException(400, "callsFile upload is required")
    if not isinstance(resources_file, _UPLOAD_FILE_TYPES):
        raise HTTPException(400, "resourcesFile upload is required")

    params: Dict[str, Any] = {}
    for key, value in form.items():
        if isinstance(value, _UPLOAD_FILE_TYPES):
            continue
        params[key] = value if isinstance(value, str) else ("" if value is None else str(value))

    job_root = os.getenv("VRP_JOB_DIR", "/srv/data/vrp-jobs")
    try:
        os.makedirs(job_root, exist_ok=True)
    except Exception:
        pass
    job_dir = tempfile.mkdtemp(prefix="vrp_job_", dir=job_root)
    calls_path = os.path.join(job_dir, "calls.xlsx")
    resources_path = os.path.join(job_dir, "resources.xlsx")

    try:
        with open(calls_path, "wb") as cf:
            cf.write(await calls_file.read())
        with open(resources_path, "wb") as rf:
            rf.write(await resources_file.read())
    finally:
        await calls_file.close()
        await resources_file.close()

    job_payload = {
        "calls_path": calls_path,
        "resources_path": resources_path,
        "params": params,
        "headers": dict(request.headers),
        "cleanup": True,
        "work_dir": job_dir,
    }

    try:
        from celery_app import celery_app  # type: ignore
        async_result = celery_app.send_task("vehicle_route_task", args=[job_payload])
    except Exception as exc:  # pragma: no cover - defensive cleanup
        try:
            os.remove(calls_path)
        except Exception:
            pass
        try:
            os.remove(resources_path)
        except Exception:
            pass
        try:
            os.rmdir(job_dir)
        except Exception:
            pass
        raise HTTPException(503, f"Failed to queue routing job: {exc}") from exc

    status_url = request.url_for("vehicle_route_status", task_id=async_result.id)
    result_url = request.url_for("vehicle_route_result", task_id=async_result.id)

    return JSONResponse(
        {
            "job_id": async_result.id,
            "status": "queued",
            "status_url": str(status_url),
            "result_url": str(result_url),
        },
        status_code=202,
    )


@app.get("/api/vehicle-route/status/{task_id}", name="vehicle_route_status")
def vehicle_route_status(task_id: str):
    from celery_app import celery_app  # type: ignore

    result = celery_app.AsyncResult(task_id)
    payload: Dict[str, Any] = {
        "job_id": task_id,
        "state": result.state,
        "ready": result.ready(),
    }
    if result.state == "FAILURE":
        payload["error"] = str(result.info)
    return JSONResponse(payload)


@app.get("/api/vehicle-route/result/{task_id}", name="vehicle_route_result")
def vehicle_route_result(task_id: str):
    from celery_app import celery_app  # type: ignore

    result = celery_app.AsyncResult(task_id)
    if not result.ready():
        return JSONResponse({
            "job_id": task_id,
            "state": result.state,
        }, status_code=202)
    if result.state != "SUCCESS":
        raise HTTPException(500, f"Routing job failed: {result.info}")

    data = result.result
    try:
        result.forget()
    except Exception:
        pass
    return JSONResponse({
        "job_id": task_id,
        "state": result.state,
        "result": data,
    })

# Optional H3 contiguity penalty config
try:
    import h3
    _H3_AVAILABLE = True
except Exception:
    h3 = None  # type: ignore
    _H3_AVAILABLE = False


# --- Tweaked parameters for tighter clusters and less overlap ---
H3_RES = int(os.getenv("H3_RES", "8"))  # smaller hex size for more compact clusters
# Increase default contiguity penalty to reduce scattered day clusters
H3_LAMBDA = float(os.getenv("H3_LAMBDA", "60000"))  # stronger penalty for splitting hexes
FLOW_ALPHA = float(os.getenv("FLOW_ALPHA", "1.6"))
FLOW_GAMMA = float(os.getenv("FLOW_GAMMA", "6.0"))  # even stronger weight for distance
USE_CELL_FLOW = int(os.getenv("USE_CELL_FLOW", "1"))
_H3_PLAN_ENV = os.getenv("H3_PLAN_RES")
if _H3_PLAN_ENV is not None:
    H3_PLAN_RES = int(_H3_PLAN_ENV)
else:
    H3_PLAN_RES = max(H3_RES - 1, 5)
FLOW_TOPK = int(os.getenv("FLOW_TOPK", "1"))
FLOW_STRICT = int(os.getenv("FLOW_STRICT", "1"))
CORRIDOR_ON = int(os.getenv("CORRIDOR_ON", "1"))  # enable corridor shaping
CORRIDOR_K = int(os.getenv("CORRIDOR_K", "3"))
CORRIDOR_SIGMA_DEG = float(os.getenv("CORRIDOR_SIGMA_DEG", "15"))
CORRIDOR_BETA = float(os.getenv("CORRIDOR_BETA", "0.25"))
SMOOTH_ITERS = int(os.getenv("SMOOTH_ITERS", "40"))  # double smoothing passes
VRP_TIME_BUDGET_ENABLED = int(os.getenv("VRP_TIME_BUDGET_ENABLED", "0"))  # 1: enable time-budget route mode
VRP_PRECLUSTER = int(os.getenv("VRP_PRECLUSTER", "1"))  # 1: precluster per-day to compact days
VRP_CLUSTER_MIN_RATIO = float(os.getenv("VRP_CLUSTER_MIN_RATIO", "0.8"))  # min_calls = ratio * max_calls
VRP_PRECLUSTER_METHOD = (os.getenv("VRP_PRECLUSTER_METHOD", "kmedoids").strip().lower() or "kmedoids")
VRP_PRECLUSTER_OSRM_MAX_N = int(os.getenv("VRP_PRECLUSTER_OSRM_MAX_N", "80"))  # above this, avoid OSRM in k-medoids
VRP_KMEDOIDS_OSRM_CENTERS = int(os.getenv("VRP_KMEDOIDS_OSRM_CENTERS", "1"))  # if 1, use OSRM only for coords->centers matrix
VRP_ROUTE_TIME_LIMIT_SEC = int(os.getenv("VRP_ROUTE_TIME_LIMIT_SEC", "10"))  # per-route solver time limit
VRP_META = (os.getenv("VRP_META", "gls").strip().lower() or "gls")  # gls|sa|greedy
VRP_FIRST = (os.getenv("VRP_FIRST", "parallel_cheapest_insertion").strip().lower() or "parallel_cheapest_insertion")
VRP_MS_TRIES = max(1, int(os.getenv("VRP_MS_TRIES", "2")))  # multi-start attempts per route/territory
VRP_MS_TOTAL_SEC = max(0, int(os.getenv("VRP_MS_TOTAL_SEC", "0")))  # 0 = disabled; otherwise cap total search seconds across tries
VRP_NUM_WORKERS = max(1, int(os.getenv("VRP_NUM_WORKERS", str(os.cpu_count() or 1))))
VRP_DROP_PENALTY = max(1, int(os.getenv("VRP_DROP_PENALTY", "10000000")))  # discourage dropping
VRP_LNS_TIME_SEC = max(0, int(os.getenv("VRP_LNS_TIME_SEC", "1")))  # 0 to disable LNS; small positive for mild improvement

CLUSTER_OSRM_MAX_N = int(os.getenv("CLUSTER_OSRM_MAX_N", "140"))
CLUSTER_PRECLUSTER_FIRST = int(os.getenv("CLUSTER_PRECLUSTER_FIRST", "1"))
CLUSTER_DEFAULT_SERVICE_MIN = float(os.getenv("CLUSTER_DEFAULT_SERVICE_MIN", "45"))
CLUSTER_SOLVER_TIME_SEC = int(os.getenv("CLUSTER_SOLVER_TIME_SEC", "20"))
CLUSTER_BASE_DAY_MINUTES = float(os.getenv("CLUSTER_BASE_DAY_MINUTES", "240"))
CLUSTER_DURATION_PENALTY = float(os.getenv("CLUSTER_DURATION_PENALTY", "12.0"))
CLUSTER_SA_POLISH = int(os.getenv("CLUSTER_SA_POLISH", "1"))
CLUSTER_SA_MAX_N = int(os.getenv("CLUSTER_SA_MAX_N", "180"))
# More polishing by default to encourage cross-day swaps when beneficial
CLUSTER_SA_ITERS = int(os.getenv("CLUSTER_SA_ITERS", "600"))
CLUSTER_SA_INIT_TEMP = float(os.getenv("CLUSTER_SA_INIT_TEMP", "80"))
CLUSTER_SA_COOLING = float(os.getenv("CLUSTER_SA_COOLING", "0.996"))
CLUSTER_REFINE_ITERS = int(os.getenv("CLUSTER_REFINE_ITERS", "2"))
CLUSTER_REFINE_IMPROVE_RATIO = float(os.getenv("CLUSTER_REFINE_IMPROVE_RATIO", "1.25"))

_OSRM_BASE_RAW = (
    os.getenv("OSRM_BASE")
    or os.getenv("OSRM_BASE_URL")
    or os.getenv("OSRM_URL")
    or "https://router.project-osrm.org"
)
OSRM_BASE = _OSRM_BASE_RAW.rstrip("/")
if not OSRM_BASE.startswith("http://") and not OSRM_BASE.startswith("https://"):
    OSRM_BASE = f"http://{OSRM_BASE.lstrip('/')}"


def _get_osrm_url() -> str:
    """Return the configured OSRM URL, falling back to OSRM_BASE if missing."""
    return globals().get("OSRM_URL", OSRM_BASE)


OSRM_URL = _get_osrm_url()

_OSRM_SESSION_LOCAL = threading.local()
# OSRM diagnostics
OSRM_STRICT = int(os.getenv("OSRM_STRICT", "0"))
_OSRM_REQ_COUNT = 0
_OSRM_ERR_COUNT = 0
_OSRM_FALLBACK_COUNT = 0
_OSRM_LAST_ERROR: Optional[str] = None
OSRM_REQUEST_ATTEMPTS = max(1, int(os.getenv("OSRM_REQUEST_ATTEMPTS", "5")))
OSRM_REQUEST_TIMEOUT = float(os.getenv("OSRM_REQUEST_TIMEOUT", "20"))
OSRM_REQUEST_BACKOFF_BASE = float(os.getenv("OSRM_REQUEST_BACKOFF_BASE", "0.75"))
OSRM_REQUEST_MAX_SLEEP = float(os.getenv("OSRM_REQUEST_MAX_SLEEP", "5.0"))
OSRM_RETRY_TOTAL = max(0, int(os.getenv("OSRM_RETRY_TOTAL", "3")))
OSRM_RETRY_BACKOFF = float(os.getenv("OSRM_RETRY_BACKOFF", "0.5"))
OSRM_DISTANCE_BATCH_SIZE = max(1, int(os.getenv("OSRM_DISTANCE_BATCH_SIZE", "20")))
OSRM_DURATION_BATCH_SIZE = max(1, int(os.getenv("OSRM_DURATION_BATCH_SIZE", "20")))
_OSRM_TRANSIENT_STATUS = {429, 500, 502, 503, 504, 520, 522, 524}


def _get_osrm_session() -> requests.Session:
    session = getattr(_OSRM_SESSION_LOCAL, "session", None)
    if session is None:
        retry = Retry(
            total=OSRM_RETRY_TOTAL,
            connect=OSRM_RETRY_TOTAL,
            read=OSRM_RETRY_TOTAL,
            backoff_factor=OSRM_RETRY_BACKOFF,
            status_forcelist=tuple(_OSRM_TRANSIENT_STATUS),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=4,
            pool_maxsize=8,
        )
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _OSRM_SESSION_LOCAL.session = session
    return session


def _osrm_backoff_delay(attempt: int) -> float:
    base = OSRM_REQUEST_BACKOFF_BASE * (2 ** attempt)
    jitter = random.uniform(0.0, OSRM_REQUEST_BACKOFF_BASE)
    return min(base + jitter, OSRM_REQUEST_MAX_SLEEP)


def _fetch_osrm_matrix(url: str, *, expected_key: str) -> List[List[float]]:
    global _OSRM_REQ_COUNT, _OSRM_ERR_COUNT, _OSRM_LAST_ERROR
    session = _get_osrm_session()
    last_exc: Optional[Exception] = None
    for attempt in range(OSRM_REQUEST_ATTEMPTS):
        response = None
        try:
            response = session.get(url, timeout=OSRM_REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if expected_key in data:
                    try:
                        _OSRM_REQ_COUNT += 1
                    except Exception:
                        pass
                    return data[expected_key]
                last_exc = RuntimeError(f"OSRM response missing key '{expected_key}'")
            else:
                snippet = "".join(response.text.splitlines())[:200]
                if response.status_code in _OSRM_TRANSIENT_STATUS:
                    last_exc = RuntimeError(
                        f"OSRM {expected_key} request failed with status {response.status_code}: {snippet}"
                    )
                else:
                    response.raise_for_status()
        except Exception as exc:  # noqa: PERF203 - explicit retry loop ok here
            last_exc = exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
        if attempt + 1 < OSRM_REQUEST_ATTEMPTS:
            time.sleep(_osrm_backoff_delay(attempt))
    if last_exc:
        try:
            _OSRM_ERR_COUNT += 1
            _OSRM_LAST_ERROR = str(last_exc)
        except Exception:
            pass
        raise RuntimeError(
            f"OSRM {expected_key} table failed after {OSRM_REQUEST_ATTEMPTS} attempts. Last error: {last_exc}"
        )
    raise RuntimeError(f"OSRM {expected_key} table failed without providing an error")

def _hex_ids_for_coords(coords: np.ndarray, res: int = H3_RES) -> list:
    """Return a list of hex IDs (or grid IDs if H3 unavailable) for coords [[lat,lng],...]."""
    ids: list = []
    if _H3_AVAILABLE:
        for lat, lng in coords:
            try:
                ids.append(h3.geo_to_h3(lat, lng, res))
            except Exception:
                ids.append(None)
        return ids
    # Fallback: simple grid bucketing (~approx to keep logic working without H3)
    # Use degree steps tuned to roughly match H3 cell sizes by res.
    r = int(res)
    r = 5 if r < 5 else (10 if r > 10 else r)
    step_map = {  # ~deg at mid-latitudes
        5: 0.02,   # ~2.2 km
        6: 0.010,  # ~1.1 km
        7: 0.005,  # ~550 m
        8: 0.0025, # ~275 m
        9: 0.0012, # ~130 m
        10: 0.0006 # ~65 m
    }
    step = step_map.get(r, 0.0025)
    for lat, lng in coords:
        try:
            gy = int(np.floor(lat / step))
            gx = int(np.floor(lng / step))
            ids.append(f"g{res}:{gy}:{gx}")
        except Exception:
            ids.append(None)
    return ids

def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees from (lat1,lon1) to (lat2,lon2) in [0,360)."""
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    brng = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return float(brng)

def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings (degrees) in [0,180]."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(d)

def _corridor_headings_for_centers(centers: np.ndarray, items: np.ndarray, k: int) -> List[List[float]]:
    """Estimate corridor headings per center using PCA on deltas to all items.
    Returns a list length = n_centers, each is a list of up to k headings (degrees).
    """
    if k <= 0:
        return [[] for _ in range(len(centers))]
    headings_per_center: List[List[float]] = []
    # Approx local meters for lat/lon
    mean_lat = float(np.mean(items[:, 0])) if len(items) else 0.0
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(mean_lat))
    for c in centers:
        cy, cx = float(c[0]), float(c[1])
        dy = (items[:, 0] - cy) * m_per_deg_lat
        dx = (items[:, 1] - cx) * m_per_deg_lon
        V = np.vstack([dx, dy]).T  # N x 2
        if len(V) < 2:
            headings_per_center.append([0.0])
            continue
        # covariance and eigenvectors
        try:
            C = np.cov(V.T)
            w, ev = np.linalg.eigh(C)
            idx = np.argsort(w)[::-1]
            ev = ev[:, idx]
            h1 = (np.degrees(np.arctan2(ev[1, 0], ev[0, 0])) + 360.0) % 360.0
            heads = [float(h1)]
            if k >= 2:
                h2 = (h1 + 90.0) % 360.0
                heads.append(float(h2))
            # pad if k>2 evenly around
            while len(heads) < k:
                heads.append((heads[0] + 180.0) % 360.0)
            headings_per_center.append(heads[:k])
        except Exception:
            # fallback evenly spaced
            heads = [t * (360.0 / max(1, k)) for t in range(max(1, k))]
            headings_per_center.append(heads)
    return headings_per_center


# ----------------------------
# OSRM helpers (retry + cache)
# ----------------------------
def _build_osrm_table_url(coords_concat: str, params: str) -> str:
    return f"{OSRM_BASE}/table/v1/driving/{coords_concat}?{params}"

@lru_cache(maxsize=200_000)
def _cached_table(url: str) -> List[List[float]]:
    # Cached by full URL (including params ordering)
    return _fetch_osrm_matrix(url, expected_key="distances")

def _coords_to_string(coords: np.ndarray) -> List[str]:
    # coords: [[lat,lng], ...] -> ["lng,lat", ...] (OSRM wants lon,lat)
    return [f"{lng},{lat}" for lat, lng in coords]

# Duration table caching (separate cache to avoid mixing with distances)
def _build_osrm_table_url_duration(coords_concat: str, params: str) -> str:
    return f"{OSRM_BASE}/table/v1/driving/{coords_concat}?{params}"

@lru_cache(maxsize=200_000)
def _cached_table_duration(url: str) -> List[List[float]]:
    return _fetch_osrm_matrix(url, expected_key="durations")


OSRM_FALLBACK_SPEED_MPS = float(os.getenv("OSRM_FALLBACK_SPEED_MPS", "13.8889"))  # ~50 km/h


def _duration_block(loc_from: np.ndarray, loc_to: np.ndarray, *, depth: int = 0) -> np.ndarray:
    global _OSRM_FALLBACK_COUNT
    """Fetch an OSRM duration block, recursively splitting on failure."""
    if len(loc_from) == 0 or len(loc_to) == 0:
        return np.zeros((len(loc_from), len(loc_to)), dtype=float)

    all_stack = np.vstack([loc_from, loc_to])
    all_str = _coords_to_string(all_stack)
    srcs = list(range(len(loc_from)))
    dsts = list(range(len(loc_from), len(loc_from) + len(loc_to)))
    url = _build_osrm_table_url_duration(
        ";".join(all_str),
        f"sources={';'.join(map(str,srcs))}&destinations={';'.join(map(str,dsts))}&annotations=duration",
    )

    try:
        chunk = np.array(_cached_table_duration(url), dtype=float)
        if chunk.shape != (len(loc_from), len(loc_to)):
            raise RuntimeError("OSRM duration response shape mismatch")
        return chunk
    except Exception as exc:
        if depth == 0:
            logger.warning(
                "OSRM duration block failed (from=%d, to=%d): %s -- attempting split",
                len(loc_from),
                len(loc_to),
                exc,
            )
        if len(loc_from) == 1 and len(loc_to) == 1:
            # Fall back to haversine distance converted to duration with assumed speed.
            dist_m = _haversine_matrix_between(loc_from, loc_to)[0, 0]
            speed = max(OSRM_FALLBACK_SPEED_MPS, 0.1)
            approx = dist_m / speed
            logger.warning(
                "Falling back to haversine duration for single pair (dist=%.2fm)",
                dist_m,
            )
            try:
                _OSRM_FALLBACK_COUNT += 1
            except Exception:
                pass
            if OSRM_STRICT:
                raise RuntimeError("OSRM_STRICT=1: duration fallback disabled")
            return np.array([[approx]], dtype=float)

        # Recursively split the larger dimension to reduce payload size.
        if len(loc_from) >= len(loc_to) and len(loc_from) > 1:
            mid = len(loc_from) // 2
            upper = _duration_block(loc_from[:mid], loc_to, depth=depth + 1)
            lower = _duration_block(loc_from[mid:], loc_to, depth=depth + 1)
            return np.vstack([upper, lower])
        if len(loc_to) > 1:
            mid = len(loc_to) // 2
            left = _duration_block(loc_from, loc_to[:mid], depth=depth + 1)
            right = _duration_block(loc_from, loc_to[mid:], depth=depth + 1)
            return np.hstack([left, right])

        # As a last resort, approximate with haversine-based durations.
        dist_block = _haversine_matrix_between(loc_from, loc_to)
        speed = max(OSRM_FALLBACK_SPEED_MPS, 0.1)
        approx = dist_block / speed
        logger.error(
            "OSRM duration block fallback to haversine for from=%d,to=%d after failures: %s",
            len(loc_from),
            len(loc_to),
            exc,
        )
        try:
            _OSRM_FALLBACK_COUNT += int(len(loc_from) * len(loc_to))
        except Exception:
            pass
        if OSRM_STRICT:
            raise RuntimeError("OSRM_STRICT=1: duration fallback disabled")
        return approx


def osrm_table_batch_duration(coords: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    """
    Square matrix of travel times (seconds) between all coords.
    """
    n = len(coords)
    M = np.zeros((n, n), dtype=float)
    step = max(1, int(batch_size if batch_size is not None else OSRM_DURATION_BATCH_SIZE))
    for i in range(0, n, step):
        from_idx = list(range(i, min(i + step, n)))
        loc_from = coords[from_idx]
        for j in range(0, n, step):
            to_idx = list(range(j, min(j + step, n)))
            loc_to = coords[to_idx]
            chunk = _duration_block(loc_from, loc_to)
            M[np.ix_(from_idx, to_idx)] = chunk
    return M

def osrm_matrix_between_duration(A: np.ndarray, B: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    """
    Rectangular matrix of travel times (seconds) from each point in A to each point in B.
    """
    nA, nB = len(A), len(B)
    M = np.zeros((nA, nB), dtype=float)
    step = max(1, int(batch_size if batch_size is not None else OSRM_DURATION_BATCH_SIZE))
    for i in range(0, nA, step):
        from_idx = list(range(i, min(i + step, nA)))
        loc_from = A[from_idx]
        for j in range(0, nB, step):
            to_idx = list(range(j, min(j + step, nB)))
            loc_to = B[to_idx]
            chunk = _duration_block(loc_from, loc_to)
            M[np.ix_(from_idx, to_idx)] = chunk
    return M

def _haversine_matrix_full(coords: np.ndarray) -> np.ndarray:
    """Fallback: great-circle distance matrix (meters) for coords Nx2 [lat,lng]."""
    # Convert to radians
    R = 6371000.0
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    lat1 = lat[:, None]
    lat2 = lat[None, :]
    dlat = lat2 - lat1
    dlon = lon[None, :] - lon[:, None]
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def _haversine_matrix_between(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fallback: great-circle distance matrix (meters) between A (n x 2) and B (m x 2)."""
    R = 6371000.0
    Alat = np.radians(A[:, 0])[:, None]
    Alon = np.radians(A[:, 1])[:, None]
    Blat = np.radians(B[:, 0])[None, :]
    Blon = np.radians(B[:, 1])[None, :]
    dlat = Blat - Alat
    dlon = Blon - Alon
    a = np.sin(dlat/2.0)**2 + np.cos(Alat) * np.cos(Blat) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def osrm_table_batch(coords: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    """
    Square matrix of road distances between all coords.
    """
    n = len(coords)
    M = np.zeros((n, n), dtype=float)
    step = max(1, int(batch_size if batch_size is not None else OSRM_DISTANCE_BATCH_SIZE))
    for i in range(0, n, step):
        from_idx = list(range(i, min(i + step, n)))
        loc_from = coords[from_idx]
        for j in range(0, n, step):
            to_idx = list(range(j, min(j + step, n)))
            loc_to = coords[to_idx]
            all_stack = np.vstack([loc_from, loc_to])
            all_str = _coords_to_string(all_stack)
            srcs = list(range(len(loc_from)))
            dsts = list(range(len(loc_from), len(loc_from) + len(loc_to)))
            url = _build_osrm_table_url(
                ";".join(all_str),
                f"sources={';'.join(map(str,srcs))}&destinations={';'.join(map(str,dsts))}&annotations=distance"
            )
            chunk = np.array(_cached_table(url), dtype=float)
            M[np.ix_(from_idx, to_idx)] = chunk
    return M

def osrm_matrix_between(A: np.ndarray, B: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    """
    Rectangular matrix of distances from each point in A to each point in B.
    """
    nA, nB = len(A), len(B)
    M = np.zeros((nA, nB), dtype=float)
    step = max(1, int(batch_size if batch_size is not None else OSRM_DISTANCE_BATCH_SIZE))
    for i in range(0, nA, step):
        from_idx = list(range(i, min(i + step, nA)))
        loc_from = A[from_idx]
        for j in range(0, nB, step):
            to_idx = list(range(j, min(j + step, nB)))
            loc_to = B[to_idx]
            all_stack = np.vstack([loc_from, loc_to])
            all_str = _coords_to_string(all_stack)
            srcs = list(range(len(loc_from)))
            dsts = list(range(len(loc_from), len(loc_from) + len(loc_to)))
            url = _build_osrm_table_url(
                ";".join(all_str),
                f"sources={';'.join(map(str,srcs))}&destinations={';'.join(map(str,dsts))}&annotations=distance"
            )
            chunk = np.array(_cached_table(url), dtype=float)
            M[np.ix_(from_idx, to_idx)] = chunk
    return M


# ----------------------------
# Territory base suggestion (k-medoids on road distances)
# ----------------------------
def k_medoids(D: np.ndarray, k: int, iters: int = 25, random_state: int = 42) -> np.ndarray:
    """
    PAM-like swap heuristic on a full road-distance matrix D (n x n).
    Returns indices of chosen medoids (length k).
    """
    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be in 1..n")
    medoids = rng.choice(n, size=k, replace=False)
    # Precompute row mins for speed
    def assign_cost(meds):
        a = np.argmin(D[:, meds], axis=1)
        return a, D[np.arange(n), meds[a]].sum()

    _, cur_cost = assign_cost(medoids)
    for _ in range(iters):
        improved = False
        for m_idx in range(k):
            m = medoids[m_idx]
            for h in range(n):
                if h in medoids:
                    continue
                trial = medoids.copy()
                trial[m_idx] = h
                _, new_cost = assign_cost(trial)
                if new_cost + 1e-6 < cur_cost:
                    medoids = trial
                    cur_cost = new_cost
                    improved = True
        if not improved:
            break
    return medoids


def _precluster_days_kmedoids(
    coords: np.ndarray,
    max_calls: int,
    min_ratio: float,
    lam: float,
) -> List[List[int]]:
    """Partition coords into up to k day-sized, compact groups using OSRM road distances + k-medoids
    + H3 contiguity penalty during balanced assign.
    Returns relative index groups.
    """
    n = len(coords)
    if n == 0:
        return []
    if max_calls <= 0:
        return [list(range(n))]
    k = max(1, int(math.ceil(n / max_calls)))
    # Use haversine for k-medoids (fast, robust). Only compute OSRM to centers if enabled.
    D_hav = _haversine_matrix_full(coords)
    meds = k_medoids(D_hav, k, iters=15, random_state=42)
    # Distances from every call to chosen centers
    if int(VRP_KMEDOIDS_OSRM_CENTERS) and n <= max(1, int(VRP_PRECLUSTER_OSRM_MAX_N)):
        try:
            centers = coords[np.array(meds, dtype=int)]
            D_to_centers = osrm_matrix_between(coords, centers)  # road distances to centers only (n x k)
        except Exception:
            D_to_centers = D_hav[:, meds]
    else:
        D_to_centers = D_hav[:, meds]
    min_calls = max(1, int(math.floor((min_ratio if min_ratio is not None else 0.0) * max_calls)))
    try:
        hex_ids = _hex_ids_for_coords(coords, H3_PLAN_RES)
    except Exception:
        hex_ids = None
    assign = balanced_assign(D_to_centers, min_calls=min_calls, max_calls=int(max_calls), hex_ids=hex_ids, lam=float(lam or 0.0))
    groups: List[List[int]] = [[] for _ in range(max(1, int(np.max(assign))+1))]
    for i, a in enumerate(assign):
        groups[int(a)].append(int(i))
    return [g for g in groups if g]


def _refine_day_groups_compact(
    coords: np.ndarray,
    groups: List[List[int]],
    min_calls: int,
    max_calls: int,
    improve_ratio: float = CLUSTER_REFINE_IMPROVE_RATIO,
    iters: int = CLUSTER_REFINE_ITERS,
) -> List[List[int]]:
    """Tighten preclustered day groups by greedily moving obvious outliers to nearer groups
    while respecting min/max sizes. Uses haversine distances to medoid-like centers.
    """
    if not groups:
        return groups
    n = len(coords)
    # assignment array 0..k-1
    k = len(groups)
    assign = -np.ones(n, dtype=int)
    for gi, g in enumerate(groups):
        for idx in g:
            if 0 <= int(idx) < n:
                assign[int(idx)] = gi
    if np.any(assign < 0):
        # ignore bad entries
        valid = np.where(assign >= 0)[0]
        if len(valid) == 0:
            return groups
    # Helper to compute medoid centers and distance-to-centers
    def centers_and_D():
        centers_idx: List[int] = []
        for gi in range(k):
            members = np.where(assign == gi)[0]
            if len(members) == 0:
                centers_idx.append(-1)
                continue
            sub = coords[members]
            # medoid by haversine
            D = _haversine_matrix_full(sub)
            med_rel = int(np.argmin(D.sum(axis=0)))
            centers_idx.append(int(members[med_rel]))
        centers = np.array([[coords[i,0], coords[i,1]] if i >= 0 else [np.nan, np.nan] for i in centers_idx])
        # Distances to centers (meters) via haversine
        valid_centers = centers[~np.isnan(centers[:,0])]
        if len(valid_centers) == 0:
            return centers, np.zeros((n, k), dtype=float)
        D_to_centers = _haversine_matrix_between(coords, centers)
        return centers, D_to_centers

    sizes = [int(np.sum(assign == gi)) for gi in range(k)]
    centers, D2C = centers_and_D()
    improve_ratio = max(1.0, float(improve_ratio))
    for _ in range(max(0, int(iters))):
        changed = False
        # Consider points farthest from their current center first
        order = np.argsort([D2C[i, assign[i]] if assign[i] >= 0 else 0.0 for i in range(n)])[::-1]
        for i in order:
            gi = int(assign[i])
            if gi < 0:
                continue
            cur_cost = float(D2C[i, gi])
            # best alternative center
            best_j = gi
            best_cost = cur_cost
            for j in range(k):
                if j == gi:
                    continue
                c = float(D2C[i, j])
                if c < best_cost:
                    best_cost = c; best_j = j
            # Move if a lot closer and capacity allows and won't violate min on donor
            if best_j != gi and best_cost * improve_ratio < cur_cost:
                if (max_calls <= 0 or sizes[best_j] + 1 <= max_calls) and (min_calls <= 0 or sizes[gi] - 1 >= min_calls):
                    assign[i] = best_j
                    sizes[gi] -= 1; sizes[best_j] += 1
                    changed = True
        if changed:
            centers, D2C = centers_and_D()
        else:
            break
    # Rebuild groups
    new_groups: List[List[int]] = [[] for _ in range(k)]
    for i in range(n):
        gi = int(assign[i])
        if gi >= 0:
            new_groups[gi].append(int(i))
    return [g for g in new_groups if g]


# ----------------------------
# Balanced assignment (min/max per territory)
# ----------------------------
def balanced_assign(
    D_call_to_center: np.ndarray,
    min_calls: int,
    max_calls: int,
    hex_ids: Optional[list] = None,
    lam: float = 0.0,
) -> np.ndarray:
    """
    Greedy min-cost assignment to centers with capacity enforcement.
    D_call_to_center: (n_calls x k) distances
    Returns: assignment array (len n_calls) with values in [0..k-1]
    """
    n, k = D_call_to_center.shape
    if max_calls <= 0:
        raise ValueError("max_calls must be > 0")
    if min_calls < 0:
        raise ValueError("min_calls must be >= 0")
    if k * max_calls < n:
        # Infeasible under strict caps → soften by allowing overflow; we'll still get something sensible.
        pass
    if k * min_calls > n:
        # Infeasible mins → reduce mins in effect
        min_calls = n // k

    # First pass: nearest center with optional H3 contiguity penalty, respect max
    cap = np.zeros(k, dtype=int)
    assign = -np.ones(n, dtype=int)

    # Running counts per hex and per (hex, center)
    use_penalty = bool(hex_ids) and lam > 0
    if use_penalty:
        hex_total: dict = {}                 # hex -> total assigned so far
        hex_center: dict = {}                # (hex, center) -> assigned count
    else:
        hex_total = {}
        hex_center = {}

    # Determine processing order: group by hex to accumulate penalty locally
    if use_penalty:
        order_idx = list(range(n))
        try:
            order_idx.sort(key=lambda i: (str(hex_ids[i]), float(np.min(D_call_to_center[i, :]))) )
        except Exception:
            pass
    else:
        order_idx = range(n)

    for i in order_idx:
        chosen_j = -1
        if use_penalty:
            h = hex_ids[i]
            best_cost = float("inf")
            for j in range(k):
                if cap[j] >= max_calls:
                    continue
                base = float(D_call_to_center[i, j])
                # penalty = lam * (# same-hex already assigned to OTHER centers)
                if h is None:
                    penalty = 0.0
                else:
                    tot = hex_total.get(h, 0)
                    own = hex_center.get((h, j), 0)
                    penalty = lam * max(0, tot - own)
                cost = base + penalty
                if cost < best_cost:
                    best_cost = cost
                    chosen_j = j
        else:
            # Original: choose nearest available
            # Iterate by increasing distance
            order = np.argsort(D_call_to_center[i, :])
            for j in order:
                if cap[j] < max_calls:
                    chosen_j = j
                    break
        if chosen_j == -1:
            # no capacity; leave for second pass
            continue
        assign[i] = chosen_j
        cap[chosen_j] += 1
        if use_penalty:
            h = hex_ids[i]
            if h is not None:
                hex_total[h] = hex_total.get(h, 0) + 1
                hex_center[(h, chosen_j)] = hex_center.get((h, chosen_j), 0) + 1

    # Second pass: any unassigned → put into least loaded (even if equals max)
    for i in np.where(assign == -1)[0]:
        if use_penalty:
            h = hex_ids[i]
            best = None
            best_cost = float('inf')
            for j in range(k):
                base = float(D_call_to_center[i, j])
                if h is None:
                    penalty = 0.0
                else:
                    tot = hex_total.get(h, 0)
                    own = hex_center.get((h, j), 0)
                    penalty = lam * max(0, tot - own)
                # Prefer lower load slightly to avoid extreme imbalance
                load_bias = 1e-6 * cap[j]
                cost = base + penalty + load_bias
                if cost < best_cost:
                    best_cost = cost
                    best = j
            j = int(best if best is not None else int(np.argmin(cap)))
        else:
            j = int(np.argmin(cap))
        assign[i] = j
        cap[j] += 1
        if use_penalty:
            if h is not None:
                hex_total[h] = hex_total.get(h, 0) + 1
                hex_center[(h, j)] = hex_center.get((h, j), 0) + 1

    # Enforce mins by moving cheapest callers from overfull to underfull
    need = np.maximum(0, min_calls - cap)
    have = np.maximum(0, cap - min_calls)

    if need.sum() > 0:
        need_idxs = list(np.where(need > 0)[0])
        for t_need in need_idxs:
            required = need[t_need]
            if required == 0:
                continue
            # Build a pool of movable callers from donors
            donors = list(np.where(have > 0)[0])
            # For each donor, candidates sorted by delta cost
            move_candidates: List[Tuple[float, int, int]] = []  # (delta, call_idx, donor_t)
            for t_d in donors:
                if have[t_d] == 0:
                    continue
                call_idxs = np.where(assign == t_d)[0]
                if len(call_idxs) == 0:
                    continue
                deltas = D_call_to_center[call_idxs, t_need] - D_call_to_center[call_idxs, t_d]
                order_move = np.argsort(deltas)
                for om in order_move:
                    u = call_idxs[om]
                    move_candidates.append((float(deltas[om]), int(u), int(t_d)))
            move_candidates.sort(key=lambda x: x[0])
            # Execute cheapest moves
            for _, u, t_d in move_candidates:
                if required == 0:
                    break
                if have[t_d] == 0:
                    continue
                assign[u] = t_need
                cap[t_d] -= 1; have[t_d] -= 1
                cap[t_need] += 1; required -= 1
            need[t_need] = required
    return assign


# ----------------------------
# Day clustering (CVRP per territory with depot at base)
# ----------------------------
def cvrp_days_for_territory(
    terr_coords: np.ndarray,
    depot_lat_lng: Tuple[float, float],
    min_calls: int,
    max_calls: int,
    time_sec: int = 30
) -> List[List[int]]:
    """
    Returns list of routes (each a list of relative indices into terr_coords).
    """
    n = len(terr_coords)
    if n == 0:
        return []

    # Compactness-first split using polar sweep around depot
    dep_lat, dep_lng = float(depot_lat_lng[0]), float(depot_lat_lng[1])
    # Angle of each point around the depot (x=lng, y=lat)
    angles = np.arctan2(terr_coords[:, 0] - dep_lat, terr_coords[:, 1] - dep_lng)
    order = np.argsort(angles)
    # Rotate to start after the largest angular gap to avoid wrap-around splitting
    ang_sorted = angles[order]
    gaps = np.diff(np.r_[ang_sorted, ang_sorted[0] + 2*np.pi])
    start = int((np.argmax(gaps) + 1) % n)
    order = np.r_[order[start:], order[:start]]

    # Number of days by capacity
    k = 1
    if max_calls and max_calls > 0:
        k = int(np.ceil(n / max_calls))
    k = max(1, k)

    # Ideal sizes, then clamp to [min_calls, max_calls]
    base = n // k
    rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    lo = max(1, int(min_calls or 1))
    hi = int(max_calls or (n if k == 1 else base + rem))
    sizes = [int(min(max(s, lo), hi)) for s in sizes]
    # Fix total size to n by adjusting last buckets
    diff = n - sum(sizes)
    if diff != 0:
        # distribute diff across buckets without violating bounds
        i = 0
        while diff != 0 and i < k * 2:
            j = i % k
            if diff > 0 and sizes[j] < hi:
                sizes[j] += 1; diff -= 1
            elif diff < 0 and sizes[j] > lo:
                sizes[j] -= 1; diff += 1
            i += 1
    # As a last resort, force the last bucket to absorb any remainder
    if sum(sizes) != n:
        sizes[-1] += (n - sum(sizes))

    routes: List[List[int]] = []
    idx = 0
    for s in sizes:
        if s <= 0:
            routes.append([])
            continue
        routes.append(order[idx: idx + s].tolist())
        idx += s

    # Hard-cap enforcement: split any oversized routes into chunks of size <= max_calls
    if max_calls > 0:
        capped: List[List[int]] = []
        for r in routes:
            if len(r) <= max_calls:
                capped.append(r)
            else:
                for i in range(0, len(r), max_calls):
                    capped.append(r[i:i + max_calls])
        routes = capped

    # Optional light rebalance to avoid tiny final days when min_calls is set
    if min_calls and min_calls > 0 and len(routes) > 1:
        # Move a few items from larger routes into undersized ones, greedily
        # Note: we don't change order, just transfer from the end of larger routes
        changed = True
        # Cap iterations to avoid infinite loops
        for _ in range(10):
            changed = False
            # Find undersized routes
            small_idxs = [i for i, r in enumerate(routes) if len(r) < min_calls]
            if not small_idxs:
                break
            # Find donors (routes strictly above min_calls)
            donor_idxs = [i for i, r in enumerate(routes) if len(r) > min_calls]
            if not donor_idxs:
                break
            for si in small_idxs:
                need = min_calls - len(routes[si])
                if need <= 0:
                    continue
                # take from the largest donor first
                donor_idxs.sort(key=lambda i: len(routes[i]), reverse=True)
                for di in donor_idxs:
                    while len(routes[di]) > min_calls and need > 0 and routes[di]:
                        routes[si].append(routes[di].pop())
                        need -= 1
                        changed = True
                    if need <= 0:
                        break
            if not changed:
                break

    return routes


# ----------------------------
# Cluster helper utilities
# ----------------------------
def _infer_service_seconds(df: pd.DataFrame) -> Tuple[List[int], Optional[str]]:
    candidates = (
        "duration", "dur", "mins", "minutes", "minute", "service", "servicetime",
        "calltime", "visittime", "appt", "appointment", "length"
    )
    default_sec = int(max(1.0, CLUSTER_DEFAULT_SERVICE_MIN) * 60)
    picked: Optional[str] = None
    for col in df.columns:
        key = str(col).lower()
        if any(token in key for token in candidates):
            picked = col
            break
    if picked is None:
        return [default_sec for _ in range(len(df))], None
    series = df[picked]
    out: List[int] = []
    for val in series:
        sec = parse_hhmm_to_seconds(val)
        if sec <= 0:
            sec = default_sec
        out.append(int(sec))
    return out, picked


def _estimate_target_minutes(service_seconds: np.ndarray, max_calls: int) -> float:
    if len(service_seconds) == 0:
        return max(CLUSTER_BASE_DAY_MINUTES, CLUSTER_DEFAULT_SERVICE_MIN * max(1, max_calls))
    median_sec = float(np.median(service_seconds)) if np.any(service_seconds) else 0.0
    if median_sec <= 0:
        median_sec = CLUSTER_DEFAULT_SERVICE_MIN * 60.0
    max_calls_eff = max(1, int(max_calls) if max_calls and max_calls > 0 else len(service_seconds))
    est = (median_sec * max_calls_eff) / 60.0
    return max(CLUSTER_BASE_DAY_MINUTES, est)


def _cluster_days_vrp(
    terr_coords: np.ndarray,
    service_seconds: np.ndarray,
    depot_lat_lng: Tuple[float, float],
    min_calls: int,
    max_calls: int,
    solver_time_limit: int,
    target_minutes: float,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not _HAS_ORTOOLS:
        raise RuntimeError("OR-Tools not available")
    n = int(len(terr_coords))
    if n == 0:
        return [], {"drive_minutes": [], "service_minutes": [], "target_day_minutes": target_minutes}

    depot = np.array([[float(depot_lat_lng[0]), float(depot_lat_lng[1])]], dtype=float)
    all_pts = np.vstack([terr_coords.astype(float), depot])
    dm = osrm_table_batch_duration(all_pts)
    depot_idx = n

    manager = pywrapcp.RoutingIndexManager(n + 1, 1, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def travel_cb(from_index: int, to_index: int) -> int:
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(max(0, dm[f, t]))

    transit = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    hard_penalty = max(10_000_000, VRP_DROP_PENALTY)
    for i in range(n):
        routing.AddDisjunction([routing.NodeToIndex(i)], hard_penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    try:
        first_map = {
            "parallel_cheapest_insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            "path_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            "savings": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            "all_unperformed": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        }
        params.first_solution_strategy = first_map.get(VRP_FIRST, routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    except Exception:
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    try:
        meta = (VRP_META or "gls").lower()
        if meta in ("gls", "guided", "guided_local_search"):
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        elif meta in ("sa", "anneal", "simulated_annealing"):
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        else:
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    except Exception:
        pass
    try:
        params.random_seed = int(seed)
    except Exception:
        params.random_seed = 42
    params.time_limit.seconds = max(3, int(solver_time_limit or CLUSTER_SOLVER_TIME_SEC))
    params.log_search = False

    tries = max(1, int(VRP_MS_TRIES))
    best_solution = None
    best_obj = None
    for t in range(tries):
        try:
            params.random_seed = int(seed + t * 7919)
        except Exception:
            pass
        sol = routing.SolveWithParameters(params)
        if not sol:
            continue
        obj = sol.ObjectiveValue()
        if best_solution is None or obj < best_obj:
            best_solution = sol
            best_obj = obj
    solution = best_solution
    if not solution:
        raise RuntimeError("OR-Tools failed to compute tour")

    order: List[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node != depot_idx:
            order.append(int(node))
        idx = solution.Value(routing.NextVar(idx))
    if len(order) != n:
        missing = sorted(set(range(n)) - set(order))
        order.extend(missing)

    min_calls_eff = max(1, int(min_calls) if min_calls and min_calls > 0 else 1)
    max_calls_eff = int(max_calls) if max_calls and max_calls > 0 else n
    max_calls_eff = max(min_calls_eff, max_calls_eff)
    svc = service_seconds.astype(float)

    prefix_service = [0.0]
    for node in order:
        prefix_service.append(prefix_service[-1] + float(svc[node]))
    prefix_edges = [0.0]
    for pos in range(1, len(order)):
        prev = order[pos - 1]
        cur = order[pos]
        prefix_edges.append(prefix_edges[-1] + float(dm[prev, cur]))

    target_minutes = max(1.0, float(target_minutes))

    def _chunk_cost(i: int, j: int) -> float:
        start = order[i]
        end = order[j - 1]
        travel = float(dm[depot_idx, start]) + float(dm[end, depot_idx])
        if j - i > 1:
            travel += float(prefix_edges[j - 1] - prefix_edges[i])
        service = float(prefix_service[j] - prefix_service[i])
        svc_minutes = service / 60.0
        travel_minutes = travel / 60.0
        over = max(0.0, svc_minutes - target_minutes)
        penalty = over * over * CLUSTER_DURATION_PENALTY
        return travel_minutes + penalty

    n_order = len(order)
    dp = [math.inf] * (n_order + 1)
    choice = [-1] * (n_order + 1)
    dp[n_order] = 0.0
    for i in range(n_order - 1, -1, -1):
        max_j = min(n_order, i + max_calls_eff)
        min_j = max(i + min_calls_eff, i + 1)
        for j in range(min_j, max_j + 1):
            remaining = n_order - j
            if remaining > 0 and remaining < min_calls_eff:
                continue
            cost = _chunk_cost(i, j) + dp[j]
            if cost < dp[i]:
                dp[i] = cost
                choice[i] = j

    routes: List[List[int]] = []
    cursor = 0
    while cursor < n_order and choice[cursor] != -1:
        nxt = choice[cursor]
        routes.append(order[cursor:nxt])
        cursor = nxt
    if cursor < n_order:
        routes.append(order[cursor:])

    if not routes:
        routes = [order[:]]

    drive_minutes: List[float] = []
    service_minutes: List[float] = []
    for seg in routes:
        if not seg:
            continue
        travel = float(dm[depot_idx, seg[0]]) + float(dm[seg[-1], depot_idx])
        for a, b in zip(seg[:-1], seg[1:]):
            travel += float(dm[a, b])
        service_total = sum(float(svc[node]) for node in seg)
        drive_minutes.append(round(travel / 60.0, 3))
        service_minutes.append(round(service_total / 60.0, 3))

    return routes, {
        "drive_minutes": drive_minutes,
        "service_minutes": service_minutes,
        "target_day_minutes": target_minutes,
    }


def _sa_polish_day_routes(
    routes: List[List[int]],
    duration_matrix: np.ndarray,
    service_seconds: np.ndarray,
    target_minutes: float,
    min_calls: int,
    max_calls: int,
    iterations: int,
    init_temp: float,
    cooling: float,
) -> List[List[int]]:
    if not routes or len(routes) < 2 or iterations <= 0 or init_temp <= 0 or cooling <= 0:
        return routes
    depot_idx = duration_matrix.shape[0] - 1
    svc_minutes = [float(v) / 60.0 for v in service_seconds.tolist()]

    def route_travel_minutes(nodes: List[int]) -> float:
        if not nodes:
            return 0.0
        unvisited = list(nodes)
        travel = 0.0
        current = depot_idx
        local_nodes = unvisited[:]
        while local_nodes:
            nxt = min(local_nodes, key=lambda idx: duration_matrix[current, idx])
            travel += duration_matrix[current, nxt]
            current = nxt
            local_nodes.remove(nxt)
        travel += duration_matrix[current, depot_idx]
        return travel / 60.0

    def routes_cost(solution: List[List[int]]) -> float:
        total = 0.0
        for day_nodes in solution:
            if not day_nodes:
                continue
            travel = route_travel_minutes(day_nodes)
            service = sum(svc_minutes[i] for i in day_nodes)
            over = max(0.0, service - target_minutes)
            total += travel + (over * over * CLUSTER_DURATION_PENALTY)
        return total

    def valid_move(src_len: int, dst_len: int) -> bool:
        if max_calls > 0 and dst_len + 1 > max_calls:
            return False
        if min_calls > 0 and src_len - 1 < min_calls:
            return False
        return True

    current = [sorted(day) for day in routes]
    best = [day[:] for day in current]
    current_cost = routes_cost(current)
    best_cost = current_cost
    temp = float(init_temp)

    for step in range(int(iterations)):
        if temp < 1e-3:
            break
        new_solution: Optional[List[List[int]]] = None
        for _ in range(16):
            if len(current) < 2:
                break
            kind = random.random()
            cand = [day[:] for day in current]
            if kind < 0.5:
                src_idx = random.randrange(len(cand))
                dst_idx = random.randrange(len(cand))
                if src_idx == dst_idx:
                    continue
                src = cand[src_idx]
                dst = cand[dst_idx]
                if not src:
                    continue
                if not valid_move(len(src), len(dst)):
                    continue
                # Block move: move a node and optionally its nearest neighbor by duration_matrix to the same day
                node = random.choice(src)
                block = [node]
                # try to add a close neighbor from the same src day
                if len(src) >= 2:
                    try:
                        nearest = min([i for i in src if i != node], key=lambda j: float(duration_matrix[node, j]))
                        block.append(nearest)
                    except Exception:
                        pass
                # respect max_calls when moving a block
                if max_calls > 0 and len(dst) + len(block) > max_calls:
                    continue
                for b in block:
                    if b in src:
                        src.remove(b)
                        dst.append(b)
                dst.sort(); src.sort()
                cand[src_idx] = src
                cand[dst_idx] = dst
                new_solution = cand
                break
            else:
                a_idx = random.randrange(len(cand))
                b_idx = random.randrange(len(cand))
                if a_idx == b_idx:
                    continue
                a = cand[a_idx]
                b = cand[b_idx]
                if not a or not b:
                    continue
                node_a = random.choice(a)
                node_b = random.choice(b)
                if min_calls > 0 and (len(a) < min_calls or len(b) < min_calls):
                    continue
                a[a.index(node_a)] = node_b
                b[b.index(node_b)] = node_a
                a.sort()
                b.sort()
                cand[a_idx] = a
                cand[b_idx] = b
                new_solution = cand
                break
        if new_solution is None:
            temp *= cooling
            continue
        neighbor_cost = routes_cost(new_solution)
        delta = neighbor_cost - current_cost
        accept = False
        if delta < 0:
            accept = True
        else:
            try:
                prob = math.exp(-delta / max(temp, 1e-6))
            except Exception:
                prob = 0.0
            if random.random() < prob:
                accept = True
        if accept:
            current = [day[:] for day in new_solution]
            current_cost = neighbor_cost
            if neighbor_cost + 1e-6 < best_cost:
                best = [day[:] for day in new_solution]
                best_cost = neighbor_cost
        temp *= cooling
    # Optional bounded greedy pair swaps (lightweight local improvement)
    def try_swap_once(sol: List[List[int]]) -> Optional[List[List[int]]]:
        n_days = len(sol)
        if n_days < 2:
            return None
        base_cost = routes_cost(sol)
        for a_idx in range(n_days):
            for b_idx in range(a_idx + 1, n_days):
                a = sol[a_idx]
                b = sol[b_idx]
                if not a or not b:
                    continue
                # sample a few candidates to keep it fast
                sample_a = a if len(a) <= 6 else random.sample(a, 6)
                sample_b = b if len(b) <= 6 else random.sample(b, 6)
                for na in sample_a:
                    for nb in sample_b:
                        # capacity constraints unaffected by 1-1 swap
                        cand = [day[:] for day in sol]
                        ca = cand[a_idx]; cb = cand[b_idx]
                        try:
                            ca[ca.index(na)] = nb
                            cb[cb.index(nb)] = na
                        except Exception:
                            continue
                        ca.sort(); cb.sort()
                        cand[a_idx] = ca; cand[b_idx] = cb
                        new_cost = routes_cost(cand)
                        if new_cost + 1e-9 < base_cost:
                            return cand
        return None

    for _ in range(3):
        cand = try_swap_once(best)
        if cand is None:
            break
        best = cand
    return best


# ----------------------------
# Helpers for VRP with time windows
# ----------------------------
_DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def parse_weekdays(bits) -> List[int]:
    """Parse various representations of working days into indices [0=Mon..6=Sun].
    Supported:
    - Classic 7-char Y/N string: e.g., 'YYYYYNN'
    - Numeric day list: '1,3,5' (1=Mon..7=Sun)
    - Labels like 'Day 1' or 'DAY 2 5' (maps 1..7 -> Mon..Sun)
    - Weekday names/abbr separated by non-letters: 'Mon Tue', 'mon,wed,fri'
    - Python list/iterable of ints or strings as above
    """
    # If already a list-like, flatten to a string and recurse element-wise
    try:
        if isinstance(bits, (list, tuple, set)):
            out: List[int] = []
            for b in bits:
                for d in parse_weekdays(b):
                    if d not in out:
                        out.append(d)
            return sorted(out)
    except Exception:
        pass

    s_raw = str(bits or "").strip()
    if not s_raw:
        return []
    s = s_raw.upper()

    # 1) Strict Y/N 7-char mask (back-compat)
    if len(s) >= 1 and all(ch in "Y N" for ch in s.replace(" ", "")):
        m = (s.replace(" ", "") + "NNNNNNN")[:7]
        return [i for i, ch in enumerate(m) if ch == 'Y']

    # 2) Extract all numbers (e.g., '1,3,5' or 'DAY 2 5')
    try:
        import re as _re
        nums = [int(x) for x in _re.findall(r"\d+", s)]
        days_num = [n for n in nums if 1 <= n <= 7]
        if days_num:
            return sorted({n-1 for n in days_num})
    except Exception:
        pass

    # 3) Weekday names/abbr: map tokens to indices
    day_map = {
        "MON":0, "MONDAY":0,
        "TUE":1, "TUES":1, "TUESDAY":1,
        "WED":2, "WEDNESDAY":2,
        "THU":3, "THUR":3, "THURS":3, "THURSDAY":3,
        "FRI":4, "FRIDAY":4,
        "SAT":5, "SATURDAY":5,
        "SUN":6, "SUNDAY":6,
    }
    tokens = [t for t in re_split_nonalpha(s) if t]
    mapped = [day_map[t] for t in tokens if t in day_map]
    if mapped:
        # dedupe and sort
        return sorted(set(mapped))

    # Fallback: no valid days
    return []

def re_split_nonalpha(text: str) -> List[str]:
    try:
        import re as _re
        return [t for t in _re.split(r"[^A-Z]+", text.upper()) if t]
    except Exception:
        return [text.upper()]

def parse_hhmm_to_seconds(val) -> int:
    """Parse a time-of-day or duration-like value into seconds.
    Handles:
    - Excel time numerics (fraction of a day, e.g., 0.375 -> 9:00)
    - "h:mm" or "hh:mm" strings
    - Bare numbers: interpret smartly as HHMM, hours, or minutes
    - AM/PM variants like "9:00 AM" or "1700" (-> 17:00)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        # Numeric handling first (Excel often stores times as day fractions)
        if isinstance(val, (int, float)):
            f = float(val)
            if 0 < f <= 1.0:
                # Excel day fraction -> seconds from midnight
                return int(round(f * 86400))
            # If looks like hours (<= 24), treat as hours-of-day
            if 0 < f <= 24:
                return int(round(f * 3600))
            # If looks like minutes (< one day in minutes)
            if 24 < f < 1440:
                return int(round(f * 60))
            # Otherwise assume already seconds
            return int(round(f))

        s = str(val).strip()
        if not s:
            return 0

        sl = s.lower().replace(" ", "")
        # AM/PM handling
        ampm = None
        if sl.endswith("am"):
            ampm = "am"; sl = sl[:-2]
        elif sl.endswith("pm"):
            ampm = "pm"; sl = sl[:-2]

        if ":" in sl:
            parts = sl.split(":")
            h = int(parts[0]) if parts[0] else 0
            m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            if ampm == "pm" and h < 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
            return int(h * 3600 + m * 60)

        # If pure digits: could be HHMM (e.g., 900 or 0900 or 1730)
        if sl.isdigit():
            num = int(sl)
            if num < 100:  # treat as hours
                h = num; m = 0
            elif num < 2400:  # HHMM
                h = num // 100
                m = num % 100
            else:
                # Large number: assume minutes
                h = 0
                m = num
            if ampm == "pm" and h < 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
            return int(h * 3600 + m * 60)

        # Fallback: numeric string as minutes
        return int(round(float(sl) * 60))
    except Exception:
        return 0

def parse_duration_to_seconds(val) -> int:
    """Parse call duration into seconds.
    Rules (per user):
    - Numbers are minutes: 90 => 90 minutes.
    - h:mm strings are hours:minutes: 1:30 => 1h30m.
    - Numeric strings are minutes; decimals interpret as fractional minutes.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        if isinstance(val, (int, float)):
            return int(round(float(val) * 60))
        s = str(val).strip()
        if not s:
            return 0
        if ":" in s:
            parts = s.split(":")
            h = int(parts[0]) if parts[0] else 0
            m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            return int(h * 3600 + m * 60)
        # default: minutes
        return int(round(float(s) * 60))
    except Exception:
        return 0

def vrp_single_route_with_time(
    coords: np.ndarray,
    depot_lat_lng: Tuple[float, float],
    service_times_sec: List[int],
    start_sec: int,
    end_sec: int,
    time_limit_sec: int = 20,
    max_stops: Optional[int] = None,
    seed: Optional[int] = 42,
    durations_matrix: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    """
    Solve a single-vehicle route visiting a subset of coords (may drop nodes) within the time window.
    Returns: (visited_relative_indices_in_order, dropped_relative_indices)
    """
    n = len(coords)
    if n == 0:
        return [], []
    depot = np.array([depot_lat_lng], dtype=float)
    all_pts = np.vstack([coords, depot])
    dm = durations_matrix if durations_matrix is not None else osrm_table_batch_duration(all_pts)
    depot_node = n

    manager = pywrapcp.RoutingIndexManager(n + 1, 1, depot_node)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        travel = int(dm[f, t])
        service = 0 if f == depot_node else int(service_times_sec[f])
        return max(0, travel + service)

    transit = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # Time dimension
    horizon = max(86400, int(end_sec + 3600))  # at least a day horizon
    routing.AddDimension(
        transit,
        0,           # no waiting slack
        horizon,
        False,       # start cumul not forced to 0
        'Time'
    )
    time_dim = routing.GetDimensionOrDie('Time')

    # Time windows: depot and all nodes share the same daily window
    start, end = int(start_sec), int(end_sec)
    # Depot start/end
    depot_start_idx = routing.Start(0)
    depot_end_idx = routing.End(0)
    time_dim.CumulVar(depot_start_idx).SetRange(start, end)
    time_dim.CumulVar(depot_end_idx).SetRange(start, end)

    # Optional max-stops/day capacity
    if max_stops is not None and int(max_stops) > 0:
        def count_cb(from_index):
            node = manager.IndexToNode(from_index)
            return 0 if node == depot_node else 1
        count = routing.RegisterUnaryTransitCallback(count_cb)
        routing.AddDimensionWithVehicleCapacity(count, 0, [int(max_stops)], True, 'Count')

    # Allow dropping nodes with a penalty
    penalty = int(VRP_DROP_PENALTY)
    for i in range(n):
        index = manager.NodeToIndex(i)
        time_dim.CumulVar(index).SetRange(start, end)
        routing.AddDisjunction([index], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    # Good first solution to seed local search (tunable)
    try:
        first_map = {
            "parallel_cheapest_insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            "path_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            "savings": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            "all_unperformed": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        }
        params.first_solution_strategy = first_map.get(VRP_FIRST, routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    except Exception:
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # Metaheuristic (configurable: gls|sa|greedy)
    try:
        meta = (VRP_META or "gls").lower()
        if meta in ("gls", "guided", "guided_local_search"):
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        elif meta in ("sa", "anneal", "simulated_annealing"):
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        else:
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    except Exception:
        pass
    try:
        params.random_seed = int(seed or 42)
    except Exception:
        pass
    # Per-route time limit (env overrides function arg if set)
    route_limit = int(VRP_ROUTE_TIME_LIMIT_SEC or 0) or int(time_limit_sec or 0) or 15
    params.time_limit.seconds = max(5, route_limit)
    try:
        params.number_of_workers = int(VRP_NUM_WORKERS)
    except Exception:
        pass
    params.log_search = False
    # Optional LNS time
    try:
        if int(VRP_LNS_TIME_SEC) > 0:
            params.lns_time_limit.seconds = int(VRP_LNS_TIME_SEC)
    except Exception:
        pass

    # Multi-start: retry with different seeds and keep the best
    best_solution = None
    best_obj = None
    total_deadline = None
    if int(VRP_MS_TOTAL_SEC) > 0:
        total_deadline = time.time() + int(VRP_MS_TOTAL_SEC)
    tries = max(1, int(VRP_MS_TRIES))
    for t in range(tries):
        if total_deadline is not None and time.time() >= total_deadline:
            break
        try:
            params.random_seed = int((seed or 42) + t * 9973)
        except Exception:
            pass
        sol = routing.SolveWithParameters(params)
        if not sol:
            continue
        # objective: total arc cost
        obj = sol.ObjectiveValue()
        if best_solution is None or obj < best_obj:
            best_solution = sol
            best_obj = obj
    solution = best_solution
    if not solution:
        return [], list(range(n))

    # Extract route and dropped
    visited: List[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node != depot_node:
            visited.append(node)
        idx = solution.Value(routing.NextVar(idx))

    dropped: List[int] = []
    for i in range(n):
        index = manager.NodeToIndex(i)
        if solution.Value(routing.NextVar(index)) == index:  # not on route
            dropped.append(i)
    return visited, dropped


# ----------------------------
# Helpers for VRP V2 (time-budget per territory)
# ----------------------------
def _compute_depot_for_territory(terr_coords: np.ndarray, strategy: str = "medoid") -> Tuple[float, float]:
    """Return (lat,lng) depot for a set of coords using strategy 'medoid' or 'nearest' (centroid)."""
    if len(terr_coords) == 0:
        return (float('nan'), float('nan'))
    if len(terr_coords) == 1:
        return (float(terr_coords[0, 0]), float(terr_coords[0, 1]))
    strategy = (strategy or "medoid").lower()
    if strategy == "medoid":
        D = osrm_table_batch(terr_coords)
        med = int(k_medoids(D, 1, iters=15, random_state=42)[0])
        return (float(terr_coords[med, 0]), float(terr_coords[med, 1]))
    # nearest to centroid fallback
    cy = float(np.mean(terr_coords[:, 0]))
    cx = float(np.mean(terr_coords[:, 1]))
    d = np.sum((terr_coords - np.array([[cy, cx]]))**2, axis=1)
    i = int(np.argmin(d))
    return (float(terr_coords[i, 0]), float(terr_coords[i, 1]))

def _routes_time_budget(
    coords: np.ndarray,
    service_times_sec: List[int],
    depot_lat_lng: Tuple[float, float],
    work_day_sec: int,
    time_limit_sec: int = 20
) -> List[List[int]]:
    """Greedy multi-day: repeatedly solve a single time-windowed route until all calls scheduled or no progress."""
    remaining = list(range(len(coords)))
    routes: List[List[int]] = []
    if len(remaining) == 0:
        return routes
    while remaining:
        sub_coords = coords[remaining]
        sub_serv = [int(service_times_sec[i]) for i in remaining]
        visited_rel, dropped_rel = vrp_single_route_with_time(
            sub_coords, depot_lat_lng, sub_serv, 0, int(work_day_sec), time_limit_sec=max(5, int(time_limit_sec))
        )
        if not visited_rel:
            # cannot place more under time budget
            break
        # Map back to absolute indices
        route_abs = [int(remaining[r]) for r in visited_rel]
        routes.append(route_abs)
        # Remove visited from remaining
        rem_set = set(remaining)
        for r in route_abs:
            if r in rem_set:
                rem_set.remove(r)
        remaining = sorted(list(rem_set))
    return routes
def vrp_routes_time_budget(
    coords: np.ndarray,
    service_times_sec: List[int],
    depot_lat_lng: Tuple[float, float],
    work_day_sec: int,
    time_limit_sec: int = 20,
    durations_matrix: Optional[np.ndarray] = None,
    windows: Optional[List[Tuple[int,int]]] = None,
    priorities: Optional[List[float]] = None,
    break_minutes: int = 0,
    max_route_minutes: Optional[int] = None,
    seed: Optional[int] = 42,
) -> List[List[int]]:
    """Multi-vehicle time-budget VRP for one territory. Returns list of routes (list of relative indices)."""
    n = len(coords)
    if n == 0:
        return []
    depot = np.array([depot_lat_lng], dtype=float)
    all_pts = np.vstack([coords, depot])
    dm = durations_matrix if durations_matrix is not None else osrm_table_batch_duration(all_pts)
    depot_node = n

    # crude upper bound on vehicles/days
    est_service = sum(max(0, int(s)) for s in service_times_sec)
    v_cap = max(1, int(np.ceil(est_service / max(1, work_day_sec))))
    num_vehicles = min(n, v_cap)

    manager = pywrapcp.RoutingIndexManager(n + 1, num_vehicles, depot_node)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        travel = int(dm[f, t])
        service = 0 if f == depot_node else int(service_times_sec[f])
        return max(0, travel + service)

    transit = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # Time dimension with cap = work_day_sec or max_route_minutes override
    cap = int(work_day_sec)
    if max_route_minutes is not None and int(max_route_minutes) > 0:
        cap = min(cap, int(max_route_minutes) * 60)
    routing.AddDimension(transit, 0, cap, True, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')

    # Optional windows
    if windows is not None and len(windows) == n:
        for i, (ws, we) in enumerate(windows):
            idx = routing.NodeToIndex(i)
            time_dim.CumulVar(idx).SetRange(int(max(0, ws)), int(max(0, we)))

    # Soft dropping with priority-aware penalties
    base_penalty = 5_000_000
    for i in range(n):
        p = float(priorities[i]) if priorities and i < len(priorities) and priorities[i] not in (None, np.nan) else 1.0
        p = max(0.1, float(p))
        routing.AddDisjunction([routing.NodeToIndex(i)], int(base_penalty / p))

    # Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    try:
        first_map = {
            "parallel_cheapest_insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            "path_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            "savings": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            "all_unperformed": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        }
        params.first_solution_strategy = first_map.get(VRP_FIRST, routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    except Exception:
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    try:
        meta = (VRP_META or "gls").lower()
        if meta in ("gls", "guided", "guided_local_search"):
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        elif meta in ("sa", "anneal", "simulated_annealing"):
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        else:
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    except Exception:
        pass
    try:
        params.random_seed = int(seed or 42)
    except Exception:
        pass
    params.time_limit.seconds = max(5, int(time_limit_sec))
    try:
        params.number_of_workers = int(VRP_NUM_WORKERS)
    except Exception:
        pass
    # Optional LNS time if available
    try:
        if int(VRP_LNS_TIME_SEC) > 0:
            params.lns_time_limit.seconds = int(VRP_LNS_TIME_SEC)
    except Exception:
        pass

    # Multi-start best-of
    best_solution = None
    best_obj = None
    tries = max(1, int(VRP_MS_TRIES))
    for t in range(tries):
        try:
            params.random_seed = int((seed or 42) + t * 7919)
        except Exception:
            pass
        sol = routing.SolveWithParameters(params)
        if not sol:
            continue
        obj = sol.ObjectiveValue()
        if best_solution is None or obj < best_obj:
            best_solution = sol
            best_obj = obj
    solution = best_solution
    routes: List[List[int]] = []
    if solution:
        for v in range(num_vehicles):
            idx = routing.Start(v)
            route = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != depot_node:
                    route.append(node)
                idx = solution.Value(routing.NextVar(idx))
            if route:
                routes.append(route)
    return routes

def _route_kpis(coords_route: np.ndarray, depot: Tuple[float,float]) -> Tuple[float, float, float]:
    """Return (drive_minutes, service_minutes, total_km) for a single route.
    Service minutes cannot be known from coords alone; caller should sum durations separately.
    """
    if len(coords_route) == 0:
        return 0.0, 0.0, 0.0
    all_pts = np.vstack([coords_route, np.array([[depot[0], depot[1]]])])
    dm = osrm_table_batch_duration(all_pts)  # seconds
    dd = osrm_table_batch(all_pts)           # meters
    depot_idx = len(coords_route)
    # drive time: depot->first + legs + last->depot
    drive_s = float(dm[depot_idx, 0])
    dist_m = float(dd[depot_idx, 0])
    for i in range(1, len(coords_route)):
        drive_s += float(dm[i-1, i])
        dist_m += float(dd[i-1, i])
    drive_s += float(dm[len(coords_route)-1, depot_idx])
    dist_m += float(dd[len(coords_route)-1, depot_idx])
    return drive_s / 60.0, 0.0, dist_m / 1000.0  # service minutes filled by caller


# ----------------------------
# Territory planning orchestration
# ----------------------------
def plan_territories(
    coords: np.ndarray,
    num_territories: int,
    min_calls: int,
    max_calls: int,
    resource_locs: Optional[List[dict]] = None,
):
    """
    Returns:
      assign: np.ndarray (len n) territory index for each call
      centers: list of dicts {lat,lng,name}
      medoid_indices: np.ndarray or None
    """
    n = len(coords)
    k = int(num_territories)
    if k <= 0:
        raise HTTPException(400, "numTerritories must be >= 1")
    if n < k:
        # fewer calls than territories: some territories will be empty, but still define them
        pass

    # Helper: balanced global assignment via min-cost flow
    def _assign_min_cost_flow(D: np.ndarray, quotas: list[int], allowed: Optional[List[Optional[List[int]]]] = None,
                              item_coords: Optional[np.ndarray] = None,
                              center_coords: Optional[np.ndarray] = None) -> np.ndarray:
        n_calls, n_centers = D.shape
        # Precompute nearest center distance for each call
        nearest = np.min(D, axis=1)
        start_indices: list[int] = []
        end_indices: list[int] = []
        capacities: list[int] = []
        unit_costs: list[int] = []
        # Node indexing: centers 0..k-1, calls k..k+n-1, source k+n, sink k+n+1
        source = n_centers + n_calls
        sink = source + 1
        num_nodes = sink + 1
        supplies = [0] * num_nodes
        # Source supply is total flow n_calls; sink is -n_calls
        supplies[source] = n_calls
        supplies[sink] = -n_calls
        # source -> center arcs
        for j in range(n_centers):
            start_indices.append(source)
            end_indices.append(j)
            capacities.append(int(quotas[j]))
            unit_costs.append(0)
        # center -> call arcs (capacity 1, cost = shaped distance)
        for j in range(n_centers):
            for i in range(n_calls):
                if allowed is not None and allowed[i] is not None and j not in allowed[i]:
                    continue
                start_indices.append(j)
                end_indices.append(n_centers + i)
                capacities.append(1)
                base = float(D[i, j])
                # Penalize if assigning much farther than this call's nearest center
                penalty = 0.0
                if nearest[i] > 0:
                    ratio = base / nearest[i]
                    if ratio > FLOW_ALPHA:
                        penalty = (ratio - FLOW_ALPHA) * FLOW_GAMMA * nearest[i]
                cost_val = base + penalty
                # Corridor discount: if enabled and coords provided, reduce cost when bearing aligns
                if CORRIDOR_ON and item_coords is not None and center_coords is not None:
                    ilat, ilon = float(item_coords[i, 0]), float(item_coords[i, 1])
                    clat, clon = float(center_coords[j, 0]), float(center_coords[j, 1])
                    br = _bearing_deg(clat, clon, ilat, ilon)  # center -> item
                    # Prefer headings computed per center based on spatial layout
                    headings = _corridor_headings_for_centers(center_coords, item_coords, CORRIDOR_K)[j]
                    diffs = [_angle_diff_deg(br, h) for h in headings]
                    if len(diffs) > 0:
                        mind = min(diffs)
                        sigma = max(1e-6, CORRIDOR_SIGMA_DEG)
                        align = float(np.exp(-0.5 * (mind / sigma) ** 2))
                        discount = CORRIDOR_BETA * align  # fraction of base
                        cost_val = max(1.0, cost_val * (1.0 - discount))
                c = int(max(1, round(cost_val)))
                unit_costs.append(c)
        # call -> sink arcs
        for i in range(n_calls):
            start_indices.append(n_centers + i)
            end_indices.append(sink)
            capacities.append(1)
            unit_costs.append(0)

        if not _HAS_FLOW:
            raise RuntimeError("Min-cost flow not available on this server")
        mcf = _new_mcf()
        # Compatibility shims for legacy vs new API
        def _add_arc(m, u, v, cap, cost):
            if hasattr(m, 'AddArcWithCapacityAndUnitCost'):
                m.AddArcWithCapacityAndUnitCost(u, v, int(cap), int(cost))
            else:
                m.add_arc_with_capacity_and_unit_cost(u, v, int(cap), int(cost))
        def _set_supply(m, node, sup):
            if hasattr(m, 'SetNodeSupply'):
                m.SetNodeSupply(node, int(sup))
            else:
                m.set_node_supply(node, int(sup))
        def _solve(m):
            if hasattr(m, 'Solve'):
                return m.Solve()
            return m.solve()
        def _num_arcs(m):
            if hasattr(m, 'NumArcs'):
                return m.NumArcs()
            return m.num_arcs()
        def _tail(m, a):
            if hasattr(m, 'Tail'):
                return m.Tail(a)
            return m.tail(a)
        def _head(m, a):
            if hasattr(m, 'Head'):
                return m.Head(a)
            return m.head(a)
        def _flow(m, a):
            if hasattr(m, 'Flow'):
                return m.Flow(a)
            return m.flow(a)

        for u, v, cap, cost in zip(start_indices, end_indices, capacities, unit_costs):
            _add_arc(mcf, u, v, cap, cost)
        for node, sup in enumerate(supplies):
            _set_supply(mcf, node, sup)

        status = _solve(mcf)
        optimal_code = getattr(mcf, 'OPTIMAL', 0)
        if status != optimal_code:
            raise RuntimeError(f"Min-cost flow failed with status {status}")
        # Extract assignment: for arcs center->call with flow 1
        assign = -np.ones(n_calls, dtype=int)
        num_arcs = _num_arcs(mcf)
        for a in range(num_arcs):
            u = _tail(mcf, a)
            v = _head(mcf, a)
            if 0 <= u < n_centers and n_centers <= v < n_centers + n_calls:
                if _flow(mcf, a) > 0:
                    call_idx = v - n_centers
                    center_idx = u
                    assign[call_idx] = center_idx
        if np.any(assign < 0):
            raise RuntimeError("Incomplete assignment from flow")
        return assign

    if resource_locs and len(resource_locs) > 0:
        # Fixed centers provided: balanced global assignment via min-cost flow
        centers_arr = np.array([[rl["lat"], rl["lng"]] for rl in resource_locs], dtype=float)
        # Use OSRM durations for network-aware assignment
        D_call_to_center = osrm_matrix_between_duration(coords, centers_arr)
        k_centers = len(centers_arr)
        # Build quotas as equal-share
        base = n // max(1, k_centers)
        rem = n % max(1, k_centers)
        quotas = [base + (1 if j < rem else 0) for j in range(k_centers)]

        if USE_CELL_FLOW:
            # Map calls to planning cells and collect membership
            cell_ids = _hex_ids_for_coords(coords, H3_PLAN_RES)
            cell_to_calls: dict[str, List[int]] = {}
            for i, cid in enumerate(cell_ids):
                key = str(cid) if cid is not None else f"none:{i}"
                cell_to_calls.setdefault(key, []).append(i)
            cell_keys = list(cell_to_calls.keys())

            # Build cell centroids
            centroids: List[Tuple[float, float]] = []
            for key in cell_keys:
                if key.startswith("none:"):
                    idx = int(key.split(":", 1)[1])
                    centroids.append((float(coords[idx, 0]), float(coords[idx, 1])))
                else:
                    try:
                        c = h3.h3_to_geo(key) if _H3_AVAILABLE else None
                    except Exception:
                        c = None
                    if c:
                        centroids.append((float(c[0]), float(c[1])))
                    else:
                        pts = coords[cell_to_calls[key]]
                        centroids.append((float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))
            centroids_arr = np.array(centroids, dtype=float) if centroids else np.zeros((0, 2))

            # Distances from cells to centers
            D_cell_to_center = osrm_matrix_between(centroids_arr, centers_arr) if len(centroids_arr) else np.zeros((0, k_centers))

            # Allowed centers per cell: top-K
            allowed_cells: List[List[int]] = []
            for i in range(D_cell_to_center.shape[0]):
                order = list(np.argsort(D_cell_to_center[i]).astype(int))
                allowed_cells.append(order[:max(1, FLOW_TOPK)])

            # Assign entire cells atomically to centers (cell-level, indivisible)
            demands = [len(cell_to_calls[k]) for k in cell_keys]

            def assign_cells_greedy(D_cells: np.ndarray, demands: List[int], quotas: List[int], allowed_cells: List[List[int]]):
                """Greedy atomic assignment of cells to centers.
                For each cell, assign the whole cell to the nearest allowed center with sufficient remaining quota.
                If no center has full remaining quota, assign to the center with the most remaining capacity.
                Returns: list of assigned center indices per cell.
                """
                m, kc = D_cells.shape
                cap = [int(q) for q in quotas]
                assign_cells = [-1] * m
                # Order cells by proximity to any center (closest first)
                min_d = np.min(D_cells, axis=1)
                order = list(np.argsort(min_d).astype(int))
                for idx in order:
                    dem = int(demands[idx])
                    centers_order = list(np.argsort(D_cells[idx]).astype(int))
                    # respect allowed list if present
                    allowed_list = allowed_cells[idx] if allowed_cells and idx < len(allowed_cells) else None
                    if allowed_list is not None:
                        centers_order = [c for c in centers_order if c in allowed_list]
                        if not centers_order:
                            centers_order = list(np.argsort(D_cells[idx]).astype(int))
                    chosen = -1
                    for c in centers_order:
                        if cap[c] >= dem:
                            chosen = c
                            break
                    if chosen == -1:
                        # pick center with maximum remaining capacity (as fallback)
                        chosen = int(np.argmax(cap))
                    assign_cells[idx] = chosen
                    cap[chosen] -= dem
                return assign_cells

            # Run atomic cell assignment
            cell_assignments = assign_cells_greedy(D_cell_to_center, demands, quotas, allowed_cells)
            # Map cell assignments to per-call assignments
            assign = -np.ones(n, dtype=int)
            cell_winner: dict[str, int] = {}
            for i, key in enumerate(cell_keys):
                dem = demands[i]
                if dem == 0:
                    continue
                center_idx = int(cell_assignments[i])
                cell_winner[key] = center_idx
                for ci in cell_to_calls[key]:
                    assign[ci] = center_idx

                # Optional light border smoothing to reduce interleaving between neighbors
                if SMOOTH_ITERS > 0 and len(cell_keys) > 1:
                    # Build neighbor map of cells; use H3 ring if available else 6 nearest by centroid
                    centroids_map = {cell_keys[i]: tuple(centroids_arr[i]) for i in range(len(cell_keys))}
                    def _neighbors_h3(c: str) -> List[str]:
                        if not _H3_AVAILABLE or c.startswith("none:"):
                            return []
                        try:
                            ns = list(h3.k_ring(c, 1))
                        except Exception:
                            try:
                                ns = list(h3.grid_disk(c, 1))
                            except Exception:
                                ns = []
                        return [x for x in ns if x != c and x in cell_to_calls]
                    cent_mat = np.array([centroids_map[k] for k in cell_keys]) if len(cell_keys) else np.zeros((0, 2))
                    def _neighbors_fallback(c: str) -> List[str]:
                        if not len(cell_keys):
                            return []
                        i = cell_keys.index(c)
                        d = np.sum((cent_mat - cent_mat[i])**2, axis=1)
                        order = np.argsort(d)
                        ids: List[str] = []
                        for oi in order:
                            if int(oi) == i:
                                continue
                            ids.append(cell_keys[int(oi)])
                            if len(ids) >= 6:
                                break
                        return ids
                    def neighbors(c: str) -> List[str]:
                        nh = _neighbors_h3(c)
                        return nh if nh else _neighbors_fallback(c)

                    # Current per-center counts in terms of calls
                    counts = np.zeros(k_centers, dtype=int)
                    for key, w in cell_winner.items():
                        counts[w] += len(cell_to_calls[key])
                    target = np.array(quotas, dtype=int)

                    for _ in range(SMOOTH_ITERS):
                        changed = False
                        # Consider border cells first (neighbors with different winner)
                        border = [key for key in cell_keys if any(cell_winner.get(nk, cell_winner[key]) != cell_winner[key] for nk in neighbors(key))]
                        # Prefer flipping small cells first
                        border.sort(key=lambda k: len(cell_to_calls[k]))
                        for key in border:
                            cur = cell_winner[key]
                            dem = len(cell_to_calls[key])
                            # Take neighbor majority
                            tally: dict[int, int] = {}
                            for nk in neighbors(key):
                                w = cell_winner.get(nk, cur)
                                tally[w] = tally.get(w, 0) + 1
                            if not tally:
                                continue
                            cand = max(tally.items(), key=lambda x: x[1])[0]
                            if cand == cur:
                                continue
                            # Respect quotas, but allow a small slack on receiver to permit helpful flips
                            # donor must stay >= target; receiver may exceed target by up to 5% of its quota
                            slack_frac = 0.05
                            max_allowed_receiver = target[cand] + max(1, int(slack_frac * max(1, target[cand])))
                            if counts[cur] - dem < target[cur]:
                                continue
                            if counts[cand] + dem > max_allowed_receiver:
                                continue
                            # Flip cell to candidate
                            cell_winner[key] = cand
                            counts[cur] -= dem
                            counts[cand] += dem
                            for ci in cell_to_calls[key]:
                                assign[ci] = cand
                            changed = True
                        if not changed:
                            break
        else:
            # Per-call flow
            if not _HAS_FLOW:
                if FLOW_STRICT:
                    raise HTTPException(503, "Flow planning unavailable on this server")
                # Fallback to previous balanced assignment if strict is off
                hex_ids = _hex_ids_for_coords(coords, H3_RES)
                terr_min = int(np.min(quotas))
                terr_max = int(np.max(quotas))
                try:
                    assign = balanced_assign(D_call_to_center, terr_min, terr_max, hex_ids=hex_ids, lam=H3_LAMBDA)
                except Exception:
                    assign = np.argmin(D_call_to_center, axis=1)
            else:
                assign = _assign_min_cost_flow(
                    D_call_to_center, quotas,
                    item_coords=coords,
                    center_coords=centers_arr
                )

        centers = [
            {"lat": float(centers_arr[i, 0]), "lng": float(centers_arr[i, 1]), "name": resource_locs[i].get("name", f"Resource {i+1}")}
            for i in range(len(centers_arr))
        ]
        medoid_indices = None
    else:
        # Suggest bases: k-medoids on call-to-call road durations, then Voronoi assignment
        D_calls = osrm_table_batch_duration(coords)
        medoid_indices = k_medoids(D_calls, k, iters=25, random_state=42)
        D_call_to_center = D_calls[:, medoid_indices]
        try:
            hex_ids = _hex_ids_for_coords(coords, H3_RES)
            # Balance territory sizes toward equal share regardless of day caps
            k_centers = D_call_to_center.shape[1]
            terr_min = max(0, int(np.floor(n / max(1, k_centers))))
            terr_max = int(np.ceil(n / max(1, k_centers)))
            assign = balanced_assign(D_call_to_center, terr_min, terr_max, hex_ids=hex_ids, lam=H3_LAMBDA)
        except Exception:
            assign = np.argmin(D_call_to_center, axis=1)
        centers = [
            {"lat": float(coords[idx, 0]), "lng": float(coords[idx, 1]), "name": f"Suggested Base {i+1}"}
            for i, idx in enumerate(medoid_indices)
        ]
    return assign, centers, medoid_indices
# ----------------------------
_geocode_cache = {}
_GEOCODE_EXECUTOR: ThreadPoolExecutor | None = None


def _postcode_key(pc: str) -> str:
    return str(pc or "").strip().replace(" ", "").upper()


_LAT_CANDIDATES = [
    "latitude",
    "lat",
    "latitudedegrees",
    "latdd",
    "latdeg",
    "y",
]
_LON_CANDIDATES = [
    "longitude",
    "lng",
    "lon",
    "long",
    "longitudedegrees",
    "longdd",
    "longdeg",
    "x",
]


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


async def _resolve_coordinates_with_fallback(
    df: pd.DataFrame,
    postcode_col: str,
) -> tuple[list[float], list[float], list[int]]:
    """Return latitude/longitude arrays using existing columns where possible and geocode the rest."""

    n = len(df)
    if n == 0:
        return [], [], []

    lat_col = _find_first_column(df, _LAT_CANDIDATES)
    lon_col = _find_first_column(df, _LON_CANDIDATES)

    if lat_col:
        lat_series = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        lat_series = pd.Series([np.nan] * n, index=df.index, dtype="float64")
    if lon_col:
        lon_series = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        lon_series = pd.Series([np.nan] * n, index=df.index, dtype="float64")

    latitudes = lat_series.astype(float).to_numpy(copy=True)
    longitudes = lon_series.astype(float).to_numpy(copy=True)

    needs_geocode_mask = np.isnan(latitudes) | np.isnan(longitudes)
    if needs_geocode_mask.any():
        idxs = [int(i) for i, flag in enumerate(needs_geocode_mask) if flag]

        def _clean_postcode(value: Any) -> str:
            if value is None:
                return ""
            s = str(value).strip()
            if not s:
                return ""
            if s.lower() in {"nan", "none", "null"}:
                return ""
            return s

        subset_postcodes = [_clean_postcode(df.iloc[i][postcode_col]) for i in idxs]
        geocoded_subset = await geocode_postcodes_bulk(subset_postcodes)
        for target_idx, (lat, lng) in zip(idxs, geocoded_subset):
            latitudes[target_idx] = lat
            longitudes[target_idx] = lng

    failed_indices = [i for i, (lat, lng) in enumerate(zip(latitudes, longitudes)) if np.isnan(lat) or np.isnan(lng)]
    return latitudes.tolist(), longitudes.tolist(), failed_indices


def _get_geocode_executor() -> ThreadPoolExecutor:
    global _GEOCODE_EXECUTOR
    if _GEOCODE_EXECUTOR is None:
        workers = max(1, min(10, int(os.getenv("GEOCODE_WORKERS", "4"))))
        _GEOCODE_EXECUTOR = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="geocode")
    return _GEOCODE_EXECUTOR


async def geocode_postcodes_bulk(postcodes: List[str]) -> List[Tuple[float, float]]:
    """Resolve many postcodes concurrently with caching, preserving order."""
    loop = asyncio.get_running_loop()
    results: Dict[int, Tuple[float, float]] = {}
    idx_by_key: Dict[str, Dict[str, Any]] = {}
    for idx, raw in enumerate(postcodes):
        key = _postcode_key(raw)
        if not key:
            results[idx] = (np.nan, np.nan)
            continue
        if key in _geocode_cache:
            results[idx] = _geocode_cache[key]
            continue
        entry = idx_by_key.setdefault(key, {"raw": raw, "idxs": []})
        entry["idxs"].append(idx)
    if idx_by_key:
        executor = _get_geocode_executor()
        pending = list(idx_by_key.items())
        tasks = [loop.run_in_executor(executor, geocode_postcode_norm, info["raw"]) for _, info in pending]
        geocoded = await asyncio.gather(*tasks, return_exceptions=True)
        for (key, info), val in zip(pending, geocoded):
            if isinstance(val, Exception):
                latlng = (np.nan, np.nan)
            else:
                latlng = val
            for idx in info["idxs"]:
                results[idx] = latlng
    return [results.get(i, (np.nan, np.nan)) for i in range(len(postcodes))]


def geocode_postcode_norm(pc: str):
    """
    Geocode a postcode robustly:
    - Normalize by stripping spaces and uppercasing
    - Try original, spaced (for formats like Irish Eircode), and with ", Ireland"
    - On API errors (e.g., HTTP 400), fall back gracefully and mark as failed
    """
    orig = str(pc or "").strip()
    key = _postcode_key(orig)
    if not key:
        return (np.nan, np.nan)
    if key in _geocode_cache:
        return _geocode_cache[key]

    candidates = []
    # Try raw string first (may be a full "eircode, town" etc.)
    if orig:
        candidates.append(orig)
    # Normalized (no spaces, uppercased)
    candidates.append(key)
    # If alphanumeric of length 6-8, try inserting a space after 3 chars (Eircode style)
    if key.isalnum() and 6 <= len(key) <= 8:
        spaced = f"{key[:3]} {key[3:]}"
        if spaced not in candidates:
            candidates.append(spaced)
    # Add country hint variants to improve UK/IE disambiguation
    more = []
    for c in list(candidates):
        for hint in (", United Kingdom", ", UK", ", Ireland"):
            v = f"{c}{hint}"
            if v not in candidates:
                more.append(v)
    candidates.extend(more)

    if gmaps is None:
        raise RuntimeError("Google Maps client not initialised")

    for q in candidates:
        try:
            res = gmaps.geocode(q)
            if res and isinstance(res, list) and len(res) > 0 and "geometry" in res[0]:
                loc = res[0]["geometry"]["location"]
                latlng = (loc["lat"], loc["lng"])
                _geocode_cache[key] = latlng
                return latlng
        except Exception:
            # Ignore API errors and try next candidate
            continue

    _geocode_cache[key] = (np.nan, np.nan)
    return (np.nan, np.nan)


# ----------------------------
# API: /api/territory-plan
# ----------------------------
@app.post("/api/territory-plan")
async def territory_plan(
    request: Request,
    file: UploadFile = File(...),
    numTerritories: int = Form(...),
    minCalls: int = Form(...),
    maxCalls: int = Form(...),
    resourceLocations: Optional[str] = Form(None),  # JSON string: list of {lat,lng,name?} or {postcode,name?}
    groupCol: Optional[str] = Form(None),
    territoryTimeLimitSec: Optional[int] = Form(20),   # reserved; not used now
    dayTimeLimitSec: Optional[int] = Form(30),
    resourcesFile: Optional[UploadFile] = File(None),   # Optional Excel containing resource centers
    projectName: Optional[str] = Form(None)
):
    # Read Excel
    buf = io.BytesIO(await file.read())
    try:
        df = pd.read_excel(buf, engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid Excel")

    # Normalise columns
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )

    postcode_variants = [
        "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
        "zip","zipcode","zip_code","zip code","zip-code"
    ]
    postcode_variants_norm = [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in postcode_variants]

    pc = next((c for c in df.columns if c in postcode_variants_norm), None)
    if not pc:
        raise HTTPException(400, "No postcode column found. Please add a postcode column.")
    print(f"[DEBUG] Using postcode column: {pc}")
    print(f"[DEBUG] First 5 postcodes: {df[pc].head().tolist()}")

    latitudes, longitudes, failed_indices = await _resolve_coordinates_with_fallback(df, pc)
    df["Latitude"] = latitudes
    df["Longitude"] = longitudes
    failed_rows = df.iloc[failed_indices].copy() if failed_indices else []
    # Build geo with valid coordinates
    geo = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if geo.empty:
        raise HTTPException(400, "No valid coordinates after geocoding.")

    # Select group column if provided
    group_col = None
    if groupCol and groupCol.lower() in [c.lower() for c in geo.columns]:
        group_col = [c for c in geo.columns if c.lower() == groupCol.lower()][0]

    # Parse resource locations
    resources = None
    resources_dbg = {"expected": 0, "used": 0, "skipped": 0, "skips": []}
    if resourcesFile is not None:
        try:
            rbuf = io.BytesIO(await resourcesFile.read())
            rdf = pd.read_excel(rbuf, engine="openpyxl").convert_dtypes()
            # Normalise
            rcols = (
                rdf.columns.str.strip().str.lower()
                .str.replace(" ", "", regex=False)
                .str.replace("_", "", regex=False)
                .str.replace("-", "", regex=False)
            )
            rdf.columns = rcols
            name_col = next((c for c in rdf.columns if c in ("name","resourcename","site","depot","base")), None)
            lat_col = next((c for c in rdf.columns if c in ("lat","latitude")), None)
            lng_col = next((c for c in rdf.columns if c in ("lng","lon","long","longitude")), None)
            # Accept a broad set of postcode header variants (normalized lowercased & no spaces/underscores/hyphens)
            pc_variants = [
                "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
                "zip","zipcode","zip_code","zip code","zip-code","eircode","eircod","targetpostcode","target_postcode","target post code"
            ]
            pc_variants_norm = [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in pc_variants]
            pc_col = next((c for c in rdf.columns if c in pc_variants_norm), None)
            if name_col is None:
                name_col = "name"  # default placeholder
                if name_col not in rdf.columns:
                    rdf[name_col] = [f"Resource {i+1}" for i in range(len(rdf))]
            resources = []
            resources_dbg["expected"] = int(len(rdf))
            for i, row in rdf.iterrows():
                name = str(row.get(name_col, f"Resource {i+1}") or f"Resource {i+1}")
                lat = row.get(lat_col) if lat_col else None
                lng = row.get(lng_col) if lng_col else None
                if (lat is not None) and (lng is not None):
                    try:
                        resources.append({"lat": float(lat), "lng": float(lng), "name": name})
                        resources_dbg["used"] += 1
                        continue
                    except Exception:
                        pass
                pc = str(row.get(pc_col, "")).strip() if pc_col else ""
                if pc:
                    latlng = geocode_postcode_norm(pc)
                    if not (np.isnan(latlng[0]) or np.isnan(latlng[1])):
                        resources.append({"lat": float(latlng[0]), "lng": float(latlng[1]), "name": name})
                        resources_dbg["used"] += 1
                    else:
                        resources_dbg["skipped"] += 1; resources_dbg["skips"].append(f"idx {i}: geocode failed for '{pc}'")
                else:
                    resources_dbg["skipped"] += 1; resources_dbg["skips"].append(f"idx {i}: no lat/lng or postcode")
            if len(resources) == 0:
                resources = None
        except Exception as e:
            resources = None
            resources_dbg["skips"].append(f"resourcesFile error: {str(e)}")
    elif resourceLocations:
        try:
            raw_resources = json.loads(resourceLocations)
            if isinstance(raw_resources, list):
                resources = []
                resources_dbg["expected"] = len(raw_resources)
                for i, r in enumerate(raw_resources):
                    if not isinstance(r, dict):
                        resources_dbg["skipped"] += 1; resources_dbg["skips"].append(f"idx {i}: not a dict")
                        continue
                    name = r.get("name", f"Resource {i+1}")
                    lat = r.get("lat"); lng = r.get("lng")
                    pc = r.get("postcode") or r.get("post_code") or r.get("post code")
                    if (lat is not None) and (lng is not None):
                        # Use provided coordinates
                        try:
                            latf = float(lat); lngf = float(lng)
                            resources.append({"lat": latf, "lng": lngf, "name": name})
                            resources_dbg["used"] += 1
                        except Exception:
                            resources_dbg["skipped"] += 1; resources_dbg["skips"].append(f"idx {i}: invalid lat/lng")
                            continue
                    elif pc and str(pc).strip():
                        # Geocode postcode into lat/lng
                        latlng = geocode_postcode_norm(str(pc))
                        if not (np.isnan(latlng[0]) or np.isnan(latlng[1])):
                            resources.append({"lat": float(latlng[0]), "lng": float(latlng[1]), "name": name})
                            resources_dbg["used"] += 1
                        else:
                            resources_dbg["skipped"] += 1; resources_dbg["skips"].append(f"idx {i}: geocode failed for '{pc}'")
                    else:
                        # Blank row → ignore
                        resources_dbg["skipped"] += 1; resources_dbg["skips"].append(f"idx {i}: empty resource")
                        continue
                if len(resources) == 0:
                    resources = None
            else:
                resources = None
        except Exception:
            resources = None

    # Validate min/max
    if minCalls > maxCalls:
        raise HTTPException(400, "minCalls must be <= maxCalls")

    # Prepare coords array
    coords = geo[["Latitude", "Longitude"]].astype(float).to_numpy()

    # Territory planning (with or without group col)
    all_labels = ["" for _ in range(len(geo))]
    all_centers = []
    total_territories = 0
    assignment_global = np.full(len(geo), -1, dtype=int)  # territory index per row (global)

    # If resources provided, plan across the whole dataset once using those centers
    if resources is not None:
        # When a resources file/list is provided, use exactly that many territories
        # and ignore the Number of Territories input.
        assign, centers, _ = plan_territories(
            coords, int(len(resources)), minCalls, maxCalls, resources
        )
        for i in range(len(geo)):
            t = int(assign[i])
            name = centers[t].get("name", f"Territory {t+1}")
            all_labels[i] = str(name)
            assignment_global[i] = t
        all_centers = centers
        total_territories = len(centers)
    elif group_col:
        # plan per group (keeps user logical splits)
        terr_counter_offset = 0
        for group_val in geo[group_col].unique():
            # Handle NA/None group values explicitly to avoid ambiguous NA
            if pd.isna(group_val):
                mask = geo[group_col].isna().to_numpy()
                group_note = "group <NA>"
            else:
                mask = geo[group_col].eq(group_val).fillna(False).to_numpy(dtype=bool)
                group_note = f"group {group_val}"
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            sub_coords = coords[idxs]

            assign, centers, _ = plan_territories(
                sub_coords, numTerritories, minCalls, maxCalls, None
            )
            # Map to global indices
            for t_local in range(max(assign) + 1 if len(assign) else 0 + 1):
                pass  # (no-op; we only need per-call assignment below)

            # Label calls + collect centers
            for rel_i, abs_i in enumerate(idxs):
                t = int(assign[rel_i])
                label = f"Territory {t + 1}"
                all_labels[abs_i] = f"{label}"
                assignment_global[abs_i] = t + terr_counter_offset
            # Shift center id space for global map
            for c in centers:
                all_centers.append(c)
            total_territories += (np.max(assign) + 1) if len(assign) else numTerritories
            terr_counter_offset += numTerritories
    else:
        assign, centers, _ = plan_territories(
            coords, numTerritories, minCalls, maxCalls, None
        )
        for i in range(len(geo)):
            all_labels[i] = f"Territory {int(assign[i]) + 1}"
            assignment_global[i] = int(assign[i])
        all_centers = centers
        total_territories = numTerritories

    geo["Territory"] = all_labels

    # Remove legacy duplicates
    drop_cols = [c for c in geo.columns if c.lower() in ("group", "day", "territory") and c not in ("Territory", "Day")]
    geo = geo.drop(columns=drop_cols, errors="ignore")

    # Per-territory day clustering
    geo["Day"] = ""
    # Build a mapping: territory index -> list of global indices
    terr_ids = np.unique(assignment_global[assignment_global >= 0])
    for t in terr_ids:
        terr_mask = (assignment_global == t)
        terr_idxs = np.where(terr_mask)[0]
        if len(terr_idxs) == 0:
            continue
        terr_coords = coords[terr_idxs]
        # Choose depot: the center we created for that territory index (wrap if group had offsets)
        # If resources were provided and fewer than territories, we still use the center list in order.
        center_idx = int(t) if (int(t) < len(all_centers)) else (len(all_centers) - 1)
        depot_lat_lng = (all_centers[center_idx]["lat"], all_centers[center_idx]["lng"])

        routes = cvrp_days_for_territory(
            terr_coords, depot_lat_lng, minCalls, maxCalls, time_sec=int(dayTimeLimitSec or 30)
        )
        # Label days
        for day_i, route in enumerate(routes, start=1):
            for rel in route:
                abs_idx = terr_idxs[rel]
                geo.at[abs_idx, "Day"] = f"Day {day_i}"

    # Normalise Day values as "Day X"
    if "Day" in geo.columns:
        def _norm_day(x):
            if isinstance(x, (int, float)) and not pd.isna(x):
                return f"Day {int(x)}"
            s = str(x)
            if not s:
                return s
            return s if s.lower().startswith("day ") else f"Day {s}"
        geo["Day"] = geo["Day"].apply(_norm_day)
        # Also provide DayNumber and CallsPerDay for easier UI consumption
        def _day_num(v):
            try:
                import re as _re
                m = _re.search(r"(\d+)", str(v))
                return int(m.group(1)) if m else None
            except Exception:
                return None
        geo["DayNumber"] = geo["Day"].map(_day_num)
        if "Territory" in geo.columns:
            cts = geo.groupby(["Territory","Day"]).size().rename("CallsPerDay").reset_index()
            geo = geo.merge(cts, on=["Territory","Day"], how="left")
        else:
            cts = geo.groupby(["Day"]).size().rename("CallsPerDay").reset_index()
            geo = geo.merge(cts, on=["Day"], how="left")

    # Build Excel-style rows: [headers, ...rows]
    keep_cols = [c for c in geo.columns if c not in drop_cols]
    outCols = [c for c in keep_cols if c not in ("Latitude", "Longitude", "Territory", "Day", "DayNumber", "CallsPerDay")]
    excelCols = ["Territory", "Day", "DayNumber", "CallsPerDay", "Latitude", "Longitude"] + outCols
    geo_excel = geo[excelCols].astype("object").where(pd.notna(geo[excelCols]), "").astype(str)
    rows = [excelCols] + geo_excel.values.tolist()

    meta = {
        "row_count": max(0, len(rows) - 1),
        "failed_geocodes": len(failed_rows),
        "num_territories": int(total_territories),
        "used_resources": bool(resources is not None),
    }
    project_id = _maybe_save_project(
        request,
        mode="territory-plan",
        rows=rows,
        meta=meta,
        explicit_name=projectName,
    )

    return JSONResponse({
        "rows": rows,
        "num_territories": int(total_territories),
        "suggested_locations": all_centers,  # list of {lat,lng,name}
        "resource_info": resources_dbg,
        "used_resources": bool(resources is not None),
        "message": f"Assigned {len(geo)} calls to {int(total_territories)} territories and clustered into days.",
        "project_id": project_id,
    })


# ----------------------------
# API: /api/cluster (only day clustering on an existing group/territory col)
# ----------------------------
@app.post("/api/cluster")
async def cluster_only(
    request: Request,
    file: UploadFile = File(...),
    minCalls: int = Form(...),
    maxCalls: int = Form(...),
    groupCol: Optional[str] = Form(None),
    splitByGroup: Optional[str] = Form(None),
    dayTimeLimitSec: Optional[int] = Form(30),
    projectName: Optional[str] = Form(None)
):
    buf = io.BytesIO(await file.read())
    try:
        df = pd.read_excel(buf, engine="openpyxl").convert_dtypes()
    except Exception:
        traceback.print_exc()
        raise HTTPException(400, "Invalid Excel")

    # Normalise columns
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )

    postcode_variants = [
        "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
        "zip","zipcode","zip_code","zip code","zip-code"
    ]
    postcode_variants_norm = [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in postcode_variants]
    pc = next((c for c in df.columns if c in postcode_variants_norm), None)
    if not pc:
        raise HTTPException(400, "No postcode column found. Please add a postcode column.")

    latitudes, longitudes, failed_indices = await _resolve_coordinates_with_fallback(df, pc)
    df["Latitude"] = latitudes
    df["Longitude"] = longitudes
    failed_rows = df.iloc[failed_indices].copy() if failed_indices else []
    geo = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if geo.empty:
        raise HTTPException(400, "No valid coordinates after geocoding.")

    service_seconds_all, service_col = _infer_service_seconds(df)
    df["_service_seconds"] = service_seconds_all
    geo = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if geo.empty:
        raise HTTPException(400, "No valid coordinates after geocoding.")

    service_seconds = geo["_service_seconds"].to_numpy(dtype=float)

    # Resolve grouping: cluster-only ignores territories by default.
    # Only split by group if the caller explicitly asks via splitByGroup=1 and provides a valid groupCol.
    group_col = None
    _respect_group = str(splitByGroup or "").strip().lower() in ("1", "true", "yes", "on")
    if _respect_group and groupCol and groupCol.lower() in [c.lower() for c in geo.columns]:
        group_col = [c for c in geo.columns if c.lower() == groupCol.lower()][0]

    minCalls = max(0, int(minCalls))
    maxCalls = int(maxCalls)
    if maxCalls <= 0:
        maxCalls = len(geo)
    if maxCalls < minCalls:
        raise HTTPException(400, "maxCalls must be >= minCalls")

    coords = geo[["Latitude", "Longitude"]].astype(float).to_numpy()
    geo["Day"] = ""

    solver_limit = int(dayTimeLimitSec or CLUSTER_SOLVER_TIME_SEC)
    solver_counts = {"ortools": 0, "fallback": 0, "sa": 0}
    target_minutes_acc: List[float] = []
    fallback_notes: List[str] = []
    sa_polished_groups = 0
    used_precluster = False

    def _choose_depot(points: np.ndarray) -> Tuple[float, float]:
        if len(points) == 0:
            return (0.0, 0.0)
        if len(points) == 1:
            return (float(points[0, 0]), float(points[0, 1]))
        try:
            if len(points) <= CLUSTER_OSRM_MAX_N:
                D = osrm_table_batch(points)
                med = int(k_medoids(D, 1, iters=10, random_state=42)[0])
                return (float(points[med, 0]), float(points[med, 1]))
        except Exception:
            pass
        return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1])))

    def _cluster_indices(idxs: np.ndarray, note_key: str) -> None:
        nonlocal solver_counts, sa_polished_groups
        if len(idxs) == 0:
            return
        terr_coords = coords[idxs]
        svc_local = service_seconds[idxs]

        # Precluster into geographically compact day-sized groups when enabled
        if int(CLUSTER_PRECLUSTER_FIRST) and int(maxCalls) > 0 and len(terr_coords) > 1:
            nonlocal used_precluster
            try:
                min_ratio = 0.0
                try:
                    if int(maxCalls) > 0:
                        min_ratio = max(0.0, min(1.0, float(minCalls) / float(maxCalls)))
                except Exception:
                    min_ratio = 0.0
                day_groups = _precluster_days_kmedoids(terr_coords, int(maxCalls), float(min_ratio), float(H3_LAMBDA))
                # Optional refinement to pull in outliers while respecting caps
                try:
                    day_groups = _refine_day_groups_compact(
                        terr_coords, day_groups, int(minCalls), int(maxCalls),
                        improve_ratio=float(CLUSTER_REFINE_IMPROVE_RATIO),
                        iters=int(CLUSTER_REFINE_ITERS)
                    )
                except Exception:
                    pass
                day_counter = 1
                for g in day_groups:
                    for rel_idx in g:
                        geo.at[int(idxs[int(rel_idx)]), "Day"] = f"Day {day_counter}"
                    day_counter += 1
                used_precluster = True
                return
            except Exception as exc:
                fallback_notes.append(f"{note_key}: precluster failed ({exc}); falling back to VRP split")
        depot_lat_lng = _choose_depot(terr_coords)
        target_minutes = _estimate_target_minutes(svc_local, maxCalls)
        target_minutes_acc.append(float(target_minutes))
        duration_matrix: Optional[np.ndarray] = None
        combined_coords = np.vstack([terr_coords, np.array([depot_lat_lng])])
        if CLUSTER_SA_POLISH and len(terr_coords) <= CLUSTER_SA_MAX_N:
            try:
                duration_matrix = osrm_table_batch_duration(combined_coords)
            except Exception as exc:
                fallback_notes.append(f"{note_key}: SA duration fallback ({exc})")
                duration_matrix = None
            if duration_matrix is None:
                dist_mat = _haversine_matrix_full(combined_coords)
                speed = max(OSRM_FALLBACK_SPEED_MPS, 0.1)
                duration_matrix = dist_mat / speed
        routes: List[List[int]]
        if len(terr_coords) == 1:
            routes = [[0]]
        elif len(terr_coords) <= CLUSTER_OSRM_MAX_N:
            try:
                routes, _ = _cluster_days_vrp(
                    terr_coords,
                    svc_local,
                    depot_lat_lng,
                    minCalls,
                    maxCalls,
                    solver_limit,
                    target_minutes,
                )
                solver_counts["ortools"] += len([r for r in routes if r])
            except Exception as exc:
                solver_counts["fallback"] += 1
                fallback_notes.append(f"{note_key}: {exc}")
                routes = cvrp_days_for_territory(terr_coords, depot_lat_lng, minCalls, maxCalls)
        else:
            solver_counts["fallback"] += 1
            fallback_notes.append(f"{note_key}: skipped OR-Tools (size>{CLUSTER_OSRM_MAX_N})")
            routes = cvrp_days_for_territory(terr_coords, depot_lat_lng, minCalls, maxCalls)

        if (
            CLUSTER_SA_POLISH
            and len(routes) > 1
            and len(terr_coords) <= CLUSTER_SA_MAX_N
            and duration_matrix is not None
        ):
            polished = _sa_polish_day_routes(
                routes,
                duration_matrix,
                svc_local,
                float(target_minutes),
                int(minCalls),
                int(maxCalls),
                int(CLUSTER_SA_ITERS),
                float(CLUSTER_SA_INIT_TEMP),
                float(CLUSTER_SA_COOLING),
            )
            if polished:
                routes = polished
                solver_counts["sa"] += 1
                sa_polished_groups += 1

        day_counter = 1
        for rel_route in routes:
            if not rel_route:
                continue
            for rel_idx in rel_route:
                geo.at[int(idxs[rel_idx]), "Day"] = f"Day {day_counter}"
            day_counter += 1

    if group_col:
        for group_val in geo[group_col].unique():
            # Build a safe boolean mask that handles pandas NA values explicitly
            if pd.isna(group_val):
                mask = geo[group_col].isna().to_numpy()
                note = "group <NA>"
            else:
                mask = geo[group_col].eq(group_val).fillna(False).to_numpy(dtype=bool)
                note = f"group {group_val}"
            idxs = np.where(mask)[0]
            _cluster_indices(idxs, note)
    else:
        _cluster_indices(np.arange(len(geo)), "all")

    # Derive DayNumber and CallsPerDay
    def _day_num(v):
        try:
            import re as _re
            m = _re.search(r"(\d+)", str(v))
            return int(m.group(1)) if m else None
        except Exception:
            return None

    geo["DayNumber"] = geo["Day"].map(_day_num)
    if group_col:
        cts = geo.groupby([group_col, "Day"]).size().rename("CallsPerDay").reset_index()
        geo = geo.merge(cts, on=[group_col, "Day"], how="left")
    else:
        cts = geo.groupby(["Day"]).size().rename("CallsPerDay").reset_index()
        geo = geo.merge(cts, on=["Day"], how="left")

    # Disambiguate day labels across groups by providing a global unique id and a combined key
    if group_col:
        try:
            # Unique (group, Day) pairs in first-seen order
            uniq_pairs = (
                geo[[group_col, "Day"]]
                .dropna(subset=["Day"])  # keep NA groups; Day must be present
                .drop_duplicates(keep="first")
            )
            # Build a stable key and factorize to 1..K
            key_series = uniq_pairs[group_col].astype(str).fillna("<NA>") + "||" + uniq_pairs["Day"].astype(str)
            codes, _uniques = pd.factorize(key_series, sort=False)
            uniq_pairs = uniq_pairs.assign(_GlobalDayNumber=(codes + 1))
            geo = geo.merge(uniq_pairs.assign(_k=key_series.values)[[group_col, "Day", "_GlobalDayNumber"]], on=[group_col, "Day"], how="left")
            geo.rename(columns={"_GlobalDayNumber": "GlobalDayNumber"}, inplace=True)
            geo["DayKey"] = geo[group_col].astype(str).fillna("<NA>") + " - " + geo["Day"].astype(str)
        except Exception:
            # Fallback: mirror DayNumber
            geo["GlobalDayNumber"] = geo["DayNumber"]
            geo["DayKey"] = geo["Day"].astype(str)
    else:
        geo["GlobalDayNumber"] = geo["DayNumber"]
        geo["DayKey"] = geo["Day"].astype(str)

    days_msg = int(len({d for d in geo["Day"] if d}))

    geo = geo.drop(columns=["_service_seconds"], errors="ignore")

    # Prepare output rows with Day columns
    outCols = [c for c in geo.columns if c not in ("Latitude", "Longitude", "Day", "DayNumber", "GlobalDayNumber", "DayKey", "CallsPerDay")]
    excelCols = ["Day", "DayNumber", "GlobalDayNumber", "DayKey", "CallsPerDay", "Latitude", "Longitude"] + outCols
    geo_excel = geo[excelCols].astype("object").where(pd.notna(geo[excelCols]), "").astype(str)
    rows = [excelCols] + geo_excel.values.tolist()

    meta = {
        "row_count": max(0, len(rows) - 1),
        "days": int(days_msg),
        "grouped": bool(group_col),
        "group_col_used": str(group_col) if group_col else None,
        "split_by_group": bool(group_col),
        "failed_geocodes": len(failed_rows),
        "solver_counts": solver_counts,
        "service_column": service_col,
        "default_service_minutes": CLUSTER_DEFAULT_SERVICE_MIN if service_col is None else None,
        "median_target_minutes": float(np.median(target_minutes_acc)) if target_minutes_acc else None,
    }
    if used_precluster:
        meta["precluster_used"] = True
    if sa_polished_groups:
        meta["sa_polished_groups"] = int(sa_polished_groups)
    if fallback_notes:
        meta["solver_notes"] = fallback_notes[:20]
    project_id = _maybe_save_project(
        request,
        mode="cluster",
        rows=rows,
        meta=meta,
        explicit_name=projectName,
    )

    return JSONResponse({
        "rows": rows,
        "message": f"Clustered {len(geo)} calls into {int(days_msg)} days.",
        "project_id": project_id,
    })


# ----------------------------
# API: /api/vehicle-route (multi-day VRP with resource windows)
# ----------------------------
async def _vehicle_route_run_from_bytes(
    *,
    calls_bytes: bytes,
    resources_bytes: bytes,
    params: Dict[str, Any],
    headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    def _param(name: str, fallback: Optional[str] = None) -> Optional[str]:
        raw = params.get(name, fallback)
        if raw is None:
            return None
        if isinstance(raw, str):
            val = raw.strip()
            return val or None
        return str(raw)

    def _param_int(name: str, fallback: Optional[int] = None) -> Optional[int]:
        val = params.get(name)
        if val is None:
            return fallback
        try:
            return int(str(val).strip())
        except Exception:
            return fallback

    def _assign_flow(D: np.ndarray, quotas: List[int], allowed: List[Optional[List[int]]]) -> np.ndarray:
        n, k = D.shape
        if not _HAS_FLOW:
            assign = -np.ones(n, dtype=int)
            cap = quotas[:]
            order_calls = list(range(n))
            order_calls.sort(key=lambda i: float(np.min(D[i])))
            for i in order_calls:
                order_centers = list(np.argsort(D[i]).astype(int))
                for j in order_centers:
                    if allowed[i] is not None and j not in allowed[i]:
                        continue
                    if cap[j] > 0:
                        assign[i] = j
                        cap[j] -= 1
                        break
            for i in range(n):
                if assign[i] == -1:
                    j = int(np.argmin([
                        D[i, jj] if (allowed[i] is None or jj in allowed[i]) else 1e18
                        for jj in range(k)
                    ]))
                    assign[i] = j
            return assign

        def _assign_min_cost_flow_vrp(D: np.ndarray, quotas: list[int], allowed: List[Optional[List[int]]]) -> np.ndarray:
            n_calls, n_centers = D.shape
            nearest = np.min(D, axis=1)
            start_indices: list[int] = []
            end_indices: list[int] = []
            capacities: list[int] = []
            unit_costs: list[int] = []
            source = n_centers + n_calls
            sink = source + 1
            num_nodes = sink + 1
            supplies = [0] * num_nodes
            supplies[source] = n_calls
            supplies[sink] = -n_calls
            for j in range(n_centers):
                start_indices.append(source)
                end_indices.append(j)
                capacities.append(int(quotas[j]))
                unit_costs.append(0)
            for j in range(n_centers):
                for i in range(n_calls):
                    if allowed[i] is not None and j not in allowed[i]:
                        continue
                    start_indices.append(j)
                    end_indices.append(n_centers + i)
                    capacities.append(1)
                    base = float(D[i, j])
                    penalty = 0.0
                    if nearest[i] > 0:
                        ratio = base / nearest[i]
                        if ratio > FLOW_ALPHA:
                            penalty = (ratio - FLOW_ALPHA) * FLOW_GAMMA * nearest[i]
                    c = int(max(1, round(base + penalty)))
                    unit_costs.append(c)
            for i in range(n_calls):
                start_indices.append(n_centers + i)
                end_indices.append(sink)
                capacities.append(1)
                unit_costs.append(0)
            mcf = _new_mcf()

            def _add_arc(m, u, v, cap, cost):
                if hasattr(m, "AddArcWithCapacityAndUnitCost"):
                    m.AddArcWithCapacityAndUnitCost(u, v, int(cap), int(cost))
                else:
                    m.add_arc_with_capacity_and_unit_cost(u, v, int(cap), int(cost))

            def _set_supply(m, node, sup):
                if hasattr(m, "SetNodeSupply"):
                    m.SetNodeSupply(node, int(sup))
                else:
                    m.set_node_supply(node, int(sup))

            def _solve(m):
                return m.Solve() if hasattr(m, "Solve") else m.solve()

            def _num_arcs(m):
                return m.NumArcs() if hasattr(m, "NumArcs") else m.num_arcs()

            def _tail(m, a):
                return m.Tail(a) if hasattr(m, "Tail") else m.tail(a)

            def _head(m, a):
                return m.Head(a) if hasattr(m, "Head") else m.head(a)

            def _flow(m, a):
                return m.Flow(a) if hasattr(m, "Flow") else m.flow(a)

            for u, v, cap, cost in zip(start_indices, end_indices, capacities, unit_costs):
                _add_arc(mcf, u, v, cap, cost)
            for node, sup in enumerate(supplies):
                _set_supply(mcf, node, sup)
            status = _solve(mcf)
            optimal_code = getattr(mcf, "OPTIMAL", 0)
            if status != optimal_code:
                raise RuntimeError(f"Flow failed {status}")
            assign = -np.ones(n_calls, dtype=int)
            for a in range(_num_arcs(mcf)):
                u = _tail(mcf, a)
                v = _head(mcf, a)
                if 0 <= u < n_centers and n_centers <= v < n_centers + n_calls:
                    if _flow(mcf, a) > 0:
                        assign[v - n_centers] = u
            return assign

        return _assign_min_cost_flow_vrp(D, quotas, allowed)

    projectName = _param("projectName")

    # Read calls
    try:
        cdf = pd.read_excel(io.BytesIO(calls_bytes), engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid calls Excel")
    cdf.columns = (
        cdf.columns.str.strip().str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    pc_variants = [
        "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
        "zip","zipcode","zip_code","zip code","zip-code","eircode"
    ]
    norm = {c.lower(): c for c in cdf.columns}

    def _norm_key(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        return s.replace(" ", "").replace("_", "").replace("-", "").lower()

    call_pc = _norm_key(_param("callPostcodeCol") or _param("callsPostcodeCol"))
    if not call_pc:
        call_pc = next((k for k in norm if k in [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in pc_variants]), None)
    if not call_pc:
        raise HTTPException(400, "No postcode column in calls file")

    call_dur = _norm_key(_param("callDurationCol") or _param("callsDurationCol"))
    if not call_dur:
        call_dur = next((k for k in norm if any(x in k for x in ("duration","mins","minutes","time","servicetime"))), None)
    if not call_dur:
        raise HTTPException(400, "No duration column in calls file (expected minutes or h:mm)")

    call_days = _norm_key(_param("callDaysCol") or _param("callsDaysCol"))
    if not call_days:
        call_days = next((k for k in norm if any(x in k for x in ("days","weekday","open","availability"))), None)
    if not call_days:
        raise HTTPException(400, "No open-days column in calls file (expected YYYYYNN)")

    # Read resources
    try:
        rdf = pd.read_excel(io.BytesIO(resources_bytes), engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid resources Excel")
    rdf.columns = (
        rdf.columns.str.strip().str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    rnorm = {c.lower(): c for c in rdf.columns}
    res_name = _norm_key(_param("resNameCol") or _param("resourcesNameCol"))
    if not res_name:
        res_name = next((k for k in rnorm if any(x in k for x in ("name","resource","rescource","depot","base"))), None)
    res_pc = _norm_key(_param("resPostcodeCol") or _param("resourcesPostcodeCol"))
    if not res_pc:
        res_pc = next((k for k in rnorm if k in [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in pc_variants]), None)
    res_days = _norm_key(_param("resDaysCol") or _param("resourcesDaysCol"))
    if not res_days:
        res_days = next((k for k in rnorm if any(x in k for x in ("days","weekday","workdays","availability"))), None)
    res_start = _norm_key(_param("resStartCol") or _param("resourcesStartCol"))
    if not res_start:
        res_start = next((k for k in rnorm if any(x in k for x in ("start","shiftstart","starttime"))), None)
    res_end = _norm_key(_param("resEndCol") or _param("resourcesEndCol"))
    if not res_end:
        res_end = next((k for k in rnorm if any(x in k for x in ("end","shiftend","endtime","finish"))), None)
    if not (res_name and res_pc and res_days and res_start and res_end):
        raise HTTPException(400, "Resources file must have name, postcode, work-days, start and end time columns")

    # Geocode calls/resources in bulk
    call_latlng_raw = await geocode_postcodes_bulk([str(v) for v in cdf[call_pc].tolist()])
    call_latlng = []
    call_durations: List[int] = []
    call_open_days: List[List[int]] = []
    for idx, row in cdf.iterrows():
        lat, lng = call_latlng_raw[idx]
        call_latlng.append((lat, lng))
        call_durations.append(int(parse_duration_to_seconds(row[call_dur])))
        call_open_days.append(parse_weekdays(str(row[call_days])))
    calls = pd.DataFrame({
        "Latitude": [lt for lt, _ in call_latlng],
        "Longitude": [ln for _, ln in call_latlng],
        "DurationSec": call_durations,
        "OpenDays": call_open_days
    })
    calls = calls.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    if calls.empty:
        raise HTTPException(400, "No valid call coordinates after geocoding")

    res_latlng_raw = await geocode_postcodes_bulk([str(v) for v in rdf[res_pc].tolist()])
    res_list = []
    resource_day_spans: List[int] = []
    for idx, row in rdf.iterrows():
        name = str(row[res_name]) if res_name in row else "Resource"
        lat, lng = res_latlng_raw[idx]
        days = parse_weekdays(str(row[res_days]))
        start_s = int(parse_hhmm_to_seconds(row[res_start]))
        end_s = int(parse_hhmm_to_seconds(row[res_end]))
        if end_s <= start_s:
            end_s += 24 * 3600
        if not (np.isnan(lat) or np.isnan(lng)):
            span = max(0, end_s - start_s)
            resource_day_spans.append(span)
            res_list.append({"name": name, "lat": float(lat), "lng": float(lng), "days": days, "start": start_s, "end": end_s})
    if not res_list:
        raise HTTPException(400, "No valid resource coordinates after geocoding")

    durations_sec = np.array([d for d in call_durations if d and d > 0], dtype=float)
    typical_service_sec = float(np.median(durations_sec)) if durations_sec.size > 0 else 3600.0
    day_spans_sec = np.array([s for s in resource_day_spans if s and s > 0], dtype=float)
    typical_day_sec = float(np.median(day_spans_sec)) if day_spans_sec.size > 0 else 8 * 3600.0
    total_service_minutes = float(durations_sec.sum() / 60.0) if durations_sec.size > 0 else 0.0
    total_capacity_minutes = 0.0
    for r, span in zip(res_list, resource_day_spans):
        total_capacity_minutes += (span / 60.0) * max(1, len(r["days"]))

    override_max_calls = _param_int("maxCallsPerDay")
    auto_max_calls_per_day: Optional[int] = None
    if override_max_calls is None and typical_service_sec > 0 and typical_day_sec > 0:
        auto_max_calls_per_day = int(max(1, math.floor(typical_day_sec / typical_service_sec)))
        auto_max_calls_per_day = max(4, min(auto_max_calls_per_day, 30))
    MAX_PER_DAY_OVERRIDE = override_max_calls
    coords = calls[["Latitude","Longitude"]].astype(float).to_numpy()
    centers_arr = np.array([[r["lat"], r["lng"]] for r in res_list], dtype=float)

    allowed: List[List[int]] = []
    for i in range(len(calls)):
        cd = set(int(d) for d in calls.at[i, "OpenDays"])
        cand = [j for j, r in enumerate(res_list) if cd.intersection(set(r["days"]))]
        allowed.append(cand or [])

    D_call_to_center_sec = osrm_matrix_between_duration(coords, centers_arr)
    n_calls, k_centers = D_call_to_center_sec.shape
    cap_sec = [sum((r["end"] - r["start"]) for _d in r["days"]) for r in res_list]
    total_cap = max(1, sum(max(0, c) for c in cap_sec))
    quotas = [max(0, int(round(n_calls * (max(0, c) / total_cap)))) for c in cap_sec]
    diff = n_calls - sum(quotas)
    if diff != 0:
        order = np.argsort([-c for c in cap_sec])
        if diff < 0:
            order = np.argsort([c for c in cap_sec])
        idx = 0
        while diff != 0 and len(order) > 0:
            j = int(order[idx % len(order)])
            if diff > 0:
                quotas[j] += 1; diff -= 1
            else:
                if quotas[j] > 0:
                    quotas[j] -= 1; diff += 1
            idx += 1

    allowed2: List[Optional[List[int]]] = [al if al else None for al in allowed]

    assign = _assign_flow(D_call_to_center_sec, quotas, allowed2)

    calls["Resource"] = ""
    calls["DayOfWeek"] = ""
    calls["Sequence"] = 0
    calls["StartSec"] = 0
    calls["EndSec"] = 0
    calls["Week"] = 0

    remaining = {int(i) for i in range(len(calls))}
    calls_by_res: dict[int, List[int]] = {}
    for i, j in enumerate(assign):
        calls_by_res.setdefault(int(j), []).append(int(i))

    TL = _param_int("timeLimitSec", 20)
    MAX_PER_DAY = MAX_PER_DAY_OVERRIDE if MAX_PER_DAY_OVERRIDE is not None else auto_max_calls_per_day
    WMAX = _param_int("maxWeeks", 8)
    weeks_used = 0
    progress_made = True
    while len(remaining) > 0 and weeks_used < max(1, WMAX or 1) and progress_made:
        progress_made = False
        weeks_used += 1
        for r_idx, idxs in calls_by_res.items():
            r = res_list[r_idx]
            depot = (r["lat"], r["lng"])
            # Precompute once per resource: durations matrix for all its calls + depot
            res_idxs = list(sorted(set(int(x) for x in idxs)))
            if len(res_idxs) > 0:
                res_coords = coords[res_idxs]
                all_pts_res = np.vstack([res_coords, np.array([depot])])
                dm_res_all = osrm_table_batch_duration(all_pts_res)  # shape (m+1, m+1), last is depot
                depot_pos = len(res_idxs)
                # Map absolute call index -> position in dm_res_all
                pos_map = {int(abs_i): int(p) for p, abs_i in enumerate(res_idxs)}
            else:
                dm_res_all = None
                depot_pos = 0
                pos_map = {}
            for d in r["days"]:
                elig = [i for i in idxs if i in remaining and d in calls.at[i, "OpenDays"]]
                if not elig:
                    continue
                sub_coords = coords[elig]
                sub_serv = [int(calls.at[i, "DurationSec"]) for i in elig]
                # Optional pre-cluster to compact days spatially before sequencing
                day_groups: List[List[int]] = []
                if VRP_PRECLUSTER and MAX_PER_DAY and int(MAX_PER_DAY) > 0:
                    min_calls = max(1, int(math.floor(VRP_CLUSTER_MIN_RATIO * int(MAX_PER_DAY))))
                    try:
                        if VRP_PRECLUSTER_METHOD == "kmedoids":
                            day_groups = _precluster_days_kmedoids(sub_coords, int(MAX_PER_DAY), VRP_CLUSTER_MIN_RATIO, H3_LAMBDA)
                        else:
                            routes = cvrp_days_for_territory(sub_coords, depot, min_calls, int(MAX_PER_DAY), time_sec=max(5, int(TL or 20)))
                            day_groups = [r for r in routes if len(r) > 0]
                    except Exception:
                        day_groups = []
                if not day_groups:
                    day_groups = [list(range(len(sub_coords)))]
                for grp in day_groups:
                    if not grp:
                        continue
                    grp_coords = sub_coords[grp]
                    grp_serv = [sub_serv[g] for g in grp]
                    # Build a small durations matrix slice for this group from the precomputed resource matrix
                    dm_small = None
                    if dm_res_all is not None and len(grp_coords) > 0:
                        grp_abs = [int(elig[g]) for g in grp]  # absolute indices in calls
                        grp_pos = [pos_map[a] for a in grp_abs if a in pos_map]
                        if len(grp_pos) == len(grp):
                            m = len(grp_pos)
                            dm_small = np.zeros((m+1, m+1), dtype=float)
                            # Fill intra-group travel times
                            for ii in range(m):
                                for jj in range(m):
                                    dm_small[ii, jj] = float(dm_res_all[grp_pos[ii], grp_pos[jj]])
                            # Depot row/col (last index)
                            for ii in range(m):
                                dm_small[m, ii] = float(dm_res_all[depot_pos, grp_pos[ii]])
                                dm_small[ii, m] = float(dm_res_all[grp_pos[ii], depot_pos])

                    visited_rel, _dropped = vrp_single_route_with_time(
                        grp_coords, depot, grp_serv, r["start"], r["end"],
                        time_limit_sec=int(TL or 20), max_stops=int(MAX_PER_DAY) if MAX_PER_DAY else None,
                        durations_matrix=dm_small
                    )
                    if not visited_rel:
                        continue
                    # Use the same dm_small (or compute once if missing) to compute start times
                    if dm_small is None:
                        all_pts = np.vstack([grp_coords, np.array([depot])])
                        dm = osrm_table_batch_duration(all_pts)
                    else:
                        dm = dm_small
                    t = int(r["start"])
                    prev = len(grp_coords)
                    for seq, rel in enumerate(visited_rel, start=1):
                        travel = int(dm[prev, rel])
                        service = int(grp_serv[rel])
                        start_t = t + travel
                        end_t = start_t + service
                        abs_idx = int(elig[grp[rel]])
                        calls.at[abs_idx, "Resource"] = str(r["name"]) if r.get("name") else f"Resource {r_idx+1}"
                        calls.at[abs_idx, "DayOfWeek"] = _DAYS[d]
                        calls.at[abs_idx, "Sequence"] = int(seq)
                        calls.at[abs_idx, "StartSec"] = int(start_t)
                        calls.at[abs_idx, "EndSec"] = int(end_t)
                        calls.at[abs_idx, "Week"] = int(weeks_used)
                        if abs_idx in remaining:
                            remaining.remove(abs_idx)
                            progress_made = True
                        t = end_t
                        prev = rel
    unscheduled = sorted(list(remaining))

    out = cdf.copy()
    out["Latitude"] = [lt for lt, _ in call_latlng]
    out["Longitude"] = [ln for _, ln in call_latlng]
    out["Resource"] = ""
    out["Week"] = 0
    out["DayOfWeek"] = ""
    out["Sequence"] = 0
    out["StartTime"] = ""
    out["EndTime"] = ""

    def _fmt_time(sec: int) -> str:
        h = int(sec // 3600); m = int((sec % 3600) // 60)
        return f"{h:02d}:{m:02d}"

    for i in range(len(calls)):
        out.at[i, "Resource"] = calls.at[i, "Resource"]
        out.at[i, "Week"] = int(calls.at[i, "Week"]) if not pd.isna(calls.at[i, "Week"]) else 0
        out.at[i, "DayOfWeek"] = calls.at[i, "DayOfWeek"]
        out.at[i, "Sequence"] = int(calls.at[i, "Sequence"]) if not pd.isna(calls.at[i, "Sequence"]) else 0
        out.at[i, "StartTime"] = _fmt_time(int(calls.at[i, "StartSec"])) if calls.at[i, "StartSec"] else ""
        out.at[i, "EndTime"] = _fmt_time(int(calls.at[i, "EndSec"])) if calls.at[i, "EndSec"] else ""

    day_map = {name: i for i, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], start=1)}
    out["DayOfWeekNum"] = out["DayOfWeek"].map(lambda d: day_map.get(str(d)[:3], None) if isinstance(d, str) else None)
    uniq_days = (
        out[["Resource","Week","DayOfWeek","DayOfWeekNum"]]
        .drop_duplicates()
        .dropna(subset=["DayOfWeek"])
        .sort_values(["Resource","Week","DayOfWeekNum"], kind="mergesort")
    )
    if len(uniq_days) > 0:
        # Number days sequentially per Resource (do not reset per week)
        uniq_days["DayNumber"] = uniq_days.groupby("Resource").cumcount() + 1
        out = out.merge(uniq_days[["Resource","Week","DayOfWeek","DayNumber"]], on=["Resource","Week","DayOfWeek"], how="left")
    else:
        out["DayNumber"] = None
    out["Day"] = out["DayNumber"].map(lambda n: f"Day {int(n)}" if pd.notna(n) and int(n) > 0 else "")
    cts = out.groupby(["Resource","Day"]).size().rename("CallsPerDay").reset_index()
    out = out.drop(columns=["CallsPerDay"], errors="ignore").merge(cts, on=["Resource","Day"], how="left")

    if "Territory" not in out.columns:
        out["Territory"] = out["Resource"].astype(str)
    out["DepartTime"] = ""
    out["ReturnTime"] = ""
    if out.shape[0] > 0:
        try:
            out["SequenceNumTmp"] = pd.to_numeric(out["Sequence"], errors="coerce")
        except Exception:
            out["SequenceNumTmp"] = None
        grp = out.groupby(["Resource","Day"], dropna=False)
        dep_map: Dict[Tuple[str, str], str] = {}
        ret_map: Dict[Tuple[str, str], str] = {}
        last_idx_map: Dict[Tuple[str, str], Optional[int]] = {}
        for key, df_g in grp:
            try:
                st = (df_g["StartTime"].astype(str).replace({"nan":""}).tolist())
                st_valid = [s for s in st if s]
                depart = min(st_valid) if st_valid else ""
            except Exception:
                depart = ""
            dep_map[key] = depart
            try:
                et = (df_g["EndTime"].astype(str).replace({"nan":""}).tolist())
                et_valid = [s for s in et if s]
                ret = max(et_valid) if et_valid else ""
            except Exception:
                ret = ""
            ret_map[key] = ret
            try:
                mx = df_g["SequenceNumTmp"].idxmax()
            except Exception:
                mx = None
            last_idx_map[key] = mx
        for key, depart in dep_map.items():
            mask = (out["Resource"]==key[0]) & (out["Day"]==key[1])
            out.loc[mask, "DepartTime"] = depart
        for key, mx in last_idx_map.items():
            if mx is not None and mx in out.index:
                out.at[mx, "ReturnTime"] = ret_map.get(key, "")
        out = out.drop(columns=["SequenceNumTmp"], errors="ignore")

    preferred_cols = [
        "Resource","Territory","Week","DayOfWeek","DayOfWeekNum","Day","DayNumber","CallsPerDay",
        "Sequence","StartTime","EndTime","DepartTime","ReturnTime","Latitude","Longitude",
        "ResourceRank","SortOrder","CallName"
    ]
    base_cols = [c for c in preferred_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in base_cols]
    excelCols = base_cols + other_cols
    out_excel = out[excelCols].astype("object").where(pd.notna(out[excelCols]), "").astype(str)
    rows = [excelCols] + out_excel.values.tolist()

    scheduled_count = int(len(calls) - len(unscheduled))
    weeks_used = int(calls["Week"].max()) if scheduled_count > 0 else 0
    weeks_planned = max(1, weeks_used or 1)
    message = f"Routed {scheduled_count} of {len(calls)} calls across {len(res_list)} resources in {weeks_planned} week(s)."
    if unscheduled:
        message += f" Unscheduled calls: {len(unscheduled)} (no feasible weekday/shift within {weeks_planned} week(s))."
    est_weeks = None
    if total_capacity_minutes > 0:
        est_weeks = math.ceil(max(total_service_minutes, 0.0) / max(total_capacity_minutes, 1.0))
        if est_weeks > weeks_planned:
            message += f" Estimated weeks needed at current capacity: {est_weeks}."
    meta = {
        "row_count": max(0, len(rows) - 1),
        "scheduled_count": scheduled_count,
        "unscheduled_count": len(unscheduled),
        "resources": len(res_list),
        "weeks_used": weeks_used,
        "auto_max_calls_per_day": auto_max_calls_per_day,
        "max_calls_per_day": MAX_PER_DAY,
        "service_minutes": total_service_minutes,
        "capacity_minutes": total_capacity_minutes,
        "estimated_weeks": est_weeks,
    }
    project_id = _maybe_save_project(
        headers or {},
        mode="vehicle-route",
        rows=rows,
        meta=meta,
        explicit_name=projectName,
    )

    return {
        "rows": rows,
        "used_resources": True,
        "num_resources": len(res_list),
        "unscheduled_count": len(unscheduled),
        "message": message,
        "project_id": project_id,
    }


@app.post("/api/vehicle-route")
async def vehicle_route(
    callsFile: UploadFile = File(...),
    resourcesFile: UploadFile = File(...),
    # Optional column mappings (normalized matching after trimming and removing spaces/_/-)
    callPostcodeCol: Optional[str] = Form(None),
    callDurationCol: Optional[str] = Form(None),
    callDaysCol: Optional[str] = Form(None),
    resNameCol: Optional[str] = Form(None),
    resPostcodeCol: Optional[str] = Form(None),
    resDaysCol: Optional[str] = Form(None),
    resStartCol: Optional[str] = Form(None),
    resEndCol: Optional[str] = Form(None),
    timeLimitSec: Optional[int] = Form(20),
    maxWeeks: Optional[int] = Form(8),
    maxCallsPerDay: Optional[int] = Form(None),
    projectName: Optional[str] = Form(None)
):
    # Read calls
    cbuf = io.BytesIO(await callsFile.read())
    try:
        cdf = pd.read_excel(cbuf, engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid calls Excel")
    # Normalize headers
    cdf.columns = (
        cdf.columns.str.strip().str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    # Detect columns
    pc_variants = [
        "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
        "zip","zipcode","zip_code","zip code","zip-code","eircode"
    ]
    norm = {c.lower(): c for c in cdf.columns}
    def _norm_key(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        return s.replace(" ", "").replace("_", "").replace("-", "").lower()
    call_pc = _norm_key(callPostcodeCol) or next((k for k in norm if k in [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in pc_variants]), None)
    if not call_pc:
        raise HTTPException(400, "No postcode column in calls file")
    call_dur = _norm_key(callDurationCol) or next((k for k in norm if any(x in k for x in ("duration","mins","minutes","time","servicetime"))), None)
    if not call_dur:
        raise HTTPException(400, "No duration column in calls file (expected minutes or h:mm)")
    call_days = _norm_key(callDaysCol) or next((k for k in norm if any(x in k for x in ("days","weekday","open","availability"))), None)
    if not call_days:
        raise HTTPException(400, "No open-days column in calls file (expected YYYYYNN)")

    # Read resources
    rbuf = io.BytesIO(await resourcesFile.read())
    try:
        rdf = pd.read_excel(rbuf, engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid resources Excel")
    rdf.columns = (
        rdf.columns.str.strip().str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    rnorm = {c.lower(): c for c in rdf.columns}
    res_name = _norm_key(resNameCol) or next((k for k in rnorm if any(x in k for x in ("name","resource","rescource","depot","base"))), None)
    res_pc = _norm_key(resPostcodeCol) or next((k for k in rnorm if k in [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in pc_variants]), None)
    res_days = _norm_key(resDaysCol) or next((k for k in rnorm if any(x in k for x in ("days","weekday","workdays","availability"))), None)
    res_start = _norm_key(resStartCol) or next((k for k in rnorm if any(x in k for x in ("start","shiftstart","starttime"))), None)
    res_end = _norm_key(resEndCol) or next((k for k in rnorm if any(x in k for x in ("end","shiftend","endtime","finish"))), None)
    if not (res_name and res_pc and res_days and res_start and res_end):
        raise HTTPException(400, "Resources file must have name, postcode, work-days, start and end time columns")

    # Geocode calls
    call_latlng_raw = await geocode_postcodes_bulk([str(v) for v in cdf[call_pc].tolist()])
    call_latlng = []
    call_durations = []
    call_open_days: List[List[int]] = []
    for idx, row in cdf.iterrows():
        lat, lng = call_latlng_raw[idx]
        call_latlng.append((lat, lng))
        call_durations.append(int(parse_duration_to_seconds(row[call_dur])))
        call_open_days.append(parse_weekdays(str(row[call_days])))
    calls = pd.DataFrame({
        "Latitude": [lt for lt, _ in call_latlng],
        "Longitude": [ln for _, ln in call_latlng],
        "DurationSec": call_durations,
        "OpenDays": call_open_days
    })
    calls = calls.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    if calls.empty:
        raise HTTPException(400, "No valid call coordinates after geocoding")

    # Geocode resources
    res_latlng_raw = await geocode_postcodes_bulk([str(v) for v in rdf[res_pc].tolist()])
    res_list = []
    for idx, row in rdf.iterrows():
        name = str(row[res_name]) if res_name in row else "Resource"
        lat, lng = res_latlng_raw[idx]
        days = parse_weekdays(str(row[res_days]))
        start_s = int(parse_hhmm_to_seconds(row[res_start]))
        end_s = int(parse_hhmm_to_seconds(row[res_end]))
        # Handle overnight shifts where end wraps past midnight
        if end_s <= start_s:
            end_s += 24 * 3600
        if not (np.isnan(lat) or np.isnan(lng)):
            res_list.append({"name": name, "lat": float(lat), "lng": float(lng), "days": days, "start": start_s, "end": end_s})
    if not res_list:
        raise HTTPException(400, "No valid resource coordinates after geocoding")

    coords = calls[["Latitude","Longitude"]].astype(float).to_numpy()
    centers_arr = np.array([[r["lat"], r["lng"]] for r in res_list], dtype=float)

    # Allowed resources per call: overlap on any day
    allowed: List[List[int]] = []
    for i in range(len(calls)):
        cd = set(int(d) for d in calls.at[i, "OpenDays"])
        cand = [j for j, r in enumerate(res_list) if cd.intersection(set(r["days"]))]
        if not cand:
            # No overlap with any resource; we'll attempt later but mark none now
            cand = []
        allowed.append(cand)

    # Assignment of calls to resources (balanced by weekly capacity)
    D_call_to_center_sec = osrm_matrix_between_duration(coords, centers_arr)
    n_calls, k_centers = D_call_to_center_sec.shape
    # Compute weekly capacity weights per resource (sum of daily shift seconds)
    cap_sec = [sum((r["end"] - r["start"]) for _d in r["days"]) for r in res_list]
    total_cap = max(1, sum(max(0, c) for c in cap_sec))
    quotas = [max(0, int(round(n_calls * (max(0, c) / total_cap)))) for c in cap_sec]
    # Adjust to match exactly n_calls
    diff = n_calls - sum(quotas)
    if diff != 0:
        # distribute diff by rounding residuals using nearest center distances as tie-breaker
        order = np.argsort([-c for c in cap_sec])  # more capacity first for positive diff
        if diff < 0:
            order = np.argsort([c for c in cap_sec])  # less capacity first to reduce
        idx = 0
        while diff != 0 and len(order) > 0:
            j = int(order[idx % len(order)])
            if diff > 0:
                quotas[j] += 1; diff -= 1
            else:
                if quotas[j] > 0:
                    quotas[j] -= 1; diff += 1
            idx += 1

    # Ensure allowed list has something; if empty, allow all as last resort
    allowed2: List[Optional[List[int]]]=[]
    for al in allowed:
        if not al:
            allowed2.append(None)
        else:
            allowed2.append(al)

    # Assign with min-cost flow if available, else greedy nearest with quotas
    def _assign_flow(D: np.ndarray, quotas: List[int], allowed: List[Optional[List[int]]]) -> np.ndarray:
        n, k = D.shape
        if not _HAS_FLOW:
            # Greedy nearest with quotas and allowed
            assign = -np.ones(n, dtype=int)
            cap = quotas[:]
            order_calls = list(range(n))
            order_calls.sort(key=lambda i: float(np.min(D[i])))
            for i in order_calls:
                order_centers = list(np.argsort(D[i]).astype(int))
                for j in order_centers:
                    if allowed[i] is not None and j not in allowed[i]:
                        continue
                    if cap[j] > 0:
                        assign[i] = j
                        cap[j] -= 1
                        break
            # Fill any remaining arbitrarily
            for i in range(n):
                if assign[i] == -1:
                    j = int(np.argmin([D[i, jj] if (allowed[i] is None or jj in allowed[i]) else 1e18 for jj in range(k)]))
                    assign[i] = j
            return assign
        # Use existing flow helper
        # Reuse internal flow builder with allowed set
        def _assign_min_cost_flow_vrp(D: np.ndarray, quotas: list[int], allowed: List[Optional[List[int]]]) -> np.ndarray:
            n_calls, n_centers = D.shape
            nearest = np.min(D, axis=1)
            start_indices: list[int] = []
            end_indices: list[int] = []
            capacities: list[int] = []
            unit_costs: list[int] = []
            source = n_centers + n_calls
            sink = source + 1
            num_nodes = sink + 1
            supplies = [0] * num_nodes
            supplies[source] = n_calls
            supplies[sink] = -n_calls
            for j in range(n_centers):
                start_indices.append(source); end_indices.append(j); capacities.append(int(quotas[j])); unit_costs.append(0)
            for j in range(n_centers):
                for i in range(n_calls):
                    if allowed[i] is not None and j not in allowed[i]:
                        continue
                    start_indices.append(j); end_indices.append(n_centers + i); capacities.append(1)
                    base = float(D[i, j])
                    penalty = 0.0
                    if nearest[i] > 0:
                        ratio = base / nearest[i]
                        if ratio > FLOW_ALPHA:
                            penalty = (ratio - FLOW_ALPHA) * FLOW_GAMMA * nearest[i]
                    c = int(max(1, round(base + penalty)))
                    unit_costs.append(c)
            for i in range(n_calls):
                start_indices.append(n_centers + i); end_indices.append(sink); capacities.append(1); unit_costs.append(0)
            mcf = _new_mcf()
            def _add_arc(m,u,v,cap,cost):
                if hasattr(m,'AddArcWithCapacityAndUnitCost'):
                    m.AddArcWithCapacityAndUnitCost(u,v,int(cap),int(cost))
                else:
                    m.add_arc_with_capacity_and_unit_cost(u,v,int(cap),int(cost))
            def _set_supply(m,node,sup):
                if hasattr(m,'SetNodeSupply'):
                    m.SetNodeSupply(node,int(sup))
                else:
                    m.set_node_supply(node,int(sup))
            def _solve(m):
                return m.Solve() if hasattr(m,'Solve') else m.solve()
            def _num_arcs(m):
                return m.NumArcs() if hasattr(m,'NumArcs') else m.num_arcs()
            def _tail(m,a):
                return m.Tail(a) if hasattr(m,'Tail') else m.tail(a)
            def _head(m,a):
                return m.Head(a) if hasattr(m,'Head') else m.head(a)
            def _flow(m,a):
                return m.Flow(a) if hasattr(m,'Flow') else m.flow(a)
            for u,v,cap,cost in zip(start_indices,end_indices,capacities,unit_costs):
                _add_arc(mcf,u,v,cap,cost)
            for node,sup in enumerate(supplies):
                _set_supply(mcf,node,sup)
            status = _solve(mcf)
            optimal_code = getattr(mcf,'OPTIMAL',0)
            if status != optimal_code:
                raise RuntimeError(f"Flow failed {status}")
            assign = -np.ones(n_calls, dtype=int)
            for a in range(_num_arcs(mcf)):
                u = _tail(mcf,a); v = _head(mcf,a)
                if 0 <= u < n_centers and n_centers <= v < n_centers + n_calls:
                    if _flow(mcf,a) > 0:
                        assign[v - n_centers] = u
            return assign
        return _assign_min_cost_flow_vrp(D, quotas, allowed)

    assign = _assign_flow(D_call_to_center_sec, quotas, allowed2)

    # Build schedules per resource across multiple weeks if needed
    calls["Resource"] = ""
    calls["DayOfWeek"] = ""
    calls["Sequence"] = 0
    calls["StartSec"] = 0
    calls["EndSec"] = 0
    calls["Week"] = 0

    remaining = {int(i) for i in range(len(calls))}
    calls_by_res: dict[int, List[int]] = {}
    for i, j in enumerate(assign):
        if int(j) not in calls_by_res:
            calls_by_res[int(j)] = []
        calls_by_res[int(j)].append(int(i))

    unscheduled: List[int] = []
    TL = int(timeLimitSec or 20)
    MAX_PER_DAY = int(maxCallsPerDay) if (maxCallsPerDay is not None and str(maxCallsPerDay).strip() != '') else None
    WMAX = int(maxWeeks or 8)
    weeks_used = 0
    progress_made = True
    while len(remaining) > 0 and weeks_used < max(1, WMAX) and progress_made:
        progress_made = False
        weeks_used += 1
        for r_idx, idxs in calls_by_res.items():
            r = res_list[r_idx]
            depot = (r["lat"], r["lng"]) 
            # Iterate through working days
            for d in r["days"]:
                # eligible calls for this weekday
                elig = [i for i in idxs if i in remaining and d in calls.at[i, "OpenDays"]]
                if not elig:
                    continue
                sub_coords = coords[elig]
                sub_serv = [int(calls.at[i, "DurationSec"]) for i in elig]
                visited_rel, _dropped = vrp_single_route_with_time(
                    sub_coords, depot, sub_serv, r["start"], r["end"], time_limit_sec=TL, max_stops=MAX_PER_DAY
                )
                if not visited_rel:
                    continue
                # Record sequence; approximate times by cumulative travel+service starting at start
                # For simplicity, compute linear schedule using OSRM durations subset
                all_pts = np.vstack([sub_coords, np.array([depot])])
                dm = osrm_table_batch_duration(all_pts)
                t = int(r["start"])  # start of day
                prev = len(sub_coords)  # depot index in all_pts
                for seq, rel in enumerate(visited_rel, start=1):
                    travel = int(dm[prev, rel])
                    service = int(sub_serv[rel])
                    start_t = t + travel
                    end_t = start_t + service
                    abs_idx = int(elig[rel])
                    calls.at[abs_idx, "Resource"] = str(r["name"]) if r.get("name") else f"Resource {r_idx+1}"
                    calls.at[abs_idx, "DayOfWeek"] = _DAYS[d]
                    calls.at[abs_idx, "Sequence"] = int(seq)
                    calls.at[abs_idx, "StartSec"] = int(start_t)
                    calls.at[abs_idx, "EndSec"] = int(end_t)
                    calls.at[abs_idx, "Week"] = int(weeks_used)
                    if abs_idx in remaining:
                        remaining.remove(abs_idx)
                        progress_made = True
                    t = end_t
                    prev = rel
        # loop next week if needed
    unscheduled = sorted(list(remaining))

    # Build output rows
    # Merge back to original calls sheet to preserve extra columns
    out = cdf.copy()
    out["Latitude"] = [lt for lt, _ in call_latlng]
    out["Longitude"] = [ln for _, ln in call_latlng]
    out["Resource"] = ""
    out["Week"] = 0
    out["DayOfWeek"] = ""
    out["Sequence"] = 0
    out["StartTime"] = ""
    out["EndTime"] = ""
    def _fmt_time(sec: int) -> str:
        h = int(sec // 3600); m = int((sec % 3600) // 60)
        return f"{h:02d}:{m:02d}"
    for i in range(len(calls)):
        out.at[i, "Resource"] = calls.at[i, "Resource"]
        out.at[i, "Week"] = int(calls.at[i, "Week"]) if not pd.isna(calls.at[i, "Week"]) else 0
        out.at[i, "DayOfWeek"] = calls.at[i, "DayOfWeek"]
        out.at[i, "Sequence"] = int(calls.at[i, "Sequence"]) if not pd.isna(calls.at[i, "Sequence"]) else 0
        out.at[i, "StartTime"] = _fmt_time(int(calls.at[i, "StartSec"])) if calls.at[i, "StartSec"] else ""
        out.at[i, "EndTime"] = _fmt_time(int(calls.at[i, "EndSec"])) if calls.at[i, "EndSec"] else ""

    # Derive:
    # - DayOfWeekNum (1..7)
    # - DayNumber: consecutive day index per resource across weeks ordered by (Week, DayOfWeekNum)
    # - Day: label 'Day N'
    # - CallsPerDay: per (Resource, Day) count
    day_map = {name: i for i, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], start=1)}
    out["DayOfWeekNum"] = out["DayOfWeek"].map(lambda d: day_map.get(str(d)[:3], None) if isinstance(d, str) else None)
    # Unique (resource, week, day) rows to compute consecutive day index
    uniq_days = (
        out[["Resource","Week","DayOfWeek","DayOfWeekNum"]]
        .drop_duplicates()
        .dropna(subset=["DayOfWeek"])
        .sort_values(["Resource","Week","DayOfWeekNum"], kind="mergesort")
    )
    if len(uniq_days) > 0:
        # Number days sequentially per Resource
        uniq_days["DayNumber"] = uniq_days.groupby("Resource").cumcount() + 1
        out = out.merge(uniq_days[["Resource","Week","DayOfWeek","DayNumber"]], on=["Resource","Week","DayOfWeek"], how="left")
    else:
        out["DayNumber"] = None
    out["Day"] = out["DayNumber"].map(lambda n: f"Day {int(n)}" if pd.notna(n) and int(n) > 0 else "")
    # Recompute CallsPerDay by (Resource, Day)
    cts = out.groupby(["Resource","Day"]).size().rename("CallsPerDay").reset_index()
    out = out.drop(columns=["CallsPerDay"], errors="ignore").merge(cts, on=["Resource","Day"], how="left")

    # Derive Territory (default to Resource if not present)
    if "Territory" not in out.columns:
        out["Territory"] = out["Resource"].astype(str)
    # DepartTime: per (Resource, Day) earliest StartTime; ReturnTime: per (Resource, Day) latest EndTime on last-seq row
    out["DepartTime"] = ""
    out["ReturnTime"] = ""
    if out.shape[0] > 0:
        # ensure Sequence numeric for grouping
        try:
            out["SequenceNumTmp"] = pd.to_numeric(out["Sequence"], errors="coerce")
        except Exception:
            out["SequenceNumTmp"] = None
        grp = out.groupby(["Resource","Day"], dropna=False)
        # Build a dict of depart/return per group
        dep_map = {}
        ret_map = {}
        last_idx_map = {}
        for key, df_g in grp:
            # earliest StartTime string
            try:
                st = (df_g["StartTime"].astype(str).replace({"nan":""}).tolist())
                st_valid = [s for s in st if s]
                depart = min(st_valid) if st_valid else ""
            except Exception:
                depart = ""
            dep_map[key] = depart
            try:
                et = (df_g["EndTime"].astype(str).replace({"nan":""}).tolist())
                et_valid = [s for s in et if s]
                ret = max(et_valid) if et_valid else ""
            except Exception:
                ret = ""
            ret_map[key] = ret
            # row index with max sequence
            try:
                mx = df_g["SequenceNumTmp"].idxmax()
            except Exception:
                mx = None
            last_idx_map[key] = mx
        # apply maps
        for key, depart in dep_map.items():
            mask = (out["Resource"]==key[0]) & (out["Day"]==key[1])
            out.loc[mask, "DepartTime"] = depart
        for key, mx in last_idx_map.items():
            if mx is not None and mx in out.index:
                out.at[mx, "ReturnTime"] = ret_map.get(key, "")
        out = out.drop(columns=["SequenceNumTmp"], errors="ignore")

    # Prepare rows array in expected order
    preferred_cols = [
        "Resource","Territory","Week","DayOfWeek","DayOfWeekNum","Day","DayNumber","CallsPerDay",
        "Sequence","StartTime","EndTime","DepartTime","ReturnTime","Latitude","Longitude",
        "ResourceRank","SortOrder","CallName"
    ]
    # Keep preferred order for those present, then append any others
    base_cols = [c for c in preferred_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in base_cols]
    excelCols = base_cols + other_cols
    out_excel = out[excelCols].astype("object").where(pd.notna(out[excelCols]), "").astype(str)
    rows = [excelCols] + out_excel.values.tolist()

    scheduled_count = int(len(calls) - len(unscheduled))
    weeks_used = int(calls["Week"].max()) if scheduled_count > 0 else 0
    message = f"Routed {scheduled_count} of {len(calls)} calls across {len(res_list)} resources in {max(1,weeks_used or 1)} week(s)."
    if unscheduled:
        message += f" Unscheduled calls: {len(unscheduled)} (no feasible weekday/shift within {max(1,weeks_used or 1)} week(s))."
    meta = {
        "row_count": max(0, len(rows) - 1),
        "scheduled_count": scheduled_count,
        "unscheduled_count": len(unscheduled),
        "resources": len(res_list),
        "weeks_used": weeks_used,
    }
    project_id = None
    project_save_request = locals().get("request")
    if project_save_request is not None:
        project_id = _maybe_save_project(
            project_save_request,
            mode="vehicle-route",
            rows=rows,
            meta=meta,
            explicit_name=projectName,
        )

    return JSONResponse({
        "rows": rows,
        "used_resources": True,
        "num_resources": len(res_list),
        "unscheduled_count": len(unscheduled),
        "message": message,
        "project_id": project_id,
    })


# ----------------------------
# New Mode Endpoints
# ----------------------------
@app.post("/api/mode/cluster")
async def mode_cluster(
    file: UploadFile = File(...),
    minCalls: int = Form(5),
    maxCalls: int = Form(6),
    groupCol: Optional[str] = Form(None),
    dayTimeLimitSec: Optional[int] = Form(30)
):
    """Wrapper that reuses /api/cluster; optional time-budget mode may be enabled via env later."""
    # Reuse cluster_only implementation directly
    return await cluster_only(file=file, minCalls=minCalls, maxCalls=maxCalls, groupCol=groupCol, dayTimeLimitSec=dayTimeLimitSec)


@app.post("/api/mode/territories")
async def mode_territories(
    file: UploadFile = File(...),
    numTerritories: int = Form(3),
    minCalls: int = Form(0),
    maxCalls: int = Form(0),
    resourcesFile: Optional[UploadFile] = File(None),
    resourceLocations: Optional[str] = Form(None),
    groupCol: Optional[str] = Form(None)
):
    """Return only Territory assignment + suggested bases + per-territory KPIs (no day routing)."""
    buf = io.BytesIO(await file.read())
    try:
        df = pd.read_excel(buf, engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid Excel")

    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "", regex=False).str.replace("_", "", regex=False).str.replace("-", "", regex=False)
    )
    postcode_variants = [
        "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
        "zip","zipcode","zip_code","zip code","zip-code","eircode","targetpostcode","target_postcode","target post code"
    ]
    pc_norm = [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in postcode_variants]
    pc = next((c for c in df.columns if c in pc_norm), None)
    if not pc:
        raise HTTPException(400, "No postcode column found")

    latitudes, longitudes, failed_indices = await _resolve_coordinates_with_fallback(df, pc)
    df["Latitude"] = latitudes
    df["Longitude"] = longitudes
    geo = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    if geo.empty:
        raise HTTPException(400, "No valid coordinates after geocoding.")

    # Parse resources (optional centers)
    resources = None
    if resourcesFile is not None:
        try:
            rbuf = io.BytesIO(await resourcesFile.read())
            rdf = pd.read_excel(rbuf, engine="openpyxl").convert_dtypes()
            rcols = rdf.columns.str.strip().str.lower().str.replace(" ", "", regex=False).str.replace("_", "", regex=False).str.replace("-", "", regex=False)
            rdf.columns = rcols
            name_col = next((c for c in rdf.columns if c in ("name","resourcename","site","depot","base")), None)
            pc_col = next((c for c in rdf.columns if c in pc_norm), None)
            lat_col = next((c for c in rdf.columns if c in ("lat","latitude")), None)
            lng_col = next((c for c in rdf.columns if c in ("lng","lon","long","longitude")), None)
            if name_col is None:
                name_col = "name"; 
                if name_col not in rdf.columns:
                    rdf[name_col] = [f"Resource {i+1}" for i in range(len(rdf))]
            rs = []
            for i, row in rdf.iterrows():
                name = str(row.get(name_col, f"Resource {i+1}") or f"Resource {i+1}")
                lat = row.get(lat_col) if lat_col else None
                lng = row.get(lng_col) if lng_col else None
                if (lat is not None) and (lng is not None):
                    try:
                        rs.append({"lat": float(lat), "lng": float(lng), "name": name})
                        continue
                    except Exception:
                        pass
                pcv = str(row.get(pc_col, "")).strip() if pc_col else ""
                if pcv:
                    lt, lg = geocode_postcode_norm(pcv)
                    if not (np.isnan(lt) or np.isnan(lg)):
                        rs.append({"lat": float(lt), "lng": float(lg), "name": name})
            if len(rs) > 0:
                resources = rs
        except Exception:
            resources = None
    elif resourceLocations:
        try:
            raw = json.loads(resourceLocations)
            if isinstance(raw, list):
                rs = []
                for i, r in enumerate(raw):
                    if isinstance(r, dict):
                        name = r.get("name", f"Resource {i+1}")
                        if r.get("lat") is not None and r.get("lng") is not None:
                            try:
                                rs.append({"lat": float(r.get("lat")), "lng": float(r.get("lng")), "name": name})
                                continue
                            except Exception:
                                pass
                        pcv = r.get("postcode") or r.get("post_code") or r.get("post code")
                        if pcv:
                            lt, lg = geocode_postcode_norm(str(pcv))
                            if not (np.isnan(lt) or np.isnan(lg)):
                                rs.append({"lat": float(lt), "lng": float(lg), "name": name})
                if len(rs) > 0:
                    resources = rs
        except Exception:
            resources = None

    coords = geo[["Latitude","Longitude"]].astype(float).to_numpy()
    if resources is not None and len(resources) > 0:
        assign, centers, _ = plan_territories(coords, len(resources), 0, max(1, len(geo)//max(1,len(resources))), resources)
    else:
        assign, centers, _ = plan_territories(coords, numTerritories, 0, max(1, int(np.ceil(len(geo)/max(1,numTerritories)))), None)

    geo["Territory"] = [centers[int(t)].get("name", f"Territory {int(t)+1}") if (resources is not None and int(t) < len(centers)) else f"Territory {int(t)+1}" for t in assign]

    # KPIs per territory
    terr_ids = np.unique(assign)
    terr_kpis = []
    for t in terr_ids:
        idxs = np.where(assign == t)[0]
        sub = coords[idxs]
        c = centers[int(t)]
        depot = (float(c["lat"]), float(c["lng"]))
        if len(sub) == 0:
            terr_kpis.append({"territory": int(t)+1, "count": 0, "avg_km_to_base": 0.0})
            continue
        M = osrm_matrix_between(sub, np.array([[depot[0], depot[1]]]))  # meters
        avg_km = float(np.mean(M[:, 0])) / 1000.0
        terr_kpis.append({"territory": int(t)+1, "count": int(len(sub)), "avg_km_to_base": round(avg_km, 3)})

    # Split within each territory into days if min/max provided; else produce territory only
    if int(maxCalls or 0) > 0:
        geo["Day"] = ""
        coords = geo[["Latitude","Longitude"]].astype(float).to_numpy()
        total_routes = 0
        for terr in geo["Territory"].unique():
            mask = (geo["Territory"] == terr).to_numpy()
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            sub = coords[idxs]
            if len(sub) == 1:
                depot = (float(sub[0,0]), float(sub[0,1]))
            else:
                D = osrm_table_batch(sub)
                medoid = k_medoids(D, 1, iters=10, random_state=42)[0]
                depot = (float(sub[medoid,0]), float(sub[medoid,1]))
            routes = cvrp_days_for_territory(sub, depot, int(minCalls or 0), int(maxCalls or 0), time_sec=20)
            total_routes += len(routes)
            for i, rels in enumerate(routes, start=1):
                for r in rels:
                    geo.at[idxs[r], "Day"] = f"Day {i}"
        # Derive DayNumber and CallsPerDay
        def _day_num(v):
            try:
                import re as _re
                m = _re.search(r"(\d+)", str(v))
                return int(m.group(1)) if m else None
            except Exception:
                return None
        geo["DayNumber"] = geo["Day"].map(_day_num)
        cts = geo.groupby(["Territory","Day"]).size().rename("CallsPerDay").reset_index()
        geo = geo.merge(cts, on=["Territory","Day"], how="left")
        outCols = [c for c in geo.columns if c not in ("Latitude","Longitude","Territory","Day","DayNumber","CallsPerDay")]
        excelCols = ["Territory","Day","DayNumber","CallsPerDay","Latitude","Longitude"] + outCols
    else:
        outCols = [c for c in geo.columns if c not in ("Latitude","Longitude","Territory")]
        excelCols = ["Territory","Latitude","Longitude"] + outCols
    geo_excel = geo[excelCols].astype("object").where(pd.notna(geo[excelCols]), "").astype(str)
    rows = [excelCols] + geo_excel.values.tolist()

    return JSONResponse({
        "rows": rows,
        "suggested_locations": centers,
        "territory_kpis": terr_kpis,
        "message": f"Assigned {len(geo)} calls into {int(len(centers))} territories."
    })


@app.post("/api/mode/routes")
async def mode_routes(
    # Primary sheet (usually contains territory/group assignment, may also include postcodes)
    file: UploadFile = File(...),
    # Optional separate calls file (durations/coordinates/time windows can be sourced here)
    callsFile: Optional[UploadFile] = File(None),
    resourcesFile: Optional[UploadFile] = File(None),
    # Manual mappings for the main sheet
    groupCol: Optional[str] = Form(None),
    durationCol: Optional[str] = Form(None),
    # Manual mappings for calls file
    callsPostcodeCol: Optional[str] = Form(None),
    callsDurationCol: Optional[str] = Form(None),
    callsDaysCol: Optional[str] = Form(None),
    callsNameCol: Optional[str] = Form(None),
    callsLatCol: Optional[str] = Form(None),
    callsLngCol: Optional[str] = Form(None),
    callsWindowStartCol: Optional[str] = Form(None),
    callsWindowEndCol: Optional[str] = Form(None),
    # Manual mappings for resources file
    resNameCol: Optional[str] = Form(None),
    resPostcodeCol: Optional[str] = Form(None),
    resLatCol: Optional[str] = Form(None),
    resLngCol: Optional[str] = Form(None),
    resDaysCol: Optional[str] = Form(None),
    resStartCol: Optional[str] = Form(None),
    resEndCol: Optional[str] = Form(None),
    # Routing knobs
    depotStrategy: Optional[str] = Form("medoid"),
    workDayMinutes: Optional[int] = Form(480),
    breakMinutes: Optional[int] = Form(0),  # reserved
    maxRouteMinutes: Optional[int] = Form(None),  # reserved
    timeLimitSec: Optional[int] = Form(20),
    maxWeeks: Optional[int] = Form(8),
    fastMode: Optional[int] = Form(0),
    seed: Optional[int] = Form(None)
):
    """Turn existing territories into day routes using time budget if enabled; else fallback to count-based split."""
    # Deterministic seeds & run metadata
    try:
        seed_int = int(seed) if seed is not None else int(os.getenv("ROUTER_SEED", "42"))
    except Exception:
        seed_int = 42
    try:
        random.seed(seed_int)
        np.random.seed(seed_int)
    except Exception:
        pass
    run_id = f"run_{int(time.time()*1000)}"
    qa = {"run_id": run_id, "seed": seed_int, "warnings": []}
    buf = io.BytesIO(await file.read())
    try:
        df = pd.read_excel(buf, engine="openpyxl").convert_dtypes()
    except Exception:
        raise HTTPException(400, "Invalid Excel")

    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "", regex=False).str.replace("_", "", regex=False).str.replace("-", "", regex=False)
    )
    # Columns
    postcode_variants = [
        "postcode","post_code","post code","post-code","postalcode","postal_code","postal code","postal-code",
        "zip","zipcode","zip_code","zip code","zip-code","eircode","targetpostcode","target_postcode","target post code"
    ]
    pc_norm = [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in postcode_variants]
    pc = next((c for c in df.columns if c in pc_norm), None)
    # Don't fail yet; if a callsFile is provided, we can use that as the primary source
    terr_col = None
    if groupCol and groupCol.lower() in [c.lower() for c in df.columns]:
        terr_col = [c for c in df.columns if c.lower() == groupCol.lower()][0]
    else:
        terr_col = next((
            c
            for c in df.columns
            if c
            in (
                "territory",
                "territories",
                "territoryname",
                "territoryid",
                "territorycode",
                "territorydescription",
                "territorydesc",
                "group",
                "grouping",
                "groupname",
                "cluster",
                "route",
                "routename",
                "team",
                "teamname",
                "teamid",
                "teamcode",
                "worker",
                "workernumber",
                "resource",
                "resourcename",
                "resourcegroup",
                "rep",
                "salesrep",
                "salesperson",
                "salespersonname",
                "driver",
                "engineer",
                "technician",
                "tech",
                "agent",
                "advisor",
                "advocate",
                "area",
                "zone",
                "region",
                "district",
                "division",
                "market",
                "branch",
                "pod",
                "crew",
                "crewname",
                "manager",
                "accountmanager",
                "owner",
                "day",
                "dayname",
                "weekday",
                "weekpart",
                "cell",
                "subteam",
            )
        ), None)
    # If no territory column, we'll route directly using resources (if provided). Only error if neither is available.
    # If a separate calls file is provided, we'll source durations from it; otherwise use this sheet
    dur_col = None
    if callsFile is None:
        if durationCol and durationCol.lower() in [c.lower() for c in df.columns]:
            dur_col = [c for c in df.columns if c.lower() == durationCol.lower()][0]
        else:
            dcands = [c for c in df.columns if any(x in c for x in ("duration","mins","minutes","time","servicetime"))]
            dur_col = dcands[0] if dcands else None
        if not dur_col:
            raise HTTPException(400, "A duration column is required (provide here or via calls file)")

    # Geocode (or reuse existing lat/lng if present)
    # Optionally merge with calls file for durations/coords/windows
    def _norm_cols(cols: pd.Index) -> pd.Index:
        return (cols.str.strip().str.lower().str.replace(" ", "", regex=False)
                .str.replace("_", "", regex=False).str.replace("-", "", regex=False))

    calls_days_series: Optional[pd.Series] = None
    calls_days_col: Optional[str] = None
    lat_col: Optional[str] = None
    lon_col: Optional[str] = None
    win_from_col: Optional[str] = None
    win_to_col: Optional[str] = None
    if callsFile is not None:
        try:
            cbuf = io.BytesIO(await callsFile.read())
            cdf = pd.read_excel(cbuf, engine="openpyxl").convert_dtypes()
        except Exception:
            raise HTTPException(400, "Invalid calls Excel")
        cdf.columns = _norm_cols(cdf.columns)
        # Determine calls columns using optional mappings
        def _norm_key(s: Optional[str]) -> Optional[str]:
            if not s:
                return None
            return s.replace(" ", "").replace("_", "").replace("-", "").lower()
        calls_pc = _norm_key(callsPostcodeCol) or next((c for c in cdf.columns if c in pc_norm), None)
        if not calls_pc:
            raise HTTPException(400, "No postcode column in calls file")
        calls_dur = _norm_key(callsDurationCol) or next((k for k in cdf.columns if any(x in k for x in ("duration","mins","minutes","time","servicetime"))), None)
        if not calls_dur:
            raise HTTPException(400, "No duration column in calls file")
        calls_lat = _norm_key(callsLatCol) or next((c for c in cdf.columns if c in ("lat","latitude")), None)
        calls_lng = _norm_key(callsLngCol) or next((c for c in cdf.columns if c in ("lng","lon","long","longitude")), None)
        calls_name = _norm_key(callsNameCol) or next((c for c in cdf.columns if any(x in c for x in ("name","id","call","customer","account","site"))), None)
        # Optional time windows from calls file
        calls_win_from = _norm_key(callsWindowStartCol) or next((c for c in cdf.columns if any(x in c for x in ("openfrom","start","windowstart"))), None)
        calls_win_to = _norm_key(callsWindowEndCol) or next((c for c in cdf.columns if any(x in c for x in ("opento","end","windowend","finish"))), None)
        calls_days = _norm_key(callsDaysCol) or next((c for c in cdf.columns if any(x in c for x in ("days","weekday","open","availability"))), None)

        if pc is None:
            # Use calls file as the main dataset directly
            df = cdf.copy()
            pc = calls_pc
            dur_col = calls_dur
            lat_col = calls_lat or next((c for c in df.columns if c in ("lat","latitude")), None)
            lon_col = calls_lng or next((c for c in df.columns if c in ("lng","lon","long","longitude")), None)
            win_from_col = calls_win_from
            win_to_col = calls_win_to
            calls_days_col = calls_days if (calls_days and calls_days in df.columns) else None
            # Map call name to a standardized column for output
            if calls_name and calls_name in df.columns and "CallName" not in df.columns:
                df = df.rename(columns={calls_name: "CallName"})
            if calls_days_col is not None:
                try:
                    calls_days_series = df[calls_days_col].astype("object")
                except Exception:
                    calls_days_series = None
        else:
            # Normalize postcode join key on both frames then merge
            def _pc_norm_series(s: pd.Series) -> pd.Series:
                return s.astype(str).str.replace(r"\s+", "", regex=True).str.upper()
            df = df.assign(_PC=_pc_norm_series(df[pc]))
            cdf = cdf.assign(_PC=_pc_norm_series(cdf[calls_pc]))
            # Select and namespace calls columns to avoid collisions
            merge_cols = [col for col in [calls_dur, calls_lat, calls_lng, calls_win_from, calls_win_to, calls_days, calls_name] if col]
            ckeep = ["_PC"] + [c for c in merge_cols if c in cdf.columns]
            csub = cdf[ckeep].copy()
            rename_map = {c: f"calls__{c}" for c in csub.columns if c != "_PC"}
            csub = csub.rename(columns=rename_map)
            df = df.merge(csub, on="_PC", how="left")
            # Set which columns will be used post-merge (namespaced)
            dur_col = f"calls__{calls_dur}" if calls_dur and f"calls__{calls_dur}" in df.columns else dur_col
            lat_col = f"calls__{calls_lat}" if calls_lat and f"calls__{calls_lat}" in df.columns else next((c for c in df.columns if c in ("lat","latitude")), None)
            lon_col = f"calls__{calls_lng}" if calls_lng and f"calls__{calls_lng}" in df.columns else next((c for c in df.columns if c in ("lng","lon","long","longitude")), None)
            win_from_col = f"calls__{calls_win_from}" if calls_win_from and f"calls__{calls_win_from}" in df.columns else None
            win_to_col = f"calls__{calls_win_to}" if calls_win_to and f"calls__{calls_win_to}" in df.columns else None
            calls_days_col = f"calls__{calls_days}" if calls_days and f"calls__{calls_days}" in df.columns else None
            if calls_name and f"calls__{calls_name}" in df.columns and "CallName" not in df.columns:
                df = df.rename(columns={f"calls__{calls_name}": "CallName"})
            if calls_days_col is not None:
                try:
                    calls_days_series = df[calls_days_col].astype("object")
                except Exception:
                    calls_days_series = None
    if callsFile is None:
        lat_col = next((c for c in df.columns if c in ("lat","latitude")), None)
        lon_col = next((c for c in df.columns if c in ("lng","lon","long","longitude")), None)
        win_from_col = next((c for c in df.columns if any(x in c for x in ("openfrom","start","windowstart"))), None)
        win_to_col = next((c for c in df.columns if any(x in c for x in ("opento","end","windowend","finish"))), None)

    # Ensure coordinates exist, geocoding if needed
    if not (lat_col and lon_col):
        coords_list = []
        geocode_fail = 0
        for _, row in df.iterrows():
            lat, lng = geocode_postcode_norm(str(row[pc]))
            if np.isnan(lat) or np.isnan(lng):
                geocode_fail += 1
            coords_list.append((lat, lng))
        df["Latitude"], df["Longitude"] = zip(*coords_list)
        qa["geocoded_rows"] = len(coords_list)
        qa["geocode_failed_rows"] = geocode_fail
    else:
        df["Latitude"], df["Longitude"] = df[lat_col], df[lon_col]
    geo = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    if geo.empty:
        raise HTTPException(400, "No valid coordinates after geocoding.")

    coords = geo[["Latitude","Longitude"]].astype(float).to_numpy()
    # durations from selected column (calls file preferred)
    if dur_col is None or dur_col not in geo.columns:
        raise HTTPException(400, "Could not resolve duration column; supply calls file or map durationCol")
    durations = [int(parse_duration_to_seconds(v)) for v in geo[dur_col].tolist()]
    if any(d <= 0 for d in durations):
        qa["warnings"].append("Some durations are <= 0; treated as 0")
    # Optional resource depots: map Territory (name) -> (lat,lng)
    depots_map: dict[str, Tuple[float,float]] = {}
    res_schedule: dict[str, dict] = {}
    if resourcesFile is not None:
        try:
            rbuf = io.BytesIO(await resourcesFile.read())
            rdf = pd.read_excel(rbuf, engine="openpyxl").convert_dtypes()
            rcols = (
                rdf.columns.str.strip().str.lower()
                .str.replace(" ", "", regex=False)
                .str.replace("_", "", regex=False)
                .str.replace("-", "", regex=False)
            )
            rdf.columns = rcols
            def _norm_key(s: Optional[str]) -> Optional[str]:
                if not s:
                    return None
                return s.replace(" ", "").replace("_", "").replace("-", "").lower()
            name_col = _norm_key(resNameCol) or next((c for c in rdf.columns if any(x in c for x in ("name","resourcename","site","depot","base"))), None)
            latc = _norm_key(resLatCol) or next((c for c in rdf.columns if c in ("lat","latitude")), None)
            lngc = _norm_key(resLngCol) or next((c for c in rdf.columns if c in ("lng","lon","long","longitude")), None)
            pc_variants = ["postcode","post_code","postcode","postalcode","postal_code","zip","zipcode","eircode"]
            pc_norm2 = [v.replace(" ", "").replace("_", "").replace("-", "").lower() for v in pc_variants]
            pcc = _norm_key(resPostcodeCol) or next((c for c in rdf.columns if c in pc_norm2), None)
            daysc = _norm_key(resDaysCol) or next((k for k in rdf.columns if any(x in k for x in ("days","weekday","workdays","availability"))), None)
            startc = _norm_key(resStartCol) or next((k for k in rdf.columns if any(x in k for x in ("start","shiftstart","starttime"))), None)
            endc = _norm_key(resEndCol) or next((k for k in rdf.columns if any(x in k for x in ("end","shiftend","endtime","finish"))), None)

            for _, rr in rdf.iterrows():
                nm = str(rr.get(name_col, "")).strip() if name_col else ""
                if not nm:
                    continue
                lt = rr.get(latc) if latc else None
                lg = rr.get(lngc) if lngc else None
                latlng: Tuple[float,float] | None = None
                if (lt is not None) and (lg is not None):
                    try:
                        latlng = (float(lt), float(lg))
                    except Exception:
                        latlng = None
                elif pcc and str(rr.get(pcc, "")).strip():
                    glt, glg = geocode_postcode_norm(str(rr.get(pcc)))
                    if not (np.isnan(glt) or np.isnan(glg)):
                        latlng = (float(glt), float(glg))
                if latlng:
                    key = nm.strip().lower()
                    if key not in depots_map:
                        depots_map[key] = latlng
                # schedule
                try:
                    days = parse_weekdays(str(rr.get(daysc))) if daysc else []
                except Exception:
                    days = []
                try:
                    s_sec = int(parse_hhmm_to_seconds(rr.get(startc))) if startc else None
                except Exception:
                    s_sec = None
                try:
                    e_sec = int(parse_hhmm_to_seconds(rr.get(endc))) if endc else None
                except Exception:
                    e_sec = None
                if s_sec is not None and e_sec is not None:
                    if e_sec <= s_sec:
                        e_sec += 24*3600
                    res_schedule[nm.strip().lower()] = {
                        "start": s_sec,
                        "end": e_sec,
                        "days": days,
                        "latlng": depots_map.get(nm.strip().lower())
                    }
        except Exception:
            depots_map = {}
            res_schedule = {}
    # If no territory/group column, but we have resources with schedules, perform resource-centric assignment and scheduling across all calls.
    if terr_col is None and len(res_schedule) > 0:
        # Build resource list in stable order
        res_list: list[dict] = []
        for nm, sch in sorted(res_schedule.items(), key=lambda kv: str(kv[0])):
            latlng = sch.get("latlng")
            if not latlng:
                # try depots_map
                latlng = depots_map.get(nm)
            if not latlng:
                continue
            res_list.append({
                "name": nm,
                "lat": float(latlng[0]),
                "lng": float(latlng[1]),
                "days": list(sch.get("days", [])),
                "start": int(sch.get("start", 0)),
                "end": int(sch.get("end", 0)),
            })
        if not res_list:
            raise HTTPException(400, "Resources must include valid locations to route without territories")

        # Compute assignment to resources by open days and distance, balancing by weekly capacity
        centers_arr = np.array([[r["lat"], r["lng"]] for r in res_list], dtype=float)
        if int(fastMode or 0) == 1:
            # approximate durations via straight-line distance at 40km/h
            Dm = _haversine_matrix_between(coords, centers_arr)  # meters
            D_call_to_center_sec = (Dm / 1000.0) / 40.0 * 3600.0
        else:
            D_call_to_center_sec = osrm_matrix_between_duration(coords, centers_arr)
        n_calls, k_centers = D_call_to_center_sec.shape
        # Call open days
        calls_open_days: List[List[int]] = []
        for i in range(len(geo)):
            if calls_days_series is not None:
                try:
                    calls_open_days.append(parse_weekdays(str(calls_days_series.iloc[i])))
                except Exception:
                    calls_open_days.append(list(range(7)))
            else:
                calls_open_days.append(list(range(7)))
        # Allowed per call
        allowed: List[List[int] | None] = []
        for i in range(n_calls):
            cd = set(int(d) for d in calls_open_days[i])
            cand = [j for j, r in enumerate(res_list) if cd.intersection(set(r["days"]))]
            allowed.append(cand if cand else None)
        # Weekly capacity quotas
        cap_sec = [sum((r["end"] - r["start"]) for _d in r["days"]) for r in res_list]
        total_cap = max(1, sum(max(0, c) for c in cap_sec))
        quotas = [max(0, int(round(n_calls * (max(0, c) / total_cap)))) for c in cap_sec]
        diff = n_calls - sum(quotas)
        if diff != 0:
            order = np.argsort([-c for c in cap_sec])
            if diff < 0:
                order = np.argsort([c for c in cap_sec])
            idx = 0
            while diff != 0 and len(order) > 0:
                j = int(order[idx % len(order)])
                if diff > 0:
                    quotas[j] += 1; diff -= 1
                else:
                    if quotas[j] > 0:
                        quotas[j] -= 1; diff += 1
                idx += 1
        # Assignment
        def _assign_with_flow_or_greedy(D: np.ndarray, quotas: List[int], allowed: List[Optional[List[int]]]) -> np.ndarray:
            n, k = D.shape
            if _HAS_FLOW:
                # Inline min-cost flow with allowed set
                nearest = np.min(D, axis=1)
                start_indices: list[int] = []
                end_indices: list[int] = []
                capacities: list[int] = []
                unit_costs: list[int] = []
                source = k + n
                sink = source + 1
                num_nodes = sink + 1
                supplies = [0] * num_nodes
                supplies[source] = n
                supplies[sink] = -n
                for j in range(k):
                    start_indices.append(source); end_indices.append(j); capacities.append(int(quotas[j])); unit_costs.append(0)
                for j in range(k):
                    for i in range(n):
                        if allowed[i] is not None and j not in allowed[i]:
                            continue
                        start_indices.append(j); end_indices.append(k + i); capacities.append(1)
                        base = float(D[i, j])
                        penalty = 0.0
                        if nearest[i] > 0:
                            ratio = base / nearest[i]
                            if ratio > FLOW_ALPHA:
                                penalty = (ratio - FLOW_ALPHA) * FLOW_GAMMA * nearest[i]
                        c = int(max(1, round(base + penalty)))
                        unit_costs.append(c)
                for i in range(n):
                    start_indices.append(k + i); end_indices.append(sink); capacities.append(1); unit_costs.append(0)
                mcf = _new_mcf()
                def _add_arc(m,u,v,cap,cost):
                    if hasattr(m,'AddArcWithCapacityAndUnitCost'):
                        m.AddArcWithCapacityAndUnitCost(u,v,int(cap),int(cost))
                    else:
                        m.add_arc_with_capacity_and_unit_cost(u,v,int(cap),int(cost))
                def _set_supply(m,node,sup):
                    if hasattr(m,'SetNodeSupply'):
                        m.SetNodeSupply(node,int(sup))
                    else:
                        m.set_node_supply(node,int(sup))
                def _solve(m):
                    return m.Solve() if hasattr(m,'Solve') else m.solve()
                def _num_arcs(m):
                    return m.NumArcs() if hasattr(m,'NumArcs') else m.num_arcs()
                def _tail(m,a):
                    return m.Tail(a) if hasattr(m,'Tail') else m.tail(a)
                def _head(m,a):
                    return m.Head(a) if hasattr(m,'Head') else m.head(a)
                def _flow(m,a):
                    return m.Flow(a) if hasattr(m,'Flow') else m.flow(a)
                for u,v,cap,cost in zip(start_indices,end_indices,capacities,unit_costs):
                    _add_arc(mcf,u,v,cap,cost)
                for node,sup in enumerate(supplies):
                    _set_supply(mcf,node,sup)
                status = _solve(mcf)
                optimal_code = getattr(mcf,'OPTIMAL',0)
                if status != optimal_code:
                    # Fallback to greedy
                    pass
                else:
                    assign = -np.ones(n, dtype=int)
                    for a in range(_num_arcs(mcf)):
                        u = _tail(mcf,a); v = _head(mcf,a)
                        if 0 <= u < k and k <= v < k + n:
                            if _flow(mcf,a) > 0:
                                assign[v - k] = u
                    if (assign >= 0).all():
                        return assign
            # Greedy nearest with quotas and allowed
            assign = -np.ones(n, dtype=int)
            cap = quotas[:]
            order_calls = list(range(n))
            order_calls.sort(key=lambda i: float(np.min(D[i])))
            for i in order_calls:
                order_centers = list(np.argsort(D[i]).astype(int))
                for j in order_centers:
                    if allowed[i] is not None and j not in allowed[i]:
                        continue
                    if cap[j] > 0:
                        assign[i] = j
                        cap[j] -= 1
                        break
            for i in range(n):
                if assign[i] == -1:
                    j = int(np.argmin([D[i, jj] if (allowed[i] is None or jj in allowed[i]) else 1e18 for jj in range(k)]))
                    assign[i] = j
            return assign
        assign = _assign_with_flow_or_greedy(D_call_to_center_sec, quotas, allowed)

        # Contiguity smoothing: reduce spatial interleaving by penalizing neighbor label mismatches
        def _smooth_contiguity(assign: np.ndarray,
                               D_call_center: np.ndarray,
                               coords: np.ndarray,
                               allowed: List[Optional[List[int]]],
                               quotas: List[int],
                               k_neighbors: int = 8,
                               lam_seconds: int = 600,
                               passes: int = 2) -> np.ndarray:
            try:
                n, k = D_call_center.shape
                assign = assign.copy().astype(int)
                # Precompute k-nearest neighbors by haversine (fast and dependency-free)
                Dcc = _haversine_matrix_full(coords)  # meters
                nn_list: List[np.ndarray] = []
                for i in range(n):
                    row = Dcc[i]
                    # exclude self (set to large)
                    row = row.copy()
                    row[i] = 1e18
                    if k_neighbors >= len(row) - 1:
                        nn = np.argsort(row)
                    else:
                        idx = np.argpartition(row, k_neighbors)[:k_neighbors]
                        nn = idx[np.argsort(row[idx])]
                    nn_list.append(nn)

                counts = np.bincount(assign, minlength=D_call_center.shape[1]).astype(int)
                # soft headroom to avoid overfilling vs quotas
                headroom = max(1, int(np.ceil(n / max(1, k) * 0.2)))  # 20% per center
                for _ in range(max(1, passes)):
                    changed = 0
                    for i in range(n):
                        curr = int(assign[i])
                        # compute current score
                        neigh = nn_list[i]
                        mismatch = int(np.sum(assign[neigh] != curr))
                        best_score = float(D_call_center[i, curr]) + lam_seconds * float(mismatch)
                        best_j = curr
                        # candidate resources
                        cand_js = allowed[i] if (allowed[i] is not None and len(allowed[i]) > 0) else list(range(k))
                        for j in cand_js:
                            j = int(j)
                            if j == curr: continue
                            # rough quota respect
                            if counts[j] >= quotas[j] + headroom:
                                continue
                            mismatch_j = int(np.sum(assign[neigh] != j))
                            score_j = float(D_call_center[i, j]) + lam_seconds * float(mismatch_j)
                            if score_j + 1e-6 < best_score:
                                best_score = score_j
                                best_j = j
                        if best_j != curr:
                            counts[curr] -= 1
                            counts[best_j] += 1
                            assign[i] = best_j
                            changed += 1
                    if changed == 0:
                        break
                return assign
            except Exception:
                return assign

    # Stronger smoothing to reduce cross-territory interleaving
    assign = _smooth_contiguity(assign, D_call_to_center_sec, coords, allowed, quotas, k_neighbors=12, lam_seconds=2400, passes=3)

    # Prepare output rows structure
    geo_out = geo.copy()
    geo_out["Territory"] = ""
    geo_out["Resource"] = ""
    geo_out["Day"] = ""
    geo_out["Sequence"] = 0
    geo_out["StartTime"] = ""
    geo_out["EndTime"] = ""
    geo_out["DepartTime"] = ""
    geo_out["ReturnTime"] = ""
    geo_out["DayOfWeek"] = ""
    geo_out["Week"] = 0
    # Friendly call name column if present
    if "CallName" not in geo_out.columns:
        # try to surface a best-effort name/id
        cand = next((c for c in geo_out.columns if any(x in c.lower() for x in ("name","id","customer","account","site","call"))), None)
        if cand:
            geo_out = geo_out.rename(columns={cand: "CallName"})

    def _fmt_local(sec: int) -> str:
        h = int(sec // 3600); m = int((sec % 3600)//60)
        return f"{h:02d}:{m:02d}"

    remaining = {int(i) for i in range(len(geo))}
    calls_by_res: dict[int, List[int]] = {}
    for i, j in enumerate(assign):
        calls_by_res.setdefault(int(j), []).append(int(i))

    TL = int(timeLimitSec or 20)
    week_used = 0
    days_total = 0
    # Continuous Day counter per resource (across weeks)
    day_counter_map: dict[int, int] = {int(k): 0 for k in calls_by_res.keys()}
    while len(remaining) > 0 and week_used < int(maxWeeks or 8):
        week_used += 1
        progress = False
        for r_idx in sorted(calls_by_res.keys()):
            idxs = calls_by_res[r_idx]
            r = res_list[r_idx]
            depot = (r["lat"], r["lng"]) 
            for d in r["days"]:
                elig = [i for i in idxs if i in remaining and d in set(calls_open_days[i])]
                if not elig:
                    continue
                sub_coords = coords[elig]
                sub_serv = [int(durations[i]) for i in elig]
                visited_rel, _dropped = vrp_single_route_with_time(
                    sub_coords, depot, sub_serv, int(r["start"]), int(r["end"]), time_limit_sec=TL, max_stops=None, seed=seed_int
                )
                if not visited_rel:
                    continue
                progress = True
                # increment continuous counter for this resource
                day_counter_map[int(r_idx)] = int(day_counter_map.get(int(r_idx), 0) + 1)
                all_pts = np.vstack([sub_coords, np.array([[depot[0], depot[1]]])])
                if int(fastMode or 0) == 1:
                    dm_route = _haversine_matrix_full(all_pts) / 1000.0 / 40.0 * 3600.0
                else:
                    dm_route = osrm_table_batch_duration(all_pts)
                prev = len(sub_coords)
                start_t = int(r["start"]) 
                # Record a constant depart time for this day (resource shift start)
                depart_time_str = _fmt_local(start_t)
                last_abs_i: Optional[int] = None
                for seq, rel in enumerate(visited_rel, start=1):
                    abs_i = int(elig[rel])
                    travel = int(dm_route[prev, rel])
                    service = int(durations[abs_i])
                    s = start_t + travel
                    e = s + service
                    res_name = str(r["name"]) or f"Resource {r_idx+1}"
                    geo_out.at[abs_i, "Territory"] = res_name
                    geo_out.at[abs_i, "Resource"] = res_name
                    geo_out.at[abs_i, "Day"] = f"Day {day_counter_map[int(r_idx)]}"
                    geo_out.at[abs_i, "DayOfWeek"] = _DAYS[d]
                    geo_out.at[abs_i, "Week"] = int(week_used)
                    geo_out.at[abs_i, "Sequence"] = int(seq)
                    geo_out.at[abs_i, "StartTime"] = _fmt_local(s)
                    geo_out.at[abs_i, "EndTime"] = _fmt_local(e)
                    geo_out.at[abs_i, "DepartTime"] = depart_time_str
                    start_t = e
                    prev = rel
                    last_abs_i = abs_i
                    if abs_i in remaining:
                        remaining.remove(abs_i)
                days_total += 1
                # After finishing this day's route, compute return to depot
                if last_abs_i is not None:
                    return_s = int(start_t) + int(dm_route[prev, len(sub_coords)])
                    geo_out.at[last_abs_i, "ReturnTime"] = _fmt_local(return_s)
        if not progress:
            # No assignments made this week; break to avoid spinning
            break
    # Derive DayOfWeekNum, DayNumber, CallsPerDay (DayNumber continuous, independent of Week/DayOfWeek)
    dow_map = {
        "mon": 1, "monday": 1,
        "tue": 2, "tues": 2, "tuesday": 2,
        "wed": 3, "weds": 3, "wednesday": 3,
        "thu": 4, "thur": 4, "thurs": 4, "thursday": 4,
        "fri": 5, "friday": 5,
        "sat": 6, "saturday": 6,
        "sun": 7, "sunday": 7,
    }
    def _dow_num(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip().lower()
        return dow_map.get(s, np.nan)
    geo_out["DayOfWeekNum"] = geo_out.get("DayOfWeek").map(_dow_num) if "DayOfWeek" in geo_out.columns else np.nan
    # DayNumber strictly from the Day label (Day 1..N), continuous per Resource across weeks
    day_label_num = pd.to_numeric(
        geo_out["Day"].astype(str).str.extract(r"(\d+)").iloc[:, 0], errors="coerce"
    )
    geo_out["DayNumber"] = day_label_num
    # Calls per day grouped by Resource and DayNumber (ignore Week)
    cts = (
        geo_out.groupby(["Resource", "DayNumber"]).size()
        .rename("CallsPerDay").reset_index()
    )
    geo_out = geo_out.merge(cts, on=["Resource", "DayNumber"], how="left")

    # Stable numeric ranks and sort key
    res_names = sorted([str(x) for x in geo_out["Resource"].dropna().unique().tolist()])
    res_rank = {name: i + 1 for i, name in enumerate(res_names)}
    geo_out["ResourceRank"] = geo_out["Resource"].map(lambda x: res_rank.get(str(x), 0))
    geo_out["SortOrder"] = (
        geo_out["ResourceRank"].astype(int) * 10_000_000
        + geo_out["DayNumber"].fillna(0).astype(int) * 100
        + geo_out["Sequence"].fillna(0).astype(int)
    )

    # Prepare rows
    base_cols = [
        "Resource", "Territory",
        "Week", "DayOfWeek", "DayOfWeekNum", "Day", "DayNumber", "CallsPerDay",
        "Sequence", "StartTime", "EndTime", "DepartTime", "ReturnTime", "Latitude", "Longitude",
        "ResourceRank", "SortOrder", "CallName"
    ]
    other_cols = [c for c in geo_out.columns if c not in base_cols]
    excelCols = base_cols + other_cols
    geo_out = geo_out.sort_values(["SortOrder"])  # stable presentation
    geo_excel = geo_out[excelCols].astype("object").where(pd.notna(geo_out[excelCols]), "").astype(str)
    rows = [excelCols] + geo_excel.values.tolist()
    return JSONResponse({
        "rows": rows,
        "route_kpis": [],
        "message": f"Built {int(days_total)} day route(s) using resources without territories.",
        "run_id": run_id,
        "seed": seed_int,
        "deterministic": True,
        "qa": qa,
    })
    # If we still don't have a territory and no resources, fail clearly
    if terr_col is None and len(res_schedule) == 0:
        raise HTTPException(400, "Provide either a Territory/Group column or a Resources file with schedules.")
    # Optional time window columns (OpenFrom/OpenTo or similar)
    win_from = win_from_col
    win_to = win_to_col
    windows: Optional[List[Tuple[int,int]]] = None
    if win_from and win_to:
        wf = [parse_hhmm_to_seconds(v) for v in geo[win_from].tolist()]
        wt = [parse_hhmm_to_seconds(v) for v in geo[win_to].tolist()]
        windows = [(int(max(0, a)), int(max(0, b if b > a else b + 24*3600))) for a, b in zip(wf, wt)]
    default_work_sec = int(max(1, int(workDayMinutes or 480)) * 60)
    TL = int(timeLimitSec or 20)

    # Prepare output structure
    geo_out = geo.copy()
    geo_out["Day"] = ""
    geo_out["Sequence"] = 0
    geo_out["StartTime"] = ""
    geo_out["EndTime"] = ""
    geo_out["DepartTime"] = ""
    geo_out["ReturnTime"] = ""
    geo_out["DayOfWeek"] = ""
    geo_out["Week"] = 0
    # Friendly call name column if present
    if "CallName" not in geo_out.columns:
        cand = next((c for c in geo_out.columns if any(x in c.lower() for x in ("name","id","customer","account","site","call"))), None)
        if cand:
            geo_out = geo_out.rename(columns={cand: "CallName"})

    route_kpis = []
    def fmt(sec: int) -> str:
        h = int(sec // 3600); m = int((sec % 3600)//60)
        return f"{h:02d}:{m:02d}"

    # For each territory
    days_total = 0
    # Build a map of fixed depots if the territory names look like resource names with lat/lng present
    # If the sheet already includes a consistent Territory naming from resources, use medoid fallback only when no hint exists.
    for terr in geo_out[terr_col].unique():
        mask = (geo_out[terr_col] == terr).to_numpy()
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        sub_coords = coords[idxs]
        sub_durs = [durations[i] for i in idxs]
        # Determine depot: prefer fixed within territory if many points cluster or territory label hints; else strategy
        # Prefer strict depot from resources map by territory label
        terr_key = str(terr).strip().lower()
        # depot and schedule from resources if name matches territory; else fallback
        terr_sched = res_schedule.get(terr_key)
        if terr_key in depots_map:
            depot = depots_map[terr_key]
        elif terr_sched and terr_sched.get("latlng"):
            depot = terr_sched["latlng"]  # type: ignore
        else:
            depot = _compute_depot_for_territory(sub_coords, depotStrategy or "medoid")

        # Precompute matrices once per territory
        all_pts = np.vstack([sub_coords, np.array([[depot[0], depot[1]]])])
        dm = osrm_table_batch_duration(all_pts)
        dd = osrm_table_batch(all_pts)

        # If we have resource schedules (days + start/end), schedule per weekday with call availability when provided
        if terr_sched and (terr_sched.get("start") is not None) and (terr_sched.get("end") is not None) and len(terr_sched.get("days", [])) > 0:
            remaining = set(int(i) for i in idxs)
            week_used = 0
            max_weeks_eff = int(maxWeeks or 8)
            day_counter = 0
            # Pre-parse calls open days per absolute index if available
            def _call_open_days_for(abs_i: int) -> list[int]:
                if calls_days_series is None:
                    return list(range(7))  # no restriction
                try:
                    return parse_weekdays(str(calls_days_series.iloc[abs_i]))
                except Exception:
                    return list(range(7))
            while len(remaining) > 0 and week_used < max(1, max_weeks_eff):
                week_used += 1
                for d in terr_sched["days"]:
                    # Eligible by availability
                    elig_abs = [a for a in list(remaining) if a in idxs and d in _call_open_days_for(a)]
                    if not elig_abs:
                        continue
                    # Subset data for elig
                    sub_coords_w = coords[elig_abs]
                    sub_serv_w = [int(durations[a]) for a in elig_abs]
                    visited_rel, _dropped = vrp_single_route_with_time(
                        sub_coords_w, depot, sub_serv_w, int(terr_sched["start"]), int(terr_sched["end"]), time_limit_sec=TL, max_stops=None, seed=seed_int
                    )
                    if not visited_rel:
                        continue
                    # Map back and write outputs
                    day_counter += 1
                    all_pts_r = np.vstack([sub_coords_w, np.array([[depot[0], depot[1]]])])
                    dm_route = osrm_table_batch_duration(all_pts_r)
                    prev = len(sub_coords_w)
                    start_t = int(terr_sched["start"])
                    drive_acc = 0
                    service_acc = 0
                    for seq, rel in enumerate(visited_rel, start=1):
                        abs_i = int(elig_abs[rel])
                        travel = int(dm_route[prev, rel])
                        service = int(durations[abs_i])
                        s = start_t + travel
                        e = s + service
                        geo_out.at[abs_i, "Day"] = f"Day {day_counter}"
                        geo_out.at[abs_i, "DayOfWeek"] = _DAYS[d]
                        geo_out.at[abs_i, "Week"] = int(week_used)
                        geo_out.at[abs_i, "Sequence"] = int(seq)
                        geo_out.at[abs_i, "StartTime"] = fmt(s)
                        geo_out.at[abs_i, "EndTime"] = fmt(e)
                        start_t = e
                        prev = rel
                        drive_acc += travel
                        service_acc += service
                        if abs_i in remaining:
                            remaining.remove(abs_i)
                    # KPI for this daily route
                    sel_coords = coords[[int(elig_abs[r]) for r in visited_rel]]
                    drive_min, _svc_min_unused, km = _route_kpis(sel_coords, depot)
                    route_kpis.append({
                        "territory": str(terr),
                        "day": int(day_counter),
                        "drive_minutes": round(drive_min, 1),
                        "service_minutes": round(service_acc/60.0, 1),
                        "total_km": round(km, 2),
                        "visits": int(len(visited_rel))
                    })
                    days_total += 1
        else:
            # Work seconds per day: use resource schedule if available, else default
            if terr_sched and (terr_sched.get("start") is not None) and (terr_sched.get("end") is not None):
                work_sec = int(terr_sched["end"]) - int(terr_sched["start"])  # type: ignore
            else:
                work_sec = default_work_sec

            if VRP_TIME_BUDGET_ENABLED:
                # Multi-vehicle time-budget split using durations
                routes_rel = vrp_routes_time_budget(
                    sub_coords, sub_durs, depot, work_sec, time_limit_sec=TL,
                    durations_matrix=dm, windows=[windows[i] for i in idxs] if windows else None,
                    priorities=None, break_minutes=int(breakMinutes or 0), max_route_minutes=maxRouteMinutes if maxRouteMinutes else None, seed=seed_int
                )
                # Map to absolute indices
                routes = [[idxs[r] for r in route] for route in routes_rel]
            else:
                # Fallback: simple heuristic using durations-aware CVRP cap approximation
                max_calls = max(1, int(np.floor((work_sec / 60.0) / 60.0)))
                routes_rel = cvrp_days_for_territory(sub_coords, depot, min_calls=0, max_calls=max_calls, time_sec=TL)
                routes = [[idxs[r] for r in route] for route in routes_rel] if routes_rel else []

            # Map routes back to absolute indices if needed
            abs_routes: List[List[int]] = routes

            # Write outputs and compute KPIs
            for di, route_abs in enumerate(abs_routes, start=1):
                if not route_abs:
                    continue
                # Times: approximate cumulative starting at 00:00 using travel+service
                all_pts = np.vstack([coords[route_abs], np.array([[depot[0], depot[1]]])])
                dm_route = osrm_table_batch_duration(all_pts)
                # If we have a resource start time, start at that absolute time; else 00:00
                start_t = int(terr_sched["start"]) if terr_sched and terr_sched.get("start") is not None else 0
                depart_time_str = fmt(start_t)
                prev = len(route_abs)  # depot index
                drive_acc = 0
                service_acc = 0
                for seq, abs_i in enumerate(route_abs, start=1):
                    travel = int(dm_route[prev, seq-1])
                    service = int(durations[abs_i])
                    s = start_t + travel
                    e = s + service
                    geo_out.at[abs_i, "Day"] = f"Day {di}"
                    geo_out.at[abs_i, "Sequence"] = int(seq)
                    geo_out.at[abs_i, "StartTime"] = fmt(s)
                    geo_out.at[abs_i, "EndTime"] = fmt(e)
                    geo_out.at[abs_i, "DepartTime"] = depart_time_str
                    start_t = e
                    prev = seq-1
                    drive_acc += travel
                    service_acc += service
                # Return to depot time on the last stop of the day
                if len(route_abs) > 0:
                    last_abs = route_abs[-1]
                    return_s = int(start_t) + int(dm_route[len(route_abs)-1, len(route_abs)])
                    geo_out.at[last_abs, "ReturnTime"] = fmt(return_s)
                # KPI for this route
                drive_min, _svc_min_unused, km = _route_kpis(coords[route_abs], depot)
                route_kpis.append({
                    "territory": str(terr),
                    "day": int(di),
                    "drive_minutes": round(drive_min, 1),
                    "service_minutes": round(service_acc/60.0, 1),
                    "total_km": round(km, 2),
                    "visits": int(len(route_abs))
                })
                days_total += 1

    # Derive DayOfWeekNum, DayNumber and CallsPerDay (DayNumber continuous regardless of Week)
    dow_map = {
        "mon": 1, "monday": 1,
        "tue": 2, "tues": 2, "tuesday": 2,
        "wed": 3, "weds": 3, "wednesday": 3,
        "thu": 4, "thur": 4, "thurs": 4, "thursday": 4,
        "fri": 5, "friday": 5,
        "sat": 6, "saturday": 6,
        "sun": 7, "sunday": 7,
    }
    def _dow_num2(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip().lower()
        return dow_map.get(s, np.nan)
    geo_out["DayOfWeekNum"] = geo_out.get("DayOfWeek").map(_dow_num2) if "DayOfWeek" in geo_out.columns else np.nan
    def _day_num2(v):
        try:
            import re as _re
            m = _re.search(r"(\d+)", str(v))
            return int(m.group(1)) if m else None
        except Exception:
            return None
    geo_out["DayNumber"] = geo_out["Day"].map(_day_num2)
    cts = geo_out.groupby([terr_col, "DayNumber"]).size().rename("CallsPerDay").reset_index()
    geo_out = geo_out.merge(cts, on=[terr_col, "DayNumber"], how="left")

    # Stable numeric ranks and sort key
    terr_names = sorted([str(x) for x in geo_out[terr_col].dropna().unique().tolist()])
    terr_rank = {name: i + 1 for i, name in enumerate(terr_names)}
    geo_out["TerritoryRank"] = geo_out[terr_col].map(lambda x: terr_rank.get(str(x), 0))
    geo_out["SortOrder"] = (
        geo_out["TerritoryRank"].astype(int) * 10_000_000
        + geo_out["DayNumber"].fillna(0).astype(int) * 100
        + geo_out["Sequence"].fillna(0).astype(int)
    )

    # Prepare output rows
    base_cols = [
        terr_col,
        "Week", "DayOfWeek", "DayOfWeekNum", "Day", "DayNumber", "CallsPerDay",
        "Sequence", "StartTime", "EndTime", "DepartTime", "ReturnTime", "Latitude", "Longitude",
        "TerritoryRank", "SortOrder", "CallName"
    ]
    other_cols = [c for c in geo_out.columns if c not in base_cols]
    excelCols = base_cols + other_cols
    geo_out = geo_out.sort_values(["SortOrder"])  # stable presentation
    geo_excel = geo_out[excelCols].astype("object").where(pd.notna(geo_out[excelCols]), "").astype(str)
    rows = [excelCols] + geo_excel.values.tolist()

    return JSONResponse({
        "rows": rows,
        "route_kpis": route_kpis,
        "message": f"Built {int(days_total)} day route(s) using {'time budget' if VRP_TIME_BUDGET_ENABLED else 'count heuristic'}.",
        "run_id": run_id,
        "seed": seed_int,
        "deterministic": True,
        "qa": qa,
    })


# ----------------------------
# Health endpoint (optional)
# ----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


# ----------------------------
# Health: OSRM connectivity check (no fallback)
# ----------------------------
@app.get("/healthz/osrm")
def healthz_osrm():
    try:
        # Two nearby points in London
        coords = np.array([[51.5074, -0.1278], [51.5007, -0.1246]], dtype=float)
        M = osrm_table_batch(coords)
        dist = float(M[0, 1]) if M.shape == (2, 2) else None
        osrm_url = _get_osrm_url()
        return {"ok": True, "osrm_url": osrm_url, "distance_sample_m": dist}
    except Exception as e:
        osrm_url = _get_osrm_url()
        return JSONResponse({"ok": False, "osrm_url": osrm_url, "error": str(e)}, status_code=503)


@app.get("/api/debug-config")
def debug_config():
    """Return runtime territory planning configuration for debugging/tuning."""
    return {
        "H3_AVAILABLE": _H3_AVAILABLE,
        "H3_RES": H3_RES,
        "H3_PLAN_RES": H3_PLAN_RES,
        "H3_LAMBDA": H3_LAMBDA,
        "USE_CELL_FLOW": bool(USE_CELL_FLOW),
        "FLOW_TOPK": FLOW_TOPK,
        "FLOW_STRICT": bool(FLOW_STRICT),
        "SMOOTH_ITERS": SMOOTH_ITERS,
        "CORRIDOR_ON": bool(CORRIDOR_ON),
        "CORRIDOR_K": CORRIDOR_K,
        "CORRIDOR_BETA": CORRIDOR_BETA,
    }


# ----------------------------
# Re-optimise a single day within each territory/resource
# ----------------------------
@app.post("/api/reopt/day")
async def reopt_day(request: Request):
    """Re-optimise routing sequence and times for a selected DayNumber within each group (Resource or Territory).
    Body JSON:
      - rows: 2D array [header, ...rows]
      - day: integer DayNumber to re-optimise
      - groupCol: optional name of the grouping column (defaults to 'Resource' if present else 'Territory')
      - depotStrategy: optional ('medoid'|'centroid'), default 'medoid'
      - startTime: optional HH:MM to use as depart time when missing in table (default 08:00)
      - endTime: optional HH:MM work end (default start+9h)
    Returns: updated rows array with Sequence/Start/End/Depart/Return times recomputed for that day.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    rows = body.get("rows")
    if not rows or not isinstance(rows, list) or not rows[0]:
        raise HTTPException(400, "rows required: [header,...]")
    try:
        day_num = int(body.get("day"))
    except Exception:
        raise HTTPException(400, "day (DayNumber) integer required")
    depot_strategy = str(body.get("depotStrategy") or "medoid").strip().lower()
    start_time_hint = str(body.get("startTime") or "08:00").strip()
    end_time_hint = str(body.get("endTime") or "").strip()
    header = [str(h) for h in rows[0]]

    # Helper: find column by regex
    import re as _re
    def _find_col(pats: list[str]) -> str | None:
        for p in pats:
            rx = _re.compile(p, _re.I)
            for h in header:
                if rx.search(h):
                    return h
        return None

    # Key columns
    lat_col = _find_col([r"^lat(itude)?$"])
    lon_col = _find_col([r"^(lon|lng|long|longitude)$"])
    if not lat_col or not lon_col:
        raise HTTPException(400, "Latitude/Longitude columns required in rows")
    group_col = body.get("groupCol")
    if not group_col or group_col not in header:
        group_col = next((c for c in ("Resource", "Territory", "Group") if c in header), None)
        if not group_col:
            raise HTTPException(400, "groupCol not provided and no Resource/Territory column found")
    daynum_col = next((c for c in header if _re.search(r"^day\s*number$", c, _re.I)), None)
    day_col = next((c for c in header if _re.fullmatch(r"day", c, _re.I)), None)
    seq_col = "Sequence" if "Sequence" in header else None
    st_col = "StartTime" if "StartTime" in header else None
    et_col = "EndTime" if "EndTime" in header else None
    dp_col = "DepartTime" if "DepartTime" in header else None
    rt_col = "ReturnTime" if "ReturnTime" in header else None
    # Optional duration column
    dur_col = next((c for c in header if _re.search(r"duration|mins|minutes|service\s*time", c, _re.I)), None)

    # Index maps
    idx_map = {h: i for i, h in enumerate(header)}
    data = rows[1:]

    # Build subset mask: DayNumber == day_num (fall back to digits in Day)
    def _row_day_number(r: list) -> int | None:
        if daynum_col and daynum_col in idx_map:
            try:
                v = r[idx_map[daynum_col]]
                n = int(str(v))
                return n
            except Exception:
                pass
        if day_col and day_col in idx_map:
            m = _re.search(r"(\d+)", str(r[idx_map[day_col]]))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    # Parse start/end time defaults
    try:
        default_start = parse_hhmm_to_seconds(start_time_hint)
    except Exception:
        default_start = 8*3600
    if end_time_hint:
        try:
            default_end = parse_hhmm_to_seconds(end_time_hint)
            if default_end <= default_start:
                default_end += 24*3600
        except Exception:
            default_end = default_start + 9*3600
    else:
        default_end = default_start + 9*3600

    # Prepare editing in place
    def _fmt(sec: int) -> str:
        h = int(sec // 3600); m = int((sec % 3600)//60)
        return f"{h:02d}:{m:02d}"

    # Group by group_col among matching day
    # Collect absolute indices relative to data (1-based in rows)
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(data):
        dn = _row_day_number(r)
        if dn != day_num:
            continue
        gval = str(r[idx_map[group_col]] or "").strip()
        if not gval:
            gval = "(Unassigned)"
        groups[gval].append(i)

    if not groups:
        raise HTTPException(400, f"No rows found for DayNumber={day_num}")

    # For each group, compute new sequence and times
    for gval, idxs in groups.items():
        # Clear any stale ReturnTime values for this day's group; we'll set only on the last stop
        if rt_col:
            for di in idxs:
                try:
                    data[di][idx_map[rt_col]] = ""
                except Exception:
                    pass
        # coords and durations
        coords = []
        service = []
        for di in idxs:
            row = data[di]
            try:
                lat = float(row[idx_map[lat_col]])
                lng = float(row[idx_map[lon_col]])
            except Exception:
                continue
            coords.append([lat, lng])
            if dur_col and dur_col in idx_map:
                try:
                    service.append(int(parse_duration_to_seconds(row[idx_map[dur_col]])))
                except Exception:
                    service.append(0)
            else:
                service.append(0)
        if len(coords) <= 1:
            # Trivial: just resequence to 1 and set times to depart only
            if seq_col:
                for j, di in enumerate(idxs, start=1):
                    data[di][idx_map[seq_col]] = str(j)
            if dp_col:
                for di in idxs:
                    data[di][idx_map[dp_col]] = _fmt(default_start)
            continue
        sub = np.array(coords, dtype=float)
        # Choose depot by strategy
        try:
            depot_lat, depot_lng = _compute_depot_for_territory(sub, strategy=depot_strategy)
        except Exception:
            # default to centroid
            depot_lat = float(np.mean(sub[:,0])); depot_lng = float(np.mean(sub[:,1]))
        # Start time: prefer existing DepartTime on this day if consistent
        start_sec = default_start
        if dp_col and dp_col in idx_map:
            # take mode of DepartTime strings in this day
            vals = [str(data[di][idx_map[dp_col]] or "").strip() for di in idxs]
            vals = [v for v in vals if v]
            if vals:
                try:
                    # pick most common
                    from collections import Counter
                    com = Counter(vals).most_common(1)[0][0]
                    ssec = parse_hhmm_to_seconds(com)
                    if isinstance(ssec, int) and ssec >= 0:
                        start_sec = ssec
                except Exception:
                    pass
        end_sec = default_end if default_end > start_sec else start_sec + 9*3600

        # Solve route order with time budget to respect a working window
        visited_rel, _dropped = vrp_single_route_with_time(
            sub, (depot_lat, depot_lng), service, int(start_sec), int(end_sec), time_limit_sec=10, max_stops=None, seed=42
        )
        if not visited_rel:
            # Fallback to original order
            order = list(range(len(idxs)))
        else:
            order = list(visited_rel)
        # Build durations matrix for time calc
        all_pts = np.vstack([sub, np.array([[depot_lat, depot_lng]])])
        try:
            dm_route = osrm_table_batch_duration(all_pts)
        except Exception:
            # fallback straight-line at 40km/h
            dm_route = _haversine_matrix_full(all_pts) / 1000.0 / 40.0 * 3600.0
        prev = len(sub)
        t = int(start_sec)
        # Write DepartTime uniformly for the day
        if dp_col:
            for di in idxs:
                data[di][idx_map[dp_col]] = _fmt(start_sec)
        last_abs_idx = None
        # Map back: idxs is list of data indices; order contains relative positions into sub
        for seq, rel in enumerate(order, start=1):
            abs_data_idx = idxs[int(rel)]
            travel = int(dm_route[prev, rel])
            svc = int(service[int(rel)]) if int(rel) < len(service) else 0
            s = t + travel; e = s + svc
            if seq_col:
                data[abs_data_idx][idx_map[seq_col]] = str(seq)
            if st_col:
                data[abs_data_idx][idx_map[st_col]] = _fmt(s)
            if et_col:
                data[abs_data_idx][idx_map[et_col]] = _fmt(e)
            t = e
            prev = rel
            last_abs_idx = abs_data_idx
        # Return time on the last visited
        if last_abs_idx is not None:
            try:
                ret_s = int(t) + int(dm_route[prev, len(sub)])
            except Exception:
                ret_s = int(t)
            if rt_col:
                data[last_abs_idx][idx_map[rt_col]] = _fmt(ret_s)

        # Final safety: resequence 1..N within this (group, day) by StartTime ascending
        if seq_col and st_col and st_col in idx_map:
            def _sec(v: str) -> int:
                try:
                    return int(parse_hhmm_to_seconds(str(v)))
                except Exception:
                    return 10**9
            # Pair (sec, data_idx) and sort
            pairs = [(_sec(data[di][idx_map[st_col]]), di) for di in idxs]
            pairs.sort(key=lambda x: x[0])
            for new_seq, (_, di) in enumerate(pairs, start=1):
                data[di][idx_map[seq_col]] = str(new_seq)

    # Optional: recompute SortOrder if available
    try:
        if "SortOrder" in header:
            rn_col = "ResourceRank" if "ResourceRank" in header else ("TerritoryRank" if "TerritoryRank" in header else None)
            so_idx = idx_map["SortOrder"]
            # derive DayNumber from Day label if possible
            import re as _re
            def _dn_from_day(v: str) -> int:
                m = _re.search(r"(\d+)", str(v))
                return int(m.group(1)) if m else 0
            dn_idx = idx_map.get(daynum_col) if daynum_col else None
            for i, r in enumerate(data):
                try:
                    rn = int(str(r[idx_map[rn_col]])) if rn_col and r[idx_map[rn_col]] not in (None, "") else 0
                except Exception:
                    rn = 0
                try:
                    dn = int(str(r[dn_idx])) if dn_idx is not None and str(r[dn_idx]).strip() else _dn_from_day(r[idx_map.get(day_col)] if day_col in idx_map else "")
                except Exception:
                    dn = 0
                try:
                    sq = int(str(r[idx_map[seq_col]])) if seq_col and str(r[idx_map[seq_col]]).strip() else 0
                except Exception:
                    sq = 0
                data[i][so_idx] = str(int(rn)*10_000_000 + int(dn)*100 + int(sq))
    except Exception:
        pass

    return {"rows": [header] + data}


# ----------------------------
# Normalise rows: fix DayOfWeekNum/DayNumber/Day, CallsPerDay, contiguous Sequence, Depart/Return
# ----------------------------
@app.post("/api/normalize/rows")
async def normalize_rows(request: Request):
    """Clean up a rows table to make schedule fields consistent and non-jumbled.
    Body JSON:
      - rows: 2D array [header, ...rows]
      - groupCol: optional name used for grouping days (defaults to 'Resource' if present else 'Territory').
    Operations per group and day:
      - Derive DayOfWeekNum from DayOfWeek.
      - Recompute DayNumber as consecutive across weeks ordered by (Week, DayOfWeekNum), then Day label 'Day N'.
      - Recompute CallsPerDay.
      - Resequence 1..N within each (group, Day) by StartTime ascending (fallback to existing Sequence then stable order).
      - DepartTime = earliest StartTime in the day; ReturnTime placed on last Sequence row with latest EndTime.
    Returns: { rows }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    rows = body.get("rows")
    if not rows or not isinstance(rows, list) or not rows[0]:
        raise HTTPException(400, "rows required: [header,...]")
    header = [str(h) for h in rows[0]]
    idx = {h: i for i, h in enumerate(header)}

    # Required minimal columns
    grp_col = body.get("groupCol")
    if not grp_col or grp_col not in header:
        grp_col = next((c for c in ("Resource", "Territory", "Group") if c in header), None)
        if not grp_col:
            raise HTTPException(400, "groupCol not provided and no Resource/Territory column found")

    # Ensure canonical columns exist in header; if missing add them and extend rows
    def _ensure_col(name: str):
        if name not in idx:
            header.append(name)
            idx[name] = len(header) - 1
            for r in rows[1:]:
                r.append("")

    for col in ("Week", "DayOfWeek", "DayOfWeekNum", "Day", "DayNumber", "CallsPerDay",
                "Sequence", "StartTime", "EndTime", "DepartTime", "ReturnTime"):
        _ensure_col(col)

    data = rows[1:]

    # 1) DayOfWeekNum from DayOfWeek
    dow_map = {n: i for i, n in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], start=1)}
    for r in data:
        d = str(r[idx["DayOfWeek"]] or "").strip()
        key = d[:3].title()
        r[idx["DayOfWeekNum"]] = dow_map.get(key, r[idx["DayOfWeekNum"]]) or ""

    # 2) Recompute DayNumber across (grp, Week, DayOfWeekNum)
    # Build DataFrame for convenience
    try:
        df = pd.DataFrame(data, columns=header)
    except Exception:
        raise HTTPException(400, "rows must be rectangular and coercible to a table")
    # Normalize types
    def _to_int_series(s):
        try:
            return pd.to_numeric(s, errors="coerce").astype("Int64")
        except Exception:
            return pd.Series([pd.NA]*len(s))
    df["Week"] = _to_int_series(df.get("Week"))
    df["DayOfWeekNum"] = _to_int_series(df.get("DayOfWeekNum"))
    # Unique days sorted by (Week, DayOfWeekNum)
    uniq = (
        df[[grp_col, "Week", "DayOfWeek", "DayOfWeekNum"]]
        .dropna(subset=["DayOfWeek"])  # consider only rows with a weekday
        .drop_duplicates()
        .sort_values([grp_col, "Week", "DayOfWeekNum"], kind="mergesort")
    )
    if len(uniq) > 0:
        uniq["DayNumber"] = uniq.groupby(grp_col).cumcount() + 1
        df = df.merge(uniq[[grp_col, "Week", "DayOfWeek", "DayNumber"]],
                      on=[grp_col, "Week", "DayOfWeek"], how="left", suffixes=("", "_new"))
        # Prefer existing DayNumber if present; only fill gaps from computed values
        if "DayNumber" in df.columns:
            df["DayNumber"] = df["DayNumber"].combine_first(df["DayNumber_new"])
        else:
            df["DayNumber"] = df["DayNumber_new"]
        df = df.drop(columns=[c for c in ("DayNumber_new",) if c in df.columns])
    # Day label
    def _fmt_day(n):
        try:
            n = int(n)
            return f"Day {n}" if n > 0 else ""
        except Exception:
            return ""
    df["Day"] = df["DayNumber"].map(_fmt_day)

    # 3) CallsPerDay by (grp, Day)
    cts = df.groupby([grp_col, "Day"]).size().rename("CallsPerDay").reset_index()
    df = df.drop(columns=["CallsPerDay"], errors="ignore").merge(cts, on=[grp_col, "Day"], how="left")

    # Identify likely aliases for time/sequence columns present in the input
    time_start_candidates = [
        c for c in df.columns if str(c).lower() in ("starttime","start","depart","departure","departtime")
    ]
    time_end_candidates = [
        c for c in df.columns if str(c).lower() in ("endtime","end","finish","return","returntime")
    ]
    seq_candidates = [
        c for c in df.columns if str(c).lower() in ("sequence","seq","order","stoporder","stop")
    ]
    # Choose canonical mapping (fall back to our ensured columns)
    src_start_col = time_start_candidates[0] if time_start_candidates else "StartTime"
    src_end_col   = time_end_candidates[0] if time_end_candidates else "EndTime"
    src_seq_col   = seq_candidates[0] if seq_candidates else "Sequence"

    # 4) Resequence within each (grp, Day) by StartTime then existing Sequence
    # Helper to parse HH:MM, return seconds for sorting
    def _parse_hhmm(v: str) -> int:
        try:
            s = parse_hhmm_to_seconds(str(v))
            return int(s)
        except Exception:
            return 10**9  # push blanks to end
    df["_StartSecTmp"] = df[src_start_col].astype(str).map(_parse_hhmm)
    # stable sort per group/day
    # Prepare an integer sequence column (use existing alias if present)
    if src_seq_col not in df.columns:
        df[src_seq_col] = df.get("Sequence")
    df[src_seq_col] = pd.to_numeric(df.get(src_seq_col), errors="coerce").fillna(0).astype(int)
    def _resequencing(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values(["_StartSecTmp", src_seq_col], kind="mergesort").copy()
        # Re-number sequence 1..N in both the alias column and the canonical "Sequence"
        new_seq = list(range(1, len(sub) + 1))
        sub[src_seq_col] = new_seq
        sub["Sequence"] = new_seq
        # Depart = earliest StartTime; Return placed on row with latest EndTime
        try:
            st_list = sub[src_start_col].astype(str).replace({"nan":""}).tolist()
            st_valid = [s for s in st_list if s]
            depart = min(st_valid) if st_valid else ""
        except Exception:
            depart = ""
        # Determine return time and row index that should carry it (last row with max EndTime)
        ret = ""
        ret_row_label = None
        try:
            et_series = sub[src_end_col].astype(str).replace({"nan":""})
            et_valid = et_series[et_series != ""]
            if len(et_valid) > 0:
                ret = str(et_valid.max())
                mask = (et_series == ret)
                if mask.any():
                    ret_row_label = mask[mask].index[-1]
        except Exception:
            ret = ""
            ret_row_label = None
        sub["DepartTime"] = depart
        sub["ReturnTime"] = ""
        if ret_row_label is not None:
            sub.loc[ret_row_label, "ReturnTime"] = ret
        return sub
    df = df.groupby([grp_col, "Day"], dropna=False, group_keys=False).apply(_resequencing)

    # Cleanup temp col
    df = df.drop(columns=[c for c in ("_StartSecTmp",) if c in df.columns])

    # If legacy time columns exist, mirror Depart/Return into them for user visibility
    legacy_depart_cols = [c for c in df.columns if str(c).lower() in ("depart","departure","departtime")]
    legacy_return_cols = [c for c in df.columns if str(c).lower() in ("return","returntime","finish","end")]
    if legacy_depart_cols:
        for c in legacy_depart_cols:
            df[c] = df["DepartTime"]
    if legacy_return_cols:
        # Keep only the last row of each (grp, Day) with the return value; others blank
        def _keep_last_return(sub: pd.DataFrame) -> pd.DataFrame:
            vals = sub["ReturnTime"].astype(str)
            sub[c] = ""
            if (vals != "").any():
                last_idx = vals[vals != ""].index[-1]
                sub.loc[last_idx, c] = vals[last_idx]
            return sub
        for c in legacy_return_cols:
            df = df.groupby([grp_col, "Day"], dropna=False, group_keys=False).apply(_keep_last_return)

    # Rebuild rows keeping original columns order but ensuring required fields are present
    preferred = [
        "Resource","Territory","Week","DayOfWeek","DayOfWeekNum","Day","DayNumber","CallsPerDay",
        "Sequence","StartTime","EndTime","DepartTime","ReturnTime","Latitude","Longitude"
    ]
    base_cols = [c for c in preferred if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols]
    out_cols = base_cols + other_cols
    df = df[out_cols]
    df = df.astype("object").where(pd.notna(df), "").astype(str)
    new_rows = [out_cols] + df.values.tolist()
    return JSONResponse({"rows": new_rows})


# ----------------------------
# Apply a day change: re-optimise the old and new day, then tidy counts
# ----------------------------
@app.post("/api/reopt/apply-day-change")
async def apply_day_change(request: Request):
    """Orchestrate a pin's day change in one call:
    - Re-optimise the route for the previous DayNumber (if any)
    - Re-optimise the route for the new DayNumber (if any)
    - Recompute CallsPerDay per (groupCol, Day)

    Body JSON:
      - rows: 2D array [header, ...rows]
      - beforeDay: optional integer (previous DayNumber)
      - afterDay: optional integer (new DayNumber)
      - groupCol: optional grouping column (defaults Resource > Territory > Group)
      - depotStrategy: optional ('medoid'|'centroid'), default 'medoid'
      - startTime: optional HH:MM default '08:00'
      - endTime: optional HH:MM default start+9h
    Returns: { rows }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    rows = body.get("rows")
    if not rows or not isinstance(rows, list) or not rows[0]:
        raise HTTPException(400, "rows required: [header,...]")

    header = [str(h) for h in rows[0]]
    data = rows[1:]

    import re as _re
    # Helper to find columns by pattern
    def _find_col(pats: list[str]) -> str | None:
        for p in pats:
            rx = _re.compile(p, _re.I)
            for h in header:
                if rx.search(h):
                    return h
        return None

    # Identify required columns
    lat_col = _find_col([r"^lat(itude)?$"])
    lon_col = _find_col([r"^(lon|lng|long|longitude)$"])
    if not lat_col or not lon_col:
        raise HTTPException(400, "Latitude/Longitude columns required in rows")

    group_col = body.get("groupCol")
    if not group_col or group_col not in header:
        group_col = next((c for c in ("Resource", "Territory", "Group") if c in header), None)
        if not group_col:
            raise HTTPException(400, "groupCol not provided and no Resource/Territory column found")

    # Canonical columns (optional)
    daynum_col = next((c for c in header if _re.search(r"^day\s*number$", c, _re.I)), None)
    day_col = next((c for c in header if _re.fullmatch(r"day", c, _re.I)), None)
    seq_col = "Sequence" if "Sequence" in header else None
    st_col = "StartTime" if "StartTime" in header else None
    et_col = "EndTime" if "EndTime" in header else None
    dp_col = "DepartTime" if "DepartTime" in header else None
    rt_col = "ReturnTime" if "ReturnTime" in header else None
    dur_col = next((c for c in header if _re.search(r"duration|mins|minutes|service\s*time", c, _re.I)), None)

    idx_map = {h: i for i, h in enumerate(header)}

    # Parse hints
    depot_strategy = str(body.get("depotStrategy") or "medoid").strip().lower()
    start_time_hint = str(body.get("startTime") or "08:00").strip()
    end_time_hint = str(body.get("endTime") or "").strip()
    try:
        default_start = parse_hhmm_to_seconds(start_time_hint)
    except Exception:
        default_start = 8*3600
    if end_time_hint:
        try:
            default_end = parse_hhmm_to_seconds(end_time_hint)
            if default_end <= default_start:
                default_end += 24*3600
        except Exception:
            default_end = default_start + 9*3600
    else:
        default_end = default_start + 9*3600

    # Utility
    def _fmt(sec: int) -> str:
        h = int(sec // 3600); m = int((sec % 3600)//60)
        return f"{h:02d}:{m:02d}"

    def _row_day_number(r: list) -> int | None:
        if daynum_col and daynum_col in idx_map:
            try:
                v = r[idx_map[daynum_col]]
                return int(str(v))
            except Exception:
                pass
        if day_col and day_col in idx_map:
            m = _re.search(r"(\d+)", str(r[idx_map[day_col]]))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    # Core: re-optimise a specific day in-place
    def _reopt_for_day(day_num: int):
        if not isinstance(day_num, int) or day_num <= 0:
            return
        from collections import defaultdict
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(data):
            dn = _row_day_number(r)
            if dn != day_num:
                continue
            gval = str(r[idx_map[group_col]] or "").strip() or "(Unassigned)"
            groups[gval].append(i)
        if not groups:
            return
        for gval, idxs in groups.items():
            # Clear stale ReturnTime on all stops in this (group, day)
            if rt_col:
                for di in idxs:
                    try:
                        data[di][idx_map[rt_col]] = ""
                    except Exception:
                        pass
            coords = []
            service = []
            for di in idxs:
                row = data[di]
                try:
                    lat = float(row[idx_map[lat_col]]); lng = float(row[idx_map[lon_col]])
                except Exception:
                    continue
                coords.append([lat, lng])
                if dur_col and dur_col in idx_map:
                    try:
                        service.append(int(parse_duration_to_seconds(row[idx_map[dur_col]])))
                    except Exception:
                        service.append(0)
                else:
                    service.append(0)
            if len(coords) <= 1:
                if seq_col:
                    for j, di in enumerate(idxs, start=1):
                        data[di][idx_map[seq_col]] = str(j)
                if dp_col:
                    for di in idxs:
                        data[di][idx_map[dp_col]] = _fmt(default_start)
                continue
            sub = np.array(coords, dtype=float)
            try:
                depot_lat, depot_lng = _compute_depot_for_territory(sub, strategy=depot_strategy)
            except Exception:
                depot_lat = float(np.mean(sub[:,0])); depot_lng = float(np.mean(sub[:,1]))
            start_sec = default_start
            if dp_col and dp_col in idx_map:
                try:
                    from collections import Counter
                    vals = [str(data[di][idx_map[dp_col]] or "").strip() for di in idxs]
                    vals = [v for v in vals if v]
                    if vals:
                        com = Counter(vals).most_common(1)[0][0]
                        ssec = parse_hhmm_to_seconds(com)
                        if isinstance(ssec, int) and ssec >= 0:
                            start_sec = ssec
                except Exception:
                    pass
            end_sec = default_end if default_end > start_sec else start_sec + 9*3600

            visited_rel, _dropped = vrp_single_route_with_time(
                sub, (depot_lat, depot_lng), service, int(start_sec), int(end_sec), time_limit_sec=10, max_stops=None, seed=42
            )
            order = list(visited_rel) if visited_rel else list(range(len(idxs)))
            all_pts = np.vstack([sub, np.array([[depot_lat, depot_lng]])])
            try:
                dm_route = osrm_table_batch_duration(all_pts)
            except Exception:
                dm_route = _haversine_matrix_full(all_pts) / 1000.0 / 40.0 * 3600.0
            prev = len(sub)
            t = int(start_sec)
            if dp_col:
                for di in idxs:
                    data[di][idx_map[dp_col]] = _fmt(start_sec)
            last_abs_idx = None
            for seq, rel in enumerate(order, start=1):
                abs_data_idx = idxs[int(rel)]
                travel = int(dm_route[prev, rel])
                svc = int(service[int(rel)]) if int(rel) < len(service) else 0
                s = t + travel; e = s + svc
                if seq_col:
                    data[abs_data_idx][idx_map[seq_col]] = str(seq)
                if st_col:
                    data[abs_data_idx][idx_map[st_col]] = _fmt(s)
                if et_col:
                    data[abs_data_idx][idx_map[et_col]] = _fmt(e)
                t = e; prev = rel; last_abs_idx = abs_data_idx
            if last_abs_idx is not None and rt_col:
                try:
                    ret_s = int(t) + int(dm_route[prev, len(sub)])
                except Exception:
                    ret_s = int(t)
                data[last_abs_idx][idx_map[rt_col]] = _fmt(ret_s)

            # Final safety: monotonic Sequence 1..N by StartTime
            if seq_col and st_col and st_col in idx_map:
                def _sec(v: str) -> int:
                    try:
                        return int(parse_hhmm_to_seconds(str(v)))
                    except Exception:
                        return 10**9
                pairs = [(_sec(data[di][idx_map[st_col]]), di) for di in idxs]
                pairs.sort(key=lambda x: x[0])
                for new_seq, (_, di) in enumerate(pairs, start=1):
                    data[di][idx_map[seq_col]] = str(new_seq)

    # Execute for both days
    before_day = body.get("beforeDay")
    after_day = body.get("afterDay")
    try:
        before_day = int(before_day) if before_day is not None else None
    except Exception:
        before_day = None
    try:
        after_day = int(after_day) if after_day is not None else None
    except Exception:
        after_day = None
    if before_day and (not after_day or before_day != after_day):
        _reopt_for_day(before_day)
    if after_day:
        _reopt_for_day(after_day)

    # Recompute CallsPerDay per (groupCol, Day)
    if "CallsPerDay" in header:
        grp_idx = idx_map.get(group_col)
        day_idx = idx_map.get(day_col) if day_col else None
        if grp_idx is not None and day_idx is not None:
            from collections import defaultdict
            counts: dict[tuple, int] = defaultdict(int)
            keys: list[tuple] = []
            for r in data:
                g = str(r[grp_idx] or "").strip(); d = str(r[day_idx] or "").strip()
                k = (g, d)
                keys.append(k)
                if g and d:
                    counts[k] += 1
            cpd_idx = idx_map["CallsPerDay"]
            for r, k in zip(data, keys):
                g, d = k
                r[cpd_idx] = str(counts[k]) if (g and d) else ""

    return {"rows": [header] + data}


# ----------------------------
# AI Q&A over result rows (optional OpenAI/Azure)
# ----------------------------
def _format_rows_for_prompt(rows: list[list], max_rows: int = 200) -> str:
    try:
        hdr = [str(h) for h in rows[0]] if rows else []
        body = rows[1:1+max_rows] if len(rows) > 1 else []
        # Compact JSONL
        out_lines = []
        for r in body:
            rec = {hdr[i]: r[i] if i < len(r) else "" for i in range(len(hdr))}
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        return "\n".join(out_lines)
    except Exception:
        return ""

def _parse_distance_query(q: str) -> Optional[Tuple[str, str]]:
    ql = (q or '').lower().strip()
    import re as _re
    # common phrasings
    pats = [
        r"(?:distance|time)\s+(?:from)\s+(.+?)\s+(?:to)\s+(.+)$",
        r"(?:from)\s+(.+?)\s+(?:to)\s+(.+?)\s+(?:distance|time)$",
        r"fastest\s+(?:route|time|distance)\s+from\s+(.+?)\s+to\s+(.+)$",
    ]
    for p in pats:
        m = _re.search(p, ql)
        if m:
            a = (m.group(1) or '').strip().strip('?').strip()
            b = (m.group(2) or '').strip().strip('?').strip()
            if a and b:
                return (a, b)
    return None

def _geocode_place(name: str) -> Optional[Tuple[float, float]]:
    if not name:
        return None
    if gmaps is None:
        raise RuntimeError("Google Maps client not initialised")
    try:
        res = gmaps.geocode(name)
        if res and len(res) > 0:
            loc = res[0]["geometry"]["location"]
            return (float(loc["lat"]), float(loc["lng"]))
    except Exception:
        return None
    return None

def _osrm_route_time_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> Optional[Tuple[float, float]]:
    try:
        url = f"{OSRM_BASE}/route/v1/driving/{lng1},{lat1};{lng2},{lat2}?overview=false&alternatives=false&annotations=duration,distance"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        routes = data.get("routes") or []
        if not routes:
            return None
        route = routes[0]
        duration = float(route.get("duration", 0.0))  # seconds
        distance = float(route.get("distance", 0.0))  # meters
        return (duration, distance)
    except Exception:
        return None

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        from math import radians, sin, cos, asin, sqrt
        R = 6371000.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c
    except Exception:
        return float('nan')

def _places_nearby(lat: float, lng: float, keyword: Optional[str], place_type: Optional[str], radius_m: int, max_results: int = 5) -> list[dict]:
    try:
        if not GOOGLE_API_KEY or gmaps is None:
            return []
        kwargs = {"location": (float(lat), float(lng)), "radius": int(radius_m)}
        if keyword:
            kwargs["keyword"] = keyword
        if place_type:
            kwargs["type"] = place_type
        res = gmaps.places_nearby(**kwargs)
        results = (res or {}).get("results", [])
        out = []
        for r in results[:max_results]:
            try:
                loc = r.get("geometry", {}).get("location", {})
                plat = float(loc.get("lat"))
                plng = float(loc.get("lng"))
                out.append({
                    "name": r.get("name"),
                    "lat": plat,
                    "lng": plng,
                    "place_id": r.get("place_id"),
                    "types": r.get("types", []),
                    "vicinity": r.get("vicinity") or r.get("formatted_address"),
                    "rating": r.get("rating"),
                })
            except Exception:
                continue
        return out
    except Exception:
        return []

def _smart_groupby_answer(question: str, rows: list[list]) -> Optional[str]:
    """Heuristic: group and aggregate by a dimension mentioned in the question."""
    try:
        if not rows or len(rows) < 2:
            return None
        import re as _re
        ql = (question or '').lower()
        if not _re.search(r"compare| by | vs |versus|top|highest|lowest|average|avg|sum|total|mean|median", ql):
            return None
        hdr = [str(h) for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=hdr)
        # choose dimension
        dim_candidates = [c for c in hdr if _re.search(r"territory|region|resource|rep|agent|owner|dayofweek|week|postcode|city|area|group", c, _re.I)]
        dim = None
        for c in dim_candidates:
            if c.lower() in ql:
                dim = c
                break
        if dim is None:
            dim = dim_candidates[0] if dim_candidates else None
        # choose metric
        metric_map = [
            (r"sales|revenue|turnover", "sum"),
            (r"margin|profit", "sum"),
            (r"duration|service\s*minutes|time|minutes", "sum"),
            (r"calls|visits|stops|orders", "count"),
        ]
        agg = None
        metric = None
        for pat, default_agg in metric_map:
            if _re.search(pat, ql):
                metric = next((c for c in hdr if _re.search(pat, c, _re.I)), None)
                agg = default_agg
                break
        if metric is None:
            # fallback: pick first numeric column
            for c in hdr:
                s = pd.to_numeric(df[c], errors='coerce')
                if s.notna().sum() >= max(3, int(0.2*len(s))):
                    metric = c
                    agg = "sum"
                    break
        if dim is None:
            return None
        # aggregation mode hint
        if _re.search(r"average|avg|mean", ql):
            agg = "mean" if metric and agg != "count" else agg
        if _re.search(r"median", ql):
            agg = "median" if metric and agg != "count" else agg
        if _re.search(r"sum|total", ql):
            agg = "sum" if metric and agg != "count" else agg
        # do groupby
        if agg == "count" or metric is None:
            g = df.groupby(dim).size().sort_values(ascending=False)
            items = g.head(10).items()
            parts = [f"{k}: {int(v)}" for k, v in items]
            return f"Calls by {dim} (top 10): " + "; ".join(parts)
        s = pd.to_numeric(df[metric], errors='coerce')
        df2 = df.assign(_metric=s)
        if agg == "mean":
            g = df2.groupby(dim)["_metric"].mean().sort_values(ascending=False)
            label = f"Average {metric} by {dim} (top 10)"
        elif agg == "median":
            g = df2.groupby(dim)["_metric"].median().sort_values(ascending=False)
            label = f"Median {metric} by {dim} (top 10)"
        else:
            g = df2.groupby(dim)["_metric"].sum().sort_values(ascending=False)
            label = f"Total {metric} by {dim} (top 10)"
        items = g.head(10).items()
        parts = [f"{k}: {round(float(v), 2)}" for k, v in items]
        return f"{label}: " + "; ".join(parts)
    except Exception:
        return None

def _infer_schema(rows: list[list], max_unique: int = 20) -> dict:
    """Infer simple schema and stats: column names, guessed type, small samples/uniques, numeric stats."""
    try:
        if not rows or len(rows) < 2:
            return {"columns": []}
        hdr = [str(h) for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=hdr)
        cols = []
        for c in hdr:
            series = df[c]
            info = {"name": c}
            # Try numeric first
            s_num = pd.to_numeric(series, errors='coerce')
            num_valid = int(s_num.notna().sum())
            if num_valid >= max(3, int(0.2 * len(series))):
                info["type"] = "number"
                info["count"] = int(len(series))
                info["valid"] = num_valid
                info["min"] = float(s_num.min()) if num_valid else None
                info["max"] = float(s_num.max()) if num_valid else None
                info["mean"] = float(s_num.mean()) if num_valid else None
                info["median"] = float(s_num.median()) if num_valid else None
            else:
                info["type"] = "text"
                vc = series.astype(str).fillna("").value_counts().head(max_unique)
                info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
            cols.append(info)
        return {"columns": cols}
    except Exception:
        return {"columns": []}

def _basic_stats_answer(question: str, rows: list[list]) -> tuple[str, list]:
    # Simple heuristic answer using pandas if available; also try to answer basic counts
    try:
        if not rows or len(rows) < 2:
            return ("No data to analyze.", [])
        hdr = [str(h) for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=hdr)
        # Try to answer "how many calls does X have" from Resource/Territory-like columns
        ql = (question or '').strip().lower()
        who = None
        import re as _re
        m = _re.search(r"how\s+many\s+calls\s+(does|for)?\s*([a-z0-9 .\-']+)", ql)
        if m:
            who = (m.group(2) or '').strip()
        # Candidate owner columns
        cand_cols = [c for c in df.columns if _re.search(r"resource|rep|agent|owner|name|territory|group", c, _re.I)]
        if who and cand_cols:
            who_l = who.lower()
            total = 0
            by_col = {}
            for c in cand_cols:
                colvals = df[c].astype(str).fillna("")
                cnt = int((colvals.str.lower() == who_l).sum())
                if cnt:
                    by_col[c] = cnt
                    total += cnt
            if total:
                detail = ", ".join([f"{c}: {v}" for c, v in by_col.items()])
                return (f"{who} has {total} calls ({detail}).", [])

        # Try to answer postcode questions like "how many calls in LS postcode" (with flexible phrasing)
        pc_term = r"(?:post\s*code|postal\s*code|zipcode|zip|eircode)"
        patterns = [
            r"how\s+many(?:\s+calls)?\s+(?:are\s+)?in\s+(?:the\s+)?([a-z0-9 ]+)\s*" + pc_term,
            r"(?:count|how\s+many(?:\s+calls)?)\s+(?:in\s+)?([a-z0-9 ]+)\s*" + pc_term,
        ]
        token = None
        for pat in patterns:
            m2 = _re.search(pat, ql, _re.I)
            if m2:
                token = (m2.group(1) or '').strip()
                break
        # If still no token, capture the word immediately before postcode term
        if not token:
            m3 = _re.search(r"\b([a-z0-9]{1,6})\b\s*" + pc_term, ql, _re.I)
            if m3:
                token = (m3.group(1) or '').strip()
        if token:
            # Find postcode-like columns
            pc_cols = [c for c in df.columns if _re.search(pc_term, c, _re.I)]
            if pc_cols:
                tnorm = token.replace(" ", "").upper()
                total = 0
                by_col = {}
                for c in pc_cols:
                    colvals = df[c].astype(str).fillna("")
                    series_norm = colvals.str.replace(r"\s+", "", regex=True).str.upper()
                    # If token looks like an area prefix (1-3 letters + optional digit/letter), prefix match; else exact
                    if _re.fullmatch(r"[A-Z]{1,3}[0-9A-Z]?", tnorm):
                        mask = series_norm.str.startswith(tnorm)
                    else:
                        mask = series_norm == tnorm
                    cnt = int(mask.sum())
                    if cnt:
                        by_col[c] = cnt
                        total += cnt
                if total:
                    detail = ", ".join([f"{c}: {v}" for c, v in by_col.items()])
                    return (f"Calls in postcode '{token}': {total} ({detail}).", [])
        # Generic summary fallback
        n = len(df)
        cols = ", ".join(hdr)
        info = [f"Rows: {n}", f"Columns: {len(hdr)} ({cols})"]
        for c in ("Territory","Day","DayOfWeek","Resource"):
            if c in df.columns:
                info.append(f"Unique {c}: {df[c].nunique()}")
        return ("\n".join(info), [])
    except Exception:
        return ("Unable to compute summary.", [])

@app.post("/api/ask")
async def ask_ai(request: Request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    rows = body.get("rows") or []
    max_rows = int(body.get("maxRows", 200))
    force_llm = bool(body.get("force", False))
    include_schema = bool(body.get("includeSchema", True))
    if not question:
        raise HTTPException(400, "question required")
    # Try OpenAI/Azure if configured; else fallback (unless REQUIRE_OPENAI)
    use_openai = bool(os.getenv("OPENAI_API_KEY")) or (bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT")))
    require_openai = bool(int(os.getenv("REQUIRE_OPENAI", "0")))
    use_openai = use_openai or force_llm or require_openai
    # Tool: distance/time between two places
    dt = _parse_distance_query(question)
    if dt:
        a_name, b_name = dt
        a = _geocode_place(a_name)
        b = _geocode_place(b_name)
        if a and b:
            dur_dist = _osrm_route_time_distance(a[0], a[1], b[0], b[1])
            if dur_dist:
                dur_s, dist_m = dur_dist
                minutes = int(round(dur_s / 60))
                km = round(dist_m / 1000.0, 1)
                miles = round(km * 0.621371, 1)
                return {"answer": f"Fastest drive: ~{minutes} minutes (~{km} km / {miles} mi)", "citations": [], "provider": "tool:osrm", "used_rows": 0}
    # Tool: deterministic postcode counts (e.g., "how many calls in LS postcode")
    try:
        ans_pc, cites_pc = _postcode_count_answer(question, rows)
        if ans_pc is not None:
            return {"answer": ans_pc, "citations": cites_pc or [], "provider": "tool:stats", "used_rows": max(0, len(rows)-1)}
    except Exception:
        pass
    # Tool: places nearby (vendors/competitors) for provided rows
    try:
        import re as _re
        ql = (question or '').lower()
        wants_nearby = _re.search(r"\bnear|nearby|around|within\b", ql) and _re.search(r"vendor|supplier|competitor|store|shop|distributor", ql)
        if wants_nearby and rows and len(rows) > 1:
            hdr = [str(h) for h in rows[0]]
            data = rows[1:]
            lat_cols = [c for c in hdr if _re.search(r"^lat(itude)?$", c, _re.I)]
            lon_cols = [c for c in hdr if _re.search(r"^(lon|lng|long|longitude)$", c, _re.I)]
            # Fallback from postcode via geocoding if no lat/lng present
            pc_cols = [c for c in hdr if _re.search(r"post\s*code|postal\s*code|zipcode|zip|eircode", c, _re.I)]
            if lat_cols and lon_cols:
                lat_c = lat_cols[0]; lon_c = lon_cols[0]
                coords = []
                for r in data:
                    try:
                        lat = float(r[hdr.index(lat_c)]); lng = float(r[hdr.index(lon_c)])
                        coords.append((lat, lng))
                    except Exception:
                        continue
            elif pc_cols:
                coords = []
                pc_c = pc_cols[0]
                seen_pc = set()
                for r in data[:50]:  # cap geocoding
                    pc = str(r[hdr.index(pc_c)] or '').strip()
                    if not pc or pc in seen_pc:
                        continue
                    seen_pc.add(pc)
                    loc = _geocode_place(pc)
                    if loc:
                        coords.append(loc)
            else:
                coords = []
            # Parse radius
            radius_m = 2000
            m = _re.search(r"within\s+(\d+)\s*(m|meters|metres|km|kilometers|kilometres|mile|miles)?", ql)
            if m:
                n = int(m.group(1) or 0)
                unit = (m.group(2) or '').lower()
                if unit in ("km","kilometers","kilometres"):
                    radius_m = int(n * 1000)
                elif unit in ("mile","miles"):
                    radius_m = int(n * 1609)
                else:
                    radius_m = int(n)
            # Keyword/type
            keyword = None; ptype = None
            if _re.search(r"competitor", ql):
                keyword = "competitor"
            if _re.search(r"vendor|supplier|distributor", ql):
                keyword = "supplier" if keyword is None else keyword
            if _re.search(r"pharmacy|hardware|plumber|electrical|grocer|builder|auto|garage|restaurant|cafe|shop|store", ql):
                # let keyword be the detected sector
                kw = _re.search(r"pharmacy|hardware|plumber|electrical|grocer|builder|auto|garage|restaurant|cafe|shop|store", ql)
                if kw: keyword = kw.group(0)
            if not keyword:
                keyword = "store"
            if not coords:
                # fallback answer
                return {"answer": "I need coordinates or postcodes in the table to search nearby places.", "citations": [], "provider": "tool:places", "used_rows": max(0, len(rows)-1)}
            # Query places; aggregate unique vendors by name and count how many calls they are near
            agg: dict[str, dict] = {}
            limit_points = min(len(coords), 30)
            for (lat, lng) in coords[:limit_points]:
                places = _places_nearby(lat, lng, keyword=keyword, place_type=ptype, radius_m=radius_m, max_results=5)
                for p in places:
                    name = str(p.get("name") or "").strip()
                    if not name:
                        continue
                    key = name.lower()
                    entry = agg.get(key)
                    if not entry:
                        entry = {"name": name, "count": 0, "sample": p}
                        agg[key] = entry
                    entry["count"] += 1
            if not agg:
                return {"answer": f"No nearby places found within ~{radius_m}m for the selected rows.", "citations": [], "provider": "tool:places", "used_rows": max(0, len(rows)-1)}
            # Top 15 by frequency
            top = sorted(agg.values(), key=lambda x: (-int(x["count"]), x["name"]))[:15]
            lines = [f"Top nearby places (radius ~{radius_m}m, keyword '{keyword}'):"]
            for e in top:
                s = e["sample"]
                vicinity = s.get("vicinity") or ""
                lines.append(f"- {e['name']} — near {e['count']} call(s){' • ' + vicinity if vicinity else ''}")
            return {"answer": "\n".join(lines), "citations": [], "provider": "tool:places", "used_rows": max(0, len(rows)-1)}
    except Exception:
        pass
    if not use_openai:
        if require_openai:
            raise HTTPException(503, "OpenAI is required but not configured.")
        ans, cites = _basic_stats_answer(question, rows)
        # Try groupby analytics if generic summary returned
        if ans.startswith("Rows:"):
            gb = _smart_groupby_answer(question, rows)
            if gb:
                ans = gb
        return {"answer": ans, "citations": cites, "provider": "fallback", "used_rows": max(0, len(rows)-1)}
    prompt_rows = _format_rows_for_prompt(rows, max_rows=max_rows)
    schema = _infer_schema(rows) if include_schema else {"columns": []}
    system = (
        "You are an assistant answering questions about a table.\n"
        "Only answer using the provided rows; if insufficient, say you don't know.\n"
        "Cite up to 5 matching rows with a short key if helpful.\n"
        "Return clear, concise answers. If the user asks for a count, return a single number with a short label."
    )
    user_parts = [
        f"Question: {question}",
    ]
    if include_schema:
        user_parts.append(f"Schema (inferred):\n{json.dumps(schema, ensure_ascii=False)}")
    user_parts.append(f"Rows (JSONL, up to {max_rows}):\n{prompt_rows}")
    user = "\n\n".join(user_parts)
    answer_text = None
    provider_used = "openai"
    try:
        # Lazy import via importlib to keep dependency optional
        import os as _os
        import importlib as _importlib
        _openai_mod = _importlib.import_module('openai')
        OpenAIClass = getattr(_openai_mod, 'OpenAI')
        provider = _os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "azure":
            client = OpenAIClass(
                api_key=_os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=f"{_os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/deployments/{_os.getenv('AZURE_OPENAI_DEPLOYMENT')}",
            )
            model = _os.getenv("AZURE_OPENAI_DEPLOYMENT")
            provider_used = "azure-openai"
        else:
            client = OpenAIClass(api_key=_os.getenv("OPENAI_API_KEY"))
            model = _os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        answer_text = resp.choices[0].message.content if resp and resp.choices else None
    except Exception as e:
        if require_openai:
            raise HTTPException(502, f"OpenAI call failed: {str(e)}")
        answer_text = None
        provider_used = f"fallback:error:{type(e).__name__}"
    if not answer_text:
        if require_openai:
            raise HTTPException(502, "OpenAI call returned no content.")
        ans, cites = _basic_stats_answer(question, rows)
        return {"answer": ans, "citations": cites, "provider": provider_used, "used_rows": max(0, len(rows)-1)}
    return {"answer": answer_text, "citations": [], "provider": provider_used, "used_rows": max(0, len(rows)-1)}

@app.get("/api/ask/status")
def ask_status():
    ok_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    ok_azure = bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT")) and bool(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    try:
        import importlib as _importlib
        _importlib.import_module('openai')
        has_pkg = True
    except Exception:
        has_pkg = False
    return {
        "openai_key": ok_openai_key,
        "azure_openai": ok_azure,
        "openai_package": has_pkg,
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "provider_env": os.getenv("LLM_PROVIDER", "openai"),
    "require_openai": bool(int(os.getenv("REQUIRE_OPENAI", "0")))
    }

def _detect_columns(header: list[str]) -> dict:
    import re as _re
    colmap = {}
    for c in header:
        if _re.search(r"post\s*code|postal\s*code|zipcode|zip|eircode", c, _re.I):
            colmap.setdefault('postcode', c)
        if _re.search(r"resource|rep|agent|owner|name|assignee|assigned\s*to", c, _re.I):
            colmap.setdefault('resource', c)
        if _re.search(r"territory|group|region|area", c, _re.I):
            colmap.setdefault('territory', c)
    return colmap

# Deterministic postcode counting tool used before LLM
def _postcode_count_answer(question: str, rows: list[list]) -> tuple[Optional[str], list]:
    import re as _re
    if not rows or not isinstance(rows, list) or not rows[0]:
        return (None, [])
    hdr = rows[0]
    data = rows[1:]
    try:
        df = pd.DataFrame(data, columns=hdr)
    except Exception:
        return (None, [])
    ql = (question or "").strip()
    pc_term = r"(?:post\s*code|postal\s*code|zipcode|zip|eircode)"
    patterns = [
        r"how\s+many(?:\s+calls)?\s+(?:are\s+)?in\s+(?:the\s+)?([a-z0-9 ]+)\s*" + pc_term,
        r"(?:count|how\s+many(?:\s+calls)?)\s+(?:in\s+)?([a-z0-9 ]+)\s*" + pc_term,
    ]
    token = None
    for pat in patterns:
        m2 = _re.search(pat, ql, _re.I)
        if m2:
            token = (m2.group(1) or '').strip()
            break
    if not token:
        m3 = _re.search(r"\b([a-z0-9]{1,6})\b\s*" + pc_term, ql, _re.I)
        if m3:
            token = (m3.group(1) or '').strip()
    # Also allow queries like "how many calls in LS" (without the word postcode)
    if not token:
        m4 = _re.search(r"how\s+many(?:\s+calls)?\s+(?:are\s+)?in\s+([a-z0-9]{1,6})\b", ql, _re.I)
        if m4:
            token = (m4.group(1) or '').strip()
    if not token:
        return (None, [])
    pc_cols = [c for c in df.columns if _re.search(pc_term, c, _re.I)]
    if not pc_cols:
        return (None, [])
    tnorm = token.replace(" ", "").upper()
    total = 0
    by_col: dict[str,int] = {}
    for c in pc_cols:
        colvals = df[c].astype(str).fillna("")
        series_norm = colvals.str.replace(r"\s+", "", regex=True).str.upper()
        if _re.fullmatch(r"[A-Z]{1,3}[0-9A-Z]?", tnorm):
            mask = series_norm.str.startswith(tnorm)
        else:
            mask = series_norm == tnorm
        cnt = int(mask.sum())
        if cnt:
            by_col[c] = cnt
            total += cnt
    if total:
        detail = ", ".join([f"{c}: {v}" for c, v in by_col.items()])
        return (f"Calls in postcode '{token}': {total} ({detail}).", [])
    return (None, [])

def _duckdb_query_rows(rows: list[list], sql: str, limit: int = 200) -> dict:
    """Execute a read-only SQL over an in-memory DuckDB table built from rows. Returns dict with columns and rows.
    Enforces LIMIT to prevent large outputs. Raises on non-select or unsafe statements.
    """
    if not _DUCKDB_AVAILABLE:
        raise RuntimeError("duckdb not available")
    if not rows or len(rows) < 2:
        return {"columns": [], "rows": []}
    hdr = [str(h) for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=hdr)
    con = _duckdb.connect(database=':memory:')
    try:
        con.register('t', df)
        q = (sql or '').strip().rstrip(';')
        # Guardrails: only allow SELECT queries
        low = q.lower()
        if not low.startswith('select'):
            raise ValueError("Only SELECT queries are allowed")
        # Inject LIMIT if missing
        if ' limit ' not in low:
            q = f"{q} LIMIT {int(limit)}"
        res = con.execute(q).fetchdf()
        cols = list(res.columns)
        rows_out = res.values.tolist()
        return {"columns": cols, "rows": rows_out}
    finally:
        con.close()

@app.post("/api/ask/analyze")
async def ask_analyze(request: Request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    rows = body.get("rows") or []
    max_rows = int(body.get("maxRows", 200))
    if not question:
        raise HTTPException(400, "question required")
    # First: deterministic postcode counter
    try:
        ans_pc, _ = _postcode_count_answer(question, rows)
        if ans_pc:
            return {"answer": ans_pc, "provider": "tool:stats", "table": None, "sql": None}
    except Exception:
        pass
    # Otherwise: attempt Text-to-SQL via LLM + DuckDB
    if not _DUCKDB_AVAILABLE:
        raise HTTPException(503, "DuckDB not available for analysis")
    # Build schema for prompt
    hdr = [str(h) for h in (rows[0] if rows else [])]
    # LLM required?
    require_openai = bool(int(os.getenv("REQUIRE_OPENAI", "0")))
    try:
        import importlib as _importlib
        _openai_mod = _importlib.import_module('openai')
        OpenAIClass = getattr(_openai_mod, 'OpenAI')
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "azure":
            client = OpenAIClass(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}",
            )
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        else:
            client = OpenAIClass(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    except Exception as e:
        if require_openai:
            raise HTTPException(503, f"OpenAI is required but not available: {e}")
        raise HTTPException(503, "LLM unavailable for analysis")
    system = (
        "You translate a natural-language analytics question into a single safe SQL SELECT over table t. "
        "Table name: t. Columns are provided. Use WHERE/GROUP BY/ORDER BY/LIMIT as needed. Do not modify data. "
        "Return ONLY the SQL, nothing else. Prefer COUNT, SUM, AVG as requested."
    )
    user = (
        f"Question: {question}\n"
        f"Columns: {json.dumps(hdr, ensure_ascii=False)}\n"
        f"Guidelines: The table is small (<=2000 rows). If counting, include a COUNT(*) with a clear alias."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.1,
        )
        sql_text = resp.choices[0].message.content.strip() if resp and resp.choices else ""
        # Extract code block if present
        import re as _re
        m = _re.search(r"```(?:sql)?\n([\s\S]*?)\n```", sql_text, _re.I)
        if m:
            sql_text = m.group(1).strip()
        if not sql_text.lower().startswith("select"):
            # Try to find a SELECT line
            m2 = _re.search(r"(select[\s\S]+)$", sql_text, _re.I)
            if m2:
                sql_text = m2.group(1).strip()
        table = _duckdb_query_rows(rows, sql_text, limit=200)
        return {"answer": "analysis", "provider": "openai+duckdb", "table": table, "sql": sql_text}
    except HTTPException:
        raise
    except Exception as e:
        if require_openai:
            raise HTTPException(502, f"Analysis failed: {e}")
        raise HTTPException(502, "Analysis failed")

def _best_column_match(header: list[str], name: str) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower().replace('_',' ').replace('-',' ')
    # exact (case-insensitive)
    for h in header:
        if h.strip().lower() == n:
            return h
    # contains
    for h in header:
        if n in h.strip().lower():
            return h
    return None

# ----------------------------
# Re-optimise territories from an existing table (keep day numbers intact)
# ----------------------------
@app.post("/api/reopt/territories")
async def reopt_territories(request: Request):
    """Recompute Territory assignment for the provided rows using current tuning settings.
    Body JSON:
      - rows: 2D array [header, ...rows]
      - numTerritories: optional int; if missing, inferred from existing Territory labels
      - minCalls, maxCalls: optional per-territory min/max (defaults to even split bounds)
      - keepNames: optional bool, default true. If Territory column exists, try to map new clusters to existing names to minimise renames.
    Returns: updated rows array with Territory (and TerritoryRank/SortOrder if present) refreshed.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    rows = body.get("rows")
    if not rows or not isinstance(rows, list) or not rows[0]:
        raise HTTPException(400, "rows required: [header,...]")
    header = [str(h) for h in rows[0]]
    data = rows[1:]

    # Find coordinate columns
    import re as _re
    lat_col = next((h for h in header if _re.fullmatch(r"lat(itude)?", str(h), _re.I)), None)
    lon_col = next((h for h in header if _re.fullmatch(r"(lon|lng|long|longitude)", str(h), _re.I)), None)
    if not lat_col or not lon_col:
        raise HTTPException(400, "Latitude/Longitude columns required in rows")
    lat_idx = header.index(lat_col)
    lon_idx = header.index(lon_col)

    # Optional existing Territory column
    terr_col = next((h for h in header if _re.search(r"territory", str(h), _re.I)), None)
    terr_idx = header.index(terr_col) if terr_col else None
    keep_names = bool(body.get("keepNames", True))

    # Build coords array
    coords_list: list[list[float]] = []
    valid_mask: list[bool] = []
    for r in data:
        try:
            lat = float(r[lat_idx]); lng = float(r[lon_idx])
            coords_list.append([lat, lng]); valid_mask.append(True)
        except Exception:
            coords_list.append([math.nan, math.nan]); valid_mask.append(False)
    coords = np.array(coords_list, dtype=float)

    n = int(len(coords))
    if n == 0:
        raise HTTPException(400, "No data rows")

    # Determine k
    numTerritories = body.get("numTerritories")
    if numTerritories is None:
        if terr_col:
            uniq = sorted(set(str(r[terr_idx]) for r in data if terr_idx is not None))
            k = max(1, len([u for u in uniq if u.strip()]))
        else:
            k = 3
    else:
        k = int(numTerritories)
    if k < 1:
        raise HTTPException(400, "numTerritories must be >= 1")

    # Bounds
    minCalls = body.get("minCalls")
    maxCalls = body.get("maxCalls")
    if minCalls is None or maxCalls is None:
        even = int(math.ceil(n / max(1, k)))
        minCalls = 0
        maxCalls = max(1, even)

    # Filter valid coords
    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_idx:
        raise HTTPException(400, "No valid lat/lng rows")
    sub_coords = coords[valid_idx, :]

    # Run planner (no fixed resources)
    try:
        assign_sub, centers, _ = plan_territories(sub_coords, int(k), int(minCalls), int(maxCalls), None)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"territory planning failed: {e}")

    # Expand assignment to all rows (invalid rows keep original)
    new_assign_global = [-1] * n
    for j, i in enumerate(valid_idx):
        new_assign_global[i] = int(assign_sub[j]) if j < len(assign_sub) else -1

    # Map indices to names
    new_names: list[str] = [f"Territory {i+1}" for i in range(int(k))]
    if keep_names and terr_col:
        # Build overlap matrix between new clusters and existing names
        from collections import defaultdict
        old_names = [str(r[terr_idx] or "").strip() for r in data]
        uniq_old = sorted(list({nm for nm in old_names if nm}))
        if uniq_old:
            # Count overlaps
            counts = [[0 for _ in uniq_old] for __ in range(int(k))]
            for i, a in enumerate(new_assign_global):
                if a is None or a < 0 or i >= len(old_names):
                    continue
                on = old_names[i]
                if not on:
                    continue
                try:
                    j = uniq_old.index(on)
                except Exception:
                    continue
                if int(a) < int(k):
                    counts[int(a)][j] += 1
            # Greedy matching: assign each new cluster to the old name with max overlap, unique if possible
            used_old = set()
            mapping = {}
            # Sort clusters by their best overlap descending to stabilise
            order = list(range(int(k)))
            order.sort(key=lambda ci: max(counts[ci]) if counts[ci] else 0, reverse=True)
            for ci in order:
                # choose best available old name
                best_j = None; best_v = -1
                for j, v in enumerate(counts[ci]):
                    if j in used_old:
                        continue
                    if v > best_v:
                        best_v = v; best_j = j
                if best_j is not None and best_v > 0:
                    mapping[ci] = uniq_old[best_j]
                    used_old.add(best_j)
            # Fill unmapped clusters with new default names
            for ci in range(int(k)):
                if ci not in mapping:
                    mapping[ci] = f"Territory {ci+1}"
            new_names = [mapping[i] for i in range(int(k))]

    # Ensure Territory column exists
    if not terr_col:
        header.append("Territory")
        terr_col = "Territory"
        terr_idx = len(header) - 1
        for i in range(len(data)):
            data[i].append("")
    else:
        terr_idx = header.index(terr_col)

    # Write new names
    for i in range(n):
        a = new_assign_global[i]
        if a is None or a < 0:
            continue
        if int(a) < len(new_names):
            data[i][terr_idx] = new_names[int(a)]

    # Optional: recompute TerritoryRank and SortOrder if present
    try:
        if "TerritoryRank" in header:
            # new rank from name order in new_names
            rank_map = {nm: (idx+1) for idx, nm in enumerate(new_names)}
            tr_idx = header.index("TerritoryRank")
            for i in range(n):
                nm = str(data[i][terr_idx] or "")
                data[i][tr_idx] = str(rank_map.get(nm, 0))
        if "SortOrder" in header:
            so_idx = header.index("SortOrder")
            # Keep DayNumber and Sequence if present
            dn_idx = header.index("DayNumber") if "DayNumber" in header else None
            seq_idx = header.index("Sequence") if "Sequence" in header else None
            rn_idx = header.index("ResourceRank") if "ResourceRank" in header else (header.index("TerritoryRank") if "TerritoryRank" in header else None)
            for i in range(n):
                try:
                    rn = int(str(data[i][rn_idx])) if rn_idx is not None else 0
                    dn = int(str(data[i][dn_idx])) if dn_idx is not None and str(data[i][dn_idx]).strip() else 0
                    sq = int(str(data[i][seq_idx])) if seq_idx is not None and str(data[i][seq_idx]).strip() else 0
                    data[i][so_idx] = str(int(rn)*10_000_000 + int(dn)*100 + int(sq))
                except Exception:
                    pass
    except Exception:
        pass

    return {"rows": [header] + data, "suggested_locations": centers}

def _apply_filter_count(rows: list[list], filt: dict) -> int:
    try:
        hdr = [str(h) for h in rows[0]]
        data = rows[1:]
        col = filt.get('column')
        op = (filt.get('op') or 'equals').lower()
        val = str(filt.get('value') or '')
        ci = bool(filt.get('caseInsensitive', True))
        if col not in hdr:
            return 0
        idx = hdr.index(col)
        import re as _re
        cnt = 0
        for r in data:
            cell = '' if idx >= len(r) else str(r[idx] or '')
            a = cell
            b = val
            if ci:
                a = a.lower()
                b = b.lower()
            ok = False
            if op == 'equals':
                ok = (a == b)
            elif op == 'startswith':
                ok = a.startswith(b)
            elif op == 'contains':
                ok = (b in a)
            elif op == 'regex':
                try:
                    ok = bool(_re.search(val, cell))
                except Exception:
                    ok = False
            if ok:
                cnt += 1
        return cnt
    except Exception:
        return 0

def _heuristic_transform_plan(instruction: str, rows: list[list]) -> Optional[dict]:
    if not rows or len(rows) < 2:
        return None
    import re as _re
    hdr = [str(h) for h in rows[0]]
    cm = _detect_columns(hdr)
    ql = (instruction or '').strip()
    # Example: "change all calls in HN postcode to Mark" -> filter startsWith on postcode; update resource=Mark
    m = _re.search(r"change\s+all\s+calls\s+in\s+(?:the\s+)?([A-Za-z0-9 ]+)\s+(?:post\s*code|postal\s*code|zipcode|zip|eircode)\s+to\s+([A-Za-z0-9 .\-']+)", ql, _re.I)
    if m:
        area = (m.group(1) or '').strip()
        assignee = (m.group(2) or '').strip()
        pc_col = cm.get('postcode')
        res_col = cm.get('resource') or cm.get('territory')
        if pc_col and res_col and area and assignee:
            plan = {
                "filters": [{"column": pc_col, "op": "startsWith", "value": area, "caseInsensitive": True}],
                "updates": [{"column": res_col, "setTo": assignee}],
                "note": "Heuristic plan from instruction"
            }
            plan["matchedEstimate"] = _apply_filter_count(rows, plan["filters"][0])
            return plan
    # Pattern: set <column> to <value> for/in (the) <area> (postcode)
    m2 = _re.search(r"set\s+([A-Za-z0-9 _\-]+)\s+to\s+([A-Za-z0-9 .\-']+)\s+(?:for|in)\s+(?:the\s+)?([A-Za-z0-9 ]+)(?:\s+(?:post\s*code|postal\s*code|zipcode|zip|eircode))?", ql, _re.I)
    if m2:
        col_hint = (m2.group(1) or '').strip()
        set_val = (m2.group(2) or '').strip()
        area = (m2.group(3) or '').strip()
        hdr = [str(h) for h in rows[0]]
        pc_col = cm.get('postcode')
        if pc_col and area and set_val and col_hint:
            target_col = _best_column_match(hdr, col_hint) or cm.get('resource') or cm.get('territory')
            if target_col:
                plan = {
                    "filters": [{"column": pc_col, "op": "startsWith", "value": area, "caseInsensitive": True}],
                    "updates": [{"column": target_col, "setTo": set_val}],
                    "note": "Heuristic plan from instruction"
                }
                plan["matchedEstimate"] = _apply_filter_count(rows, plan["filters"][0])
                return plan
    return None

@app.post("/api/ask/transform")
async def ask_transform(request: Request):
    body = await request.json()
    instruction = str(body.get("instruction", "")).strip()
    rows = body.get("rows") or []
    include_schema = bool(body.get("includeSchema", True))
    max_rows = int(body.get("maxRows", 200))
    if not instruction:
        raise HTTPException(400, "instruction required")
    # Try LLM if available to return a strict JSON plan
    use_openai = bool(os.getenv("OPENAI_API_KEY")) or (bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT")))
    plan = None
    provider_used = "fallback"
    if use_openai:
        try:
            import importlib as _importlib
            import os as _os
            _openai_mod = _importlib.import_module('openai')
            OpenAIClass = getattr(_openai_mod, 'OpenAI')
            provider = _os.getenv("LLM_PROVIDER", "openai").lower()
            if provider == "azure":
                client = OpenAIClass(
                    api_key=_os.getenv("AZURE_OPENAI_API_KEY"),
                    base_url=f"{_os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/deployments/{_os.getenv('AZURE_OPENAI_DEPLOYMENT')}",
                )
                model = _os.getenv("AZURE_OPENAI_DEPLOYMENT")
                provider_used = "azure-openai"
            else:
                client = OpenAIClass(api_key=_os.getenv("OPENAI_API_KEY"))
                model = _os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                provider_used = "openai"
            hdr = [str(h) for h in (rows[0] if rows else [])]
            schema = _infer_schema(rows) if include_schema else {"columns": []}
            examples = _format_rows_for_prompt(rows, max_rows=max_rows)
            system = (
                "You produce ONLY a compact JSON plan that transforms a table.\n"
                "JSON shape: {filters:[{column,op,value,caseInsensitive}], updates:[{column,setTo}], note:string}.\n"
                "Allowed ops: equals|startsWith|contains|regex. Use AND across filters. No prose."
            )
            user = (
                f"Instruction: {instruction}\n"
                f"Columns: {json.dumps(hdr, ensure_ascii=False)}\n"
                f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
                f"SampleRows(JSONL up to {max_rows}):\n{examples}"
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content if resp and resp.choices else None
            if content:
                # Try to extract JSON
                try:
                    plan = json.loads(content.strip())
                except Exception:
                    # Try to find a JSON block
                    import re as _re
                    m = _re.search(r"\{[\s\S]*\}", content)
                    if m:
                        plan = json.loads(m.group(0))
        except Exception:
            plan = None
    if plan is None:
        plan = _heuristic_transform_plan(instruction, rows)
    if plan is None:
        # Return 200 with no plan so clients can gracefully fallback to Q&A
        return {"plan": None, "provider": provider_used, "message": "no_transform"}
    # Estimate matches
    est = 0
    try:
        if plan.get('filters'):
            # AND across filters
            hdr = [str(h) for h in rows[0]]
            data = rows[1:]
            idxs = {f['column']: hdr.index(f['column']) for f in plan['filters'] if f.get('column') in hdr}
            import re as _re
            for r in data:
                ok = True
                for f in plan['filters']:
                    if f.get('column') not in idxs:
                        ok = False; break
                    val = str(f.get('value') or '')
                    op = (f.get('op') or 'equals').lower()
                    ci = bool(f.get('caseInsensitive', True))
                    cell = '' if idxs[f['column']] >= len(r) else str(r[idxs[f['column']]] or '')
                    a = cell; b = val
                    if ci:
                        a = a.lower(); b = b.lower()
                    if op == 'equals': ok = ok and (a == b)
                    elif op == 'startswith': ok = ok and a.startswith(b)
                    elif op == 'contains': ok = ok and (b in a)
                    elif op == 'regex':
                        try: ok = ok and bool(_re.search(val, cell))
                        except Exception: ok = False
                    if not ok: break
                if ok: est += 1
    except Exception:
        est = 0
    plan['matchedEstimate'] = est
    return {"plan": plan, "provider": provider_used}