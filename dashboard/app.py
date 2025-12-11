from __future__ import annotations

import atexit
import base64
import datetime as dt
import copy
import json
import queue
import os
import shutil
import socket
import threading
import time
import fcntl
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import docker
import psutil
from flask import Flask, jsonify, request, send_from_directory
from subprocess import CalledProcessError, CompletedProcess, run
from docker.errors import DockerException

app = Flask(__name__, static_folder="static", static_url_path="")
try:
    client = docker.from_env()
except DockerException as exc:  # pragma: no cover - exercised in tests
    app.logger.warning("Docker client unavailable: %s", exc)

    class _UnavailableDockerClient:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(f"Docker client unavailable: {exc}") from exc

    client = _UnavailableDockerClient()
REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_STORE = Path(os.environ.get("METRICS_STORE", "/tmp/service_metrics.json"))
STATE_STORE = Path(os.environ.get("DASHBOARD_STATE_STORE", "/tmp/dashboard_state.json"))
METRICS_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
AUTH_TOKEN = os.environ.get("DASHBOARD_AUTH_TOKEN", "").strip()
PROTECTED_PATH_PREFIXES = ("/api/admin", "/api/nodes", "/api/services")
CONTROLLER_LOCK_FILE = Path(os.environ.get("DASHBOARD_CONTROLLER_LOCK", "/tmp/dashboard_controller.lock"))
controller_lock_handle = None


def _bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _extract_auth_token() -> str | None:
    header = request.headers.get("Authorization", "").strip()
    if header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()

    if header.lower().startswith("basic "):
        try:
            raw = base64.b64decode(header.split(" ", 1)[1]).decode()
            username, _, password = raw.partition(":")
            return password or username or None
        except Exception:  # noqa: BLE001
            return None

    return None


@app.before_request
def require_authentication() -> Any:
    path = request.path or ""
    if not any(path.startswith(prefix) for prefix in PROTECTED_PATH_PREFIXES):
        return None

    if not AUTH_TOKEN:
        return jsonify({"error": "Authentication token is not configured"}), 401

    provided = _extract_auth_token()
    if not provided or provided != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    return None


def _load_metrics_store() -> Dict[str, Any]:
    if not METRICS_STORE.exists():
        return {"services": {}, "last_updated": None}
    try:
        with METRICS_STORE.open() as fh:
            return {"services": {}, "last_updated": None} | (json.load(fh) or {})
    except Exception:  # noqa: BLE001
        return {"services": {}, "last_updated": None}


def _persist_metrics_store(store: Dict[str, Any]) -> None:
    METRICS_STORE.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOCK:
        with METRICS_STORE.open("w") as fh:
            json.dump(store, fh, indent=2)


def _load_state_store() -> Dict[str, Any]:
    if not STATE_STORE.exists():
        return {}

    try:
        with STATE_LOCK:
            with STATE_STORE.open() as fh:
                return json.load(fh) or {}
    except Exception:  # noqa: BLE001
        return {}


def _persist_state_store(store: Dict[str, Any]) -> None:
    STATE_STORE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_LOCK:
        with STATE_STORE.open("w") as fh:
            json.dump(store, fh, indent=2)


def _persist_swarm_state(tokens: Dict[str, Any] | None, advertise_addr: str | None) -> Dict[str, Any]:
    store = _load_state_store()
    swarm_state = store.get("swarm", {})
    if tokens:
        swarm_state["tokens"] = tokens
    if advertise_addr:
        swarm_state["advertise_addr"] = advertise_addr
    store["swarm"] = swarm_state
    _persist_state_store(store)
    return swarm_state


def load_swarm_state() -> Dict[str, Any]:
    store = _load_state_store()
    return store.get("swarm", {}) if isinstance(store, dict) else {}


def exponential_smoothing(values: List[float], alpha: float, default: float | None = None) -> float | None:
    if not values:
        return default
    smoothed = values[0]
    for value in values[1:]:
        smoothed = alpha * value + (1 - alpha) * smoothed
    return smoothed


# Auto-remediation configuration
AUTO_REMEDIATE_ENABLED = _bool_env("AUTO_REMEDIATE_ENABLED", False)
AUTO_REMEDIATE_INTERVAL = int(os.environ.get("AUTO_REMEDIATE_INTERVAL", "30"))
AUTO_REMEDIATE_MAX_RESTARTS = int(os.environ.get("AUTO_REMEDIATE_MAX_RESTARTS", "3"))
AUTO_REMEDIATE_COOLDOWN_SECONDS = int(os.environ.get("AUTO_REMEDIATE_COOLDOWN_SECONDS", "180"))
AUTO_REMEDIATE_MIN_MANAGERS = int(os.environ.get("AUTO_REMEDIATE_MIN_MANAGERS", "1"))
AUTO_REMEDIATE_EVENT_BUFFER = int(os.environ.get("AUTO_REMEDIATE_EVENT_BUFFER", "200"))

AUTO_SCALE_ENABLED = _bool_env("AUTO_SCALE_ENABLED", True)
AUTO_SCALE_INTERVAL = int(os.environ.get("AUTO_SCALE_INTERVAL", "30"))
AUTO_SCALE_SMOOTHING = float(os.environ.get("AUTO_SCALE_SMOOTHING", "0.35"))
AUTO_SCALE_MIN_REPLICAS = int(os.environ.get("AUTO_SCALE_MIN_REPLICAS", "1"))
AUTO_SCALE_HISTORY_LIMIT = int(os.environ.get("AUTO_SCALE_HISTORY_LIMIT", "200"))
AUTO_SCALE_FAILURE_WEIGHT = float(os.environ.get("AUTO_SCALE_FAILURE_WEIGHT", "1.5"))

DEFAULT_CONTROLLER_CONFIG = {
    "auto_remediate_enabled": AUTO_REMEDIATE_ENABLED,
    "auto_remediate_interval": AUTO_REMEDIATE_INTERVAL,
    "auto_remediate_max_restarts": AUTO_REMEDIATE_MAX_RESTARTS,
    "auto_remediate_cooldown_seconds": AUTO_REMEDIATE_COOLDOWN_SECONDS,
    "auto_remediate_min_managers": AUTO_REMEDIATE_MIN_MANAGERS,
    "auto_remediate_event_buffer": AUTO_REMEDIATE_EVENT_BUFFER,
    "auto_scale_enabled": AUTO_SCALE_ENABLED,
    "auto_scale_interval": AUTO_SCALE_INTERVAL,
    "auto_scale_smoothing": AUTO_SCALE_SMOOTHING,
    "auto_scale_min_replicas": AUTO_SCALE_MIN_REPLICAS,
    "auto_scale_history_limit": AUTO_SCALE_HISTORY_LIMIT,
    "auto_scale_failure_weight": AUTO_SCALE_FAILURE_WEIGHT,
}


def _persist_controller_config(config: Dict[str, Any]) -> None:
    state = _load_state_store()
    state["controller_config"] = config
    _persist_state_store(state)


def _apply_controller_config(config: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
    global AUTO_REMEDIATE_ENABLED
    global AUTO_REMEDIATE_INTERVAL
    global AUTO_REMEDIATE_MAX_RESTARTS
    global AUTO_REMEDIATE_COOLDOWN_SECONDS
    global AUTO_REMEDIATE_MIN_MANAGERS
    global AUTO_REMEDIATE_EVENT_BUFFER
    global AUTO_SCALE_ENABLED
    global AUTO_SCALE_INTERVAL
    global AUTO_SCALE_SMOOTHING
    global AUTO_SCALE_MIN_REPLICAS
    global AUTO_SCALE_HISTORY_LIMIT
    global AUTO_SCALE_FAILURE_WEIGHT
    global REMEDIATION_EVENTS

    merged = {**DEFAULT_CONTROLLER_CONFIG, **config}

    AUTO_REMEDIATE_ENABLED = bool(merged.get("auto_remediate_enabled"))
    AUTO_REMEDIATE_INTERVAL = max(5, int(merged.get("auto_remediate_interval", 30)))
    AUTO_REMEDIATE_MAX_RESTARTS = max(1, int(merged.get("auto_remediate_max_restarts", 3)))
    AUTO_REMEDIATE_COOLDOWN_SECONDS = max(30, int(merged.get("auto_remediate_cooldown_seconds", 180)))
    AUTO_REMEDIATE_MIN_MANAGERS = max(1, int(merged.get("auto_remediate_min_managers", 1)))
    AUTO_REMEDIATE_EVENT_BUFFER = max(50, int(merged.get("auto_remediate_event_buffer", 200)))

    AUTO_SCALE_ENABLED = bool(merged.get("auto_scale_enabled", True))
    AUTO_SCALE_INTERVAL = max(5, int(merged.get("auto_scale_interval", 30)))
    AUTO_SCALE_SMOOTHING = float(merged.get("auto_scale_smoothing", 0.35))
    AUTO_SCALE_MIN_REPLICAS = max(1, int(merged.get("auto_scale_min_replicas", 1)))
    AUTO_SCALE_HISTORY_LIMIT = max(20, int(merged.get("auto_scale_history_limit", 200)))
    AUTO_SCALE_FAILURE_WEIGHT = float(merged.get("auto_scale_failure_weight", 1.5))

    REMEDIATION_EVENTS = deque(list(REMEDIATION_EVENTS), maxlen=AUTO_REMEDIATE_EVENT_BUFFER)

    current = controller_config()
    if persist:
        _persist_controller_config(current)
    return current


def _hydrate_controller_config() -> Dict[str, Any]:
    store = _load_state_store()
    stored = store.get("controller_config") if isinstance(store, dict) else None
    if isinstance(stored, dict):
        return _apply_controller_config(stored, persist=False)
    return controller_config()


def controller_config() -> Dict[str, Any]:
    return {
        "auto_remediate_enabled": AUTO_REMEDIATE_ENABLED,
        "auto_remediate_interval": AUTO_REMEDIATE_INTERVAL,
        "auto_remediate_max_restarts": AUTO_REMEDIATE_MAX_RESTARTS,
        "auto_remediate_cooldown_seconds": AUTO_REMEDIATE_COOLDOWN_SECONDS,
        "auto_remediate_min_managers": AUTO_REMEDIATE_MIN_MANAGERS,
        "auto_remediate_event_buffer": AUTO_REMEDIATE_EVENT_BUFFER,
        "auto_scale_enabled": AUTO_SCALE_ENABLED,
        "auto_scale_interval": AUTO_SCALE_INTERVAL,
        "auto_scale_smoothing": AUTO_SCALE_SMOOTHING,
        "auto_scale_min_replicas": AUTO_SCALE_MIN_REPLICAS,
        "auto_scale_history_limit": AUTO_SCALE_HISTORY_LIMIT,
        "auto_scale_failure_weight": AUTO_SCALE_FAILURE_WEIGHT,
    }


TEXTURE_LLM_NAME = os.environ.get("TEXTURE_LLM_NAME", "Texture LLM")
TEXTURE_LLM_ENDPOINT = os.environ.get("TEXTURE_LLM_ENDPOINT")
TEXTURE_LLM_ENABLED = _bool_env("TEXTURE_LLM_ENABLED", True)

REMEDIATION_EVENTS: deque[Dict[str, Any]] = deque(maxlen=AUTO_REMEDIATE_EVENT_BUFFER)
SERVICE_ATTEMPTS: Dict[str, Dict[str, Any]] = {}
remediation_lock = threading.Lock()
remediation_thread_started = False
scaler_thread_started = False
scale_events: deque[Dict[str, Any]] = deque(maxlen=100)
controller_lock = threading.Lock()
onboarding_lock = threading.Lock()
onboarding_logs: deque[Dict[str, Any]] = deque(maxlen=500)
onboarding_state: Dict[str, Any] = {
    "status": "idle",
    "manager": None,
    "workers": [],
    "ssh_user": "pi",
    "ssh_key_path": None,
    "advertise_addr": None,
    "started_at": None,
    "updated_at": None,
    "error": None,
    "validation": None,
    "tokens": {},
}

_hydrate_controller_config()


class BackgroundTaskRunner:
    def __init__(self) -> None:
        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, func, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.queue.put((func, args, kwargs))

    def _worker(self) -> None:
        while True:
            func, args, kwargs = self.queue.get()
            try:
                func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                app.logger.exception("Background task failed: %s", exc)
            finally:
                self.queue.task_done()


task_runner = BackgroundTaskRunner()


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def onboarding_snapshot() -> Dict[str, Any]:
    with onboarding_lock:
        state = dict(onboarding_state)
        state["logs"] = list(onboarding_logs)
    return state


def record_onboarding_log(target: str, message: str, level: str = "info") -> None:
    entry = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "target": target,
        "level": level,
        "message": message,
    }
    with onboarding_lock:
        onboarding_logs.append(entry)
        onboarding_state["updated_at"] = entry["ts"]


def set_onboarding_state(**changes: Any) -> Dict[str, Any]:
    with onboarding_lock:
        onboarding_state.update(changes)
        onboarding_state["updated_at"] = dt.datetime.utcnow().isoformat() + "Z"
        state = dict(onboarding_state)
        state["logs"] = list(onboarding_logs)
    return state


def collect_nodes() -> List[Dict[str, Any]]:
    nodes = []
    for node in client.nodes.list():
        attrs = node.attrs
        status = attrs.get("Status", {})
        spec = attrs.get("Spec", {})
        desc = attrs.get("Description", {})
        resources = desc.get("Resources", {})
        manager = attrs.get("ManagerStatus", {})
        nodes.append(
            {
                "id": node.id[:12],
                "hostname": desc.get("Hostname") or spec.get("Name"),
                "addr": status.get("Addr"),
                "state": status.get("State"),
                "availability": spec.get("Availability"),
                "role": spec.get("Role"),
                "reachability": manager.get("Reachability"),
                "manager_addr": manager.get("Addr"),
                "cpu": resources.get("NanoCPUs", 0) / 1e9,
                "memory_gb": round(resources.get("MemoryBytes", 0) / 1024 / 1024 / 1024, 2),
                "engine_version": desc.get("Engine", {}).get("EngineVersion"),
                "last_heartbeat": manager.get("LastHeartbeat"),
                "platform": desc.get("Platform", {}).get("Architecture"),
                "os": desc.get("Platform", {}).get("OS"),
                "labels": spec.get("Labels", {}),
            }
        )
    return nodes


def node_health_state(node: Dict[str, Any]) -> str:
    if node.get("availability") == "drain":
        return "maintenance"
    if node.get("state") in {"down", "disconnected", "unknown"}:
        return "unreachable"
    if node.get("reachability") == "unreachable":
        return "unreachable"
    if node.get("state") not in {"ready", "active"}:
        return "degraded"
    return "healthy"


def summarize_node_health(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {"healthy": 0, "maintenance": 0, "degraded": 0, "unreachable": 0}
    for node in nodes:
        state = node_health_state(node)
        summary[state] = summary.get(state, 0) + 1
    summary["total"] = len(nodes)
    return summary


def remediation_config() -> Dict[str, Any]:
    cfg = controller_config()
    return {
        "enabled": cfg.get("auto_remediate_enabled", False),
        "interval_seconds": cfg.get("auto_remediate_interval"),
        "max_service_restarts": cfg.get("auto_remediate_max_restarts"),
        "cooldown_seconds": cfg.get("auto_remediate_cooldown_seconds"),
        "min_managers": cfg.get("auto_remediate_min_managers"),
        "event_buffer": cfg.get("auto_remediate_event_buffer"),
    }


def texture_llm_status() -> Dict[str, Any]:
    status = "online" if TEXTURE_LLM_ENABLED else "standby"
    headline = "Active remediation and insights" if TEXTURE_LLM_ENABLED else "Ready on demand"
    return {
        "name": TEXTURE_LLM_NAME,
        "enabled": TEXTURE_LLM_ENABLED,
        "status": status,
        "headline": headline,
        "endpoint": TEXTURE_LLM_ENDPOINT,
        "message": "Intelligent cluster guidance powered by Texture." if TEXTURE_LLM_ENABLED else "Enable Texture to enrich decisions.",
    }


def record_remediation_event(action: str, target: str, status: str, message: str) -> None:
    event = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "action": action,
        "target": target,
        "status": status,
        "message": message,
    }
    with remediation_lock:
        REMEDIATION_EVENTS.appendleft(event)
    app.logger.info("[auto-remediation] %s %s - %s", action, target, message)


def summarize_service_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for task in tasks:
        state = task.get("Status", {}).get("State", "unknown")
        summary[state] = summary.get(state, 0) + 1
    return summary


def summarize_service(service: docker.models.services.Service) -> Dict[str, Any]:
    attrs = service.attrs
    spec = attrs.get("Spec", {})
    mode = spec.get("Mode", {})
    replicas = None
    desired = None
    if "Replicated" in mode:
        replicas = mode["Replicated"].get("Replicas")
        desired = replicas
    start = time.perf_counter()
    tasks = service.tasks()
    task_query_ms = (time.perf_counter() - start) * 1000
    running = len([t for t in tasks if t.get("Status", {}).get("State") == "running"])
    desired = desired or len(tasks)

    task_states = summarize_service_tasks(tasks)
    template = spec.get("TaskTemplate", {}).get("ContainerSpec", {})
    return {
        "id": service.id[:12],
        "name": spec.get("Name"),
        "image": template.get("Image"),
        "mode": "replicated" if "Replicated" in mode else "global",
        "running": running,
        "desired": desired,
        "ports": attrs.get("Endpoint", {}).get("Ports", []),
        "updated": attrs.get("UpdatedAt"),
        "replicas": replicas,
        "task_states": task_states,
        "labels": template.get("Labels", {}),
        "service_labels": spec.get("Labels", {}),
        "env": template.get("Env", []),
        "task_query_ms": task_query_ms,
    }


def summarize_services() -> List[Dict[str, Any]]:
    services = []
    for svc in client.services.list():
        services.append(summarize_service(svc))
    return services


def evaluate_service_health(services: List[Dict[str, Any]]) -> Dict[str, Any]:
    under_replicated = [s for s in services if s.get("desired") and s.get("running") < s.get("desired")]
    unstable = [
        s
        for s in services
        if s.get("task_states", {}).get("failed", 0) or s.get("task_states", {}).get("shutdown", 0)
    ]

    unhealthy_ids = {s.get("id") for s in under_replicated + unstable}

    return {
        "total": len(services),
        "under_replicated": under_replicated,
        "unstable": unstable,
        "healthy": len(services) - len(unhealthy_ids),
    }


def _within_limits(service_id: str) -> bool:
    now = time.time()
    meta = SERVICE_ATTEMPTS.get(service_id, {"count": 0, "last": 0.0})
    if now - meta.get("last", 0) > AUTO_REMEDIATE_COOLDOWN_SECONDS:
        meta = {"count": 0, "last": 0.0}
    if meta["count"] >= AUTO_REMEDIATE_MAX_RESTARTS:
        return False
    SERVICE_ATTEMPTS[service_id] = meta
    return True


def _register_attempt(service_id: str) -> None:
    meta = SERVICE_ATTEMPTS.get(service_id, {"count": 0, "last": 0.0})
    meta["count"] += 1
    meta["last"] = time.time()
    SERVICE_ATTEMPTS[service_id] = meta


def _update_service_spec(
    service: docker.models.services.Service, modifier: Any
) -> Tuple[Dict[str, Any] | None, Exception | None]:
    try:
        service.reload()
        spec = copy.deepcopy(service.attrs.get("Spec", {}))
        modifier(spec)

        kwargs = {
            "task_template": spec.get("TaskTemplate"),
            "name": spec.get("Name"),
            "labels": spec.get("Labels"),
            "mode": spec.get("Mode"),
            "update_config": spec.get("UpdateConfig"),
            "networks": spec.get("Networks"),
            "endpoint_spec": spec.get("EndpointSpec"),
            "rollback_config": spec.get("RollbackConfig"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        service.update(**kwargs)
        service.reload()
        return summarize_service(service), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def remediate_services(services: List[Dict[str, Any]]) -> None:
    overrides = service_automation_overrides()
    health = evaluate_service_health(services)
    unhealthy = {s.get("id"): s for s in health.get("under_replicated", []) + health.get("unstable", [])}

    for service_id, service_data in unhealthy.items():
        settings = overrides.get(service_id, {})
        if settings.get("automation_paused") or not settings.get("auto_remediate", True):
            record_remediation_event(
                "service-skip",
                service_data.get("name", service_id),
                "skipped",
                "Automation paused for this service.",
            )
            continue

        if not _within_limits(service_id):
            record_remediation_event(
                "service-skip",
                service_data.get("name", service_id),
                "skipped",
                "Restart limit reached; waiting for cooldown",
            )
            continue

        try:
            service = client.services.get(service_id)
            desired = service_data.get("desired") or service_data.get("replicas")
            if service_data.get("mode") == "replicated" and desired and service_data.get("running") is not None:
                service.scale(desired)
            service.force_update()
            _register_attempt(service_id)
            record_remediation_event(
                "service-redeploy",
                service_data.get("name", service_id),
                "success",
                f"Force-updated after instability (running {service_data.get('running')} of {service_data.get('desired')}).",
            )
        except Exception as exc:  # noqa: BLE001
            record_remediation_event(
                "service-redeploy",
                service_data.get("name", service_id),
                "error",
                str(exc),
            )


def _drain_unreachable_nodes(nodes: List[Dict[str, Any]]) -> None:
    for node in nodes:
        if node_health_state(node) == "unreachable" and node.get("availability") != "drain":
            try:
                swarm_node = client.nodes.get(node.get("id"))
                spec = dict(swarm_node.attrs.get("Spec", {}))
                spec["Availability"] = "drain"
                swarm_node.update(spec)
                record_remediation_event(
                    "node-drain",
                    node.get("hostname", node.get("id", "unknown")),
                    "success",
                    "Node unreachable; drained to protect scheduling.",
                )
            except Exception as exc:  # noqa: BLE001
                record_remediation_event(
                    "node-drain",
                    node.get("hostname", node.get("id", "unknown")),
                    "error",
                    str(exc),
                )


def _promote_managers(nodes: List[Dict[str, Any]]) -> None:
    healthy_managers = [n for n in nodes if n.get("role") == "manager" and node_health_state(n) == "healthy"]
    if len(healthy_managers) >= AUTO_REMEDIATE_MIN_MANAGERS:
        return

    candidates = [
        n
        for n in nodes
        if n.get("role") == "worker"
        and node_health_state(n) == "healthy"
        and n.get("availability") != "drain"
    ]
    if not candidates:
        record_remediation_event(
            "manager-promote",
            "cluster",
            "skipped",
            "No healthy workers available for promotion.",
        )
        return

    try:
        target = candidates[0]
        swarm_node = client.nodes.get(target.get("id"))
        spec = dict(swarm_node.attrs.get("Spec", {}))
        spec["Role"] = "manager"
        swarm_node.update(spec)
        record_remediation_event(
            "manager-promote",
            target.get("hostname", target.get("id", "unknown")),
            "success",
            "Promoted worker to maintain manager quorum.",
        )
    except Exception as exc:  # noqa: BLE001
        record_remediation_event(
            "manager-promote",
            target.get("hostname", target.get("id", "unknown")),
            "error",
            str(exc),
        )


def run_remediation_cycle() -> None:
    nodes = collect_nodes()
    services = summarize_services()

    _drain_unreachable_nodes(nodes)
    _promote_managers(nodes)
    remediate_services(services)


def worker_echo_rates() -> Dict[str, Any]:
    redis_address = os.environ.get("REDIS_URL", "redis-bus")
    data = {"redis": redis_address, "reachable": False, "ping_ms": None}
    try:
        start = time.perf_counter()
        container = client.containers.run(
            image="redis:7-alpine",
            command=["redis-cli", "-h", redis_address, "PING"],
            remove=True,
            network=os.environ.get("OVERLAY_NETWORK", "cluster_net"),
        )
        duration_ms = (time.perf_counter() - start) * 1000
        data.update({"reachable": container.decode().strip() == "PONG", "ping_ms": duration_ms})
    except Exception:
        data.update({"reachable": False, "ping_ms": None})
    return data


def cluster_metrics(swarm: Dict[str, Any], nodes: List[Dict[str, Any]], services: List[Dict[str, Any]]) -> Dict[str, Any]:
    managers = int(swarm.get("Managers") or 0)
    total_nodes = len(nodes)
    workers = max(total_nodes - managers, 0)

    ready = len([n for n in nodes if n.get("state") in {"ready", "active"}])
    drained = len([n for n in nodes if n.get("availability") == "drain"])

    total_cpu = round(sum(n.get("cpu") or 0 for n in nodes), 2)
    total_mem = round(sum(n.get("memory_gb") or 0 for n in nodes), 2)

    total_services = len(services)
    replicated = len([s for s in services if s.get("mode") == "replicated"])
    global_services = len([s for s in services if s.get("mode") == "global"])
    running_tasks = sum(s.get("running") or 0 for s in services)
    desired_tasks = sum(s.get("desired") or 0 for s in services)

    return {
        "nodes": total_nodes,
        "managers": managers,
        "workers": workers,
        "ready_nodes": ready,
        "drained_nodes": drained,
        "cpu_total": total_cpu,
        "memory_total_gb": total_mem,
        "services": total_services,
        "replicated_services": replicated,
        "global_services": global_services,
        "running_tasks": running_tasks,
        "desired_tasks": desired_tasks,
    }


def host_metrics() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    load1, load5, load15 = os.getloadavg()
    uptime = format_duration(time.time() - psutil.boot_time())
    disk = psutil.disk_usage("/")

    return {
        "cpu_percent": cpu,
        "mem_percent": mem.percent,
        "mem_total_gb": round(mem.total / 1024 / 1024 / 1024, 2),
        "load": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "uptime": uptime,
        "disk_percent": disk.percent,
        "hostname": socket.gethostname(),
    }


def record_service_metrics(services: List[Dict[str, Any]], host: Dict[str, Any]) -> Dict[str, Any]:
    store = _load_metrics_store()
    services_bucket = store.setdefault("services", {})
    now = dt.datetime.utcnow().isoformat() + "Z"

    for svc in services:
        service_id = svc.get("id")
        if not service_id:
            continue

        bucket = services_bucket.setdefault(
            service_id,
            {
                "history": [],
                "name": svc.get("name"),
                "auto_scale": False,
                "auto_remediate": True,
                "automation_paused": False,
            },
        )
        bucket["name"] = svc.get("name")
        bucket.setdefault("history", [])
        bucket.setdefault("auto_remediate", True)
        bucket.setdefault("automation_paused", False)
        bucket.setdefault("auto_scale", False)

        task_states = svc.get("task_states", {}) or {}
        failures = int(task_states.get("failed", 0)) + int(task_states.get("shutdown", 0))

        bucket["history"].append(
            {
                "ts": now,
                "running": svc.get("running"),
                "desired": svc.get("desired"),
                "replicas": svc.get("replicas"),
                "failures": failures,
                "response_ms": svc.get("task_query_ms"),
                "host_cpu": host.get("cpu_percent"),
                "host_mem": host.get("mem_percent"),
            }
        )

        if len(bucket["history"]) > AUTO_SCALE_HISTORY_LIMIT:
            bucket["history"] = bucket["history"][-AUTO_SCALE_HISTORY_LIMIT:]

    store["last_updated"] = now
    _persist_metrics_store(store)
    return store


def service_automation_overrides() -> Dict[str, Dict[str, Any]]:
    store = _load_metrics_store()
    services_bucket = store.get("services", {}) if isinstance(store, dict) else {}
    overrides: Dict[str, Dict[str, Any]] = {}
    for service_id, bucket in services_bucket.items():
        overrides[service_id] = {
            "name": bucket.get("name"),
            "auto_scale": bool(bucket.get("auto_scale", False)),
            "auto_remediate": bool(bucket.get("auto_remediate", True)),
            "automation_paused": bool(bucket.get("automation_paused", False)),
        }
    return overrides


def update_service_automation(
    service_id: str,
    *,
    name: str | None = None,
    auto_scale: bool | None = None,
    auto_remediate: bool | None = None,
    automation_paused: bool | None = None,
) -> Dict[str, Any]:
    store = _load_metrics_store()
    services_bucket = store.setdefault("services", {})
    bucket = services_bucket.setdefault(
        service_id,
        {
            "history": [],
            "name": name,
            "auto_scale": False,
            "auto_remediate": True,
            "automation_paused": False,
        },
    )
    if name is not None:
        bucket["name"] = name
    if auto_scale is not None:
        bucket["auto_scale"] = bool(auto_scale)
    if auto_remediate is not None:
        bucket["auto_remediate"] = bool(auto_remediate)
    if automation_paused is not None:
        bucket["automation_paused"] = bool(automation_paused)

    services_bucket[service_id] = bucket
    _persist_metrics_store(store)

    return {
        "service": service_id,
        "auto_scale": bucket.get("auto_scale", False),
        "auto_remediate": bucket.get("auto_remediate", True),
        "automation_paused": bucket.get("automation_paused", False),
    }


def forecast_replicas(service_id: str, service_data: Dict[str, Any], current_replicas: int | None) -> Dict[str, Any]:
    history = service_data.get("history", [])[-20:]
    cpu_series = [float(h.get("host_cpu")) for h in history if h.get("host_cpu") is not None]
    mem_series = [float(h.get("host_mem")) for h in history if h.get("host_mem") is not None]
    failure_series = [float(h.get("failures", 0)) for h in history]
    response_series = [float(h.get("response_ms")) for h in history if h.get("response_ms")]

    predicted_cpu = exponential_smoothing(cpu_series, AUTO_SCALE_SMOOTHING, 0.0) or 0.0
    predicted_mem = exponential_smoothing(mem_series, AUTO_SCALE_SMOOTHING, 0.0) or 0.0
    predicted_failures = exponential_smoothing(failure_series, AUTO_SCALE_SMOOTHING, 0.0) or 0.0
    predicted_response = exponential_smoothing(response_series, AUTO_SCALE_SMOOTHING, 0.0)

    predicted_load = max(predicted_cpu, predicted_mem)
    replicas = max(current_replicas or AUTO_SCALE_MIN_REPLICAS, AUTO_SCALE_MIN_REPLICAS)

    if predicted_load > 75:
        replicas += 1
    elif predicted_load < 25 and replicas > AUTO_SCALE_MIN_REPLICAS:
        replicas -= 1

    if predicted_failures * AUTO_SCALE_FAILURE_WEIGHT >= 1:
        replicas = max(replicas, (current_replicas or AUTO_SCALE_MIN_REPLICAS) + 1)

    details = {
        "service_id": service_id,
        "predicted_cpu": predicted_cpu,
        "predicted_mem": predicted_mem,
        "predicted_failures": predicted_failures,
        "predicted_response_ms": predicted_response,
        "recommended_replicas": int(replicas),
    }
    return details


def build_forecast_snapshot(services: List[Dict[str, Any]]) -> Dict[str, Any]:
    store = _load_metrics_store()
    services_bucket = store.get("services", {})
    forecasts: List[Dict[str, Any]] = []

    for svc in services:
        service_id = svc.get("id")
        if not service_id:
            continue

        bucket = services_bucket.get(service_id, {"history": [], "auto_scale": False})
        recommendation = forecast_replicas(service_id, bucket, svc.get("replicas") or svc.get("desired"))
        forecasts.append(
            {
                **recommendation,
                "service_name": svc.get("name"),
                "auto_scale": bucket.get("auto_scale", False),
                "automation_paused": bucket.get("automation_paused", False),
                "current_replicas": svc.get("replicas") or svc.get("desired"),
                "recent": bucket.get("history", [])[-5:],
            }
        )

    return {"services": forecasts, "recorded_at": store.get("last_updated")}


def record_scale_event(service_name: str, desired: int | None, status: str, message: str) -> None:
    event = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "service": service_name,
        "replicas": desired,
        "status": status,
        "message": message,
    }
    scale_events.appendleft(event)
    app.logger.info("[autoscaler] %s -> %s (%s)", service_name, desired, message)


def update_service_replicas(service_id: str, replicas: int) -> None:
    service = client.services.get(service_id)
    mode = service.attrs.get("Spec", {}).get("Mode", {})
    if "Replicated" not in mode:
        raise ValueError("Only replicated services can be auto-scaled")

    service.update(mode={"Replicated": {"Replicas": replicas}}, fetch_current_spec=True)


def is_swarm_manager() -> bool:
    try:
        info = client.info()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Unable to determine swarm manager status: %s", exc)
        return False

    swarm = info.get("Swarm", {})
    is_manager = swarm.get("ControlAvailable") and swarm.get("LocalNodeState") == "active"
    if not is_manager:
        app.logger.info(
            "Skipping controller startup: state=%s, control_available=%s",
            swarm.get("LocalNodeState"),
            swarm.get("ControlAvailable"),
        )
    return bool(is_manager)


def acquire_controller_lock() -> bool:
    global controller_lock_handle
    with controller_lock:
        if controller_lock_handle:
            return True

        try:
            CONTROLLER_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            handle = CONTROLLER_LOCK_FILE.open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            app.logger.info("Controller lock unavailable (%s): %s", CONTROLLER_LOCK_FILE, exc)
            return False

        controller_lock_handle = handle
        app.logger.info("Acquired controller lock at %s", CONTROLLER_LOCK_FILE)
        return True


def release_controller_lock() -> None:
    global controller_lock_handle
    with controller_lock:
        if not controller_lock_handle:
            return

        try:
            fcntl.flock(controller_lock_handle, fcntl.LOCK_UN)
        finally:
            controller_lock_handle.close()
            controller_lock_handle = None
            app.logger.info("Released controller lock at %s", CONTROLLER_LOCK_FILE)


def scaler_loop() -> None:
    while True:
        if not AUTO_SCALE_ENABLED:
            time.sleep(AUTO_SCALE_INTERVAL)
            continue

        try:
            services = summarize_services()
            host = host_metrics()
            store = record_service_metrics(services, host)
            for svc in services:
                svc_id = svc.get("id")
                if not svc_id:
                    continue

                bucket = store.get("services", {}).get(svc_id, {})
                if bucket.get("automation_paused"):
                    continue

                if not bucket.get("auto_scale"):
                    continue

                recommendation = forecast_replicas(svc_id, bucket, svc.get("replicas") or svc.get("desired"))
                target = max(int(recommendation.get("recommended_replicas", 0)), AUTO_SCALE_MIN_REPLICAS)
                current = svc.get("replicas") or svc.get("desired") or 0
                if target != current:
                    try:
                        update_service_replicas(svc_id, target)
                        record_scale_event(svc.get("name", svc_id), target, "applied", "Adjusted replicas from forecast")
                    except Exception as exc:  # noqa: BLE001
                        record_scale_event(svc.get("name", svc_id), target, "error", str(exc))
        except Exception as exc:  # noqa: BLE001
            record_scale_event("controller", None, "error", str(exc))

        time.sleep(AUTO_SCALE_INTERVAL)


def git_command(args: List[str]) -> CompletedProcess[str]:
    return run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True)


def repo_update_status(apply: bool = False) -> Dict[str, Any]:
    try:
        if shutil.which("git") is None:
            return {
                "error": "Git client is not installed on this host. Install git to enable auto-updates.",
                "applied": False,
            }

        git_command(["git", "fetch", "origin"])
        branch = git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        local_rev = git_command(["git", "rev-parse", "HEAD"]).stdout.strip()
        upstream = git_command(["git", "rev-parse", f"origin/{branch}"]).stdout.strip()
        ahead_behind = git_command(["git", "rev-list", "--left-right", "--count", f"HEAD...origin/{branch}"]).stdout.strip()
        behind = int(ahead_behind.split()[1]) if " " in ahead_behind else 0
        output = ""

        if apply and behind > 0:
            result = git_command(["git", "pull", "--rebase", "origin", branch])
            output = result.stdout + result.stderr

        return {
            "branch": branch,
            "local": local_rev,
            "remote": upstream,
            "behind": behind,
            "applied": apply and behind > 0,
            "output": output or None,
        }
    except FileNotFoundError:
        return {
            "error": "Unable to run git commands because the git executable could not be found.",
            "applied": False,
        }
    except CalledProcessError as exc:  # noqa: PERF203
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        return {"error": stderr or "Git command failed", "applied": False}


def execute_ssh_command(host: str, command: str, allowed_hosts: List[str]) -> Tuple[Dict[str, Any], int]:
    if host not in allowed_hosts:
        return {"error": "Host is not part of the swarm or is unreachable."}, 400

    if not command:
        return {"error": "Command is required."}, 400

    try:
        result = run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                f"pi@{host}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        return {"error": "SSH client is not available on this host."}, 500
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, 500

    payload: Dict[str, Any] = {
        "host": host,
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    status = 200 if result.returncode == 0 else 400
    return payload, status


def _ssh_base_command(host: str, user: str, key_path: str | None) -> List[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if key_path:
        cmd.extend(["-i", key_path])
    cmd.append(f"{user}@{host}")
    return cmd


def run_remote_command(
    host: str, user: str, key_path: str | None, remote_command: List[str], input_data: str | None = None
) -> CompletedProcess[str]:
    try:
        return run(
            _ssh_base_command(host, user, key_path) + remote_command,
            capture_output=True,
            text=True,
            timeout=180,
            input=input_data,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return CompletedProcess(args=remote_command, returncode=1, stdout="", stderr=str(exc))


def run_remote_script(host: str, user: str, key_path: str | None, script_path: Path, args: List[str] | None = None) -> CompletedProcess[str]:
    if not script_path.exists():
        return CompletedProcess(args=[], returncode=1, stdout="", stderr=f"Script not found: {script_path}")

    script_body = script_path.read_text()
    remote_cmd = ["bash", "-s", "--"] + list(args or [])
    return run_remote_command(host, user, key_path, remote_cmd, script_body)


def fetch_remote_swarm_facts(host: str, user: str, key_path: str | None) -> Dict[str, Any]:
    result = run_remote_command(host, user, key_path, ["docker", "info", "--format", "{{json .Swarm}}"])
    facts: Dict[str, Any] = {"return_code": result.returncode}
    if result.returncode != 0 or not result.stdout:
        facts["stderr"] = result.stderr
        return facts

    try:
        swarm_info = json.loads(result.stdout)
    except json.JSONDecodeError:
        facts["stderr"] = "Unable to parse swarm info"
        return facts

    tokens = swarm_info.get("JoinTokens", {})
    managers = swarm_info.get("RemoteManagers", [])
    advertise_addr = managers[0].get("Addr") if managers else None

    facts.update({"tokens": tokens, "advertise_addr": advertise_addr})
    return facts


def verify_swarm_health(
    manager_host: str, user: str, key_path: str | None, expected_nodes: int | None, advertise_addr: str | None
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"nodes": [], "overlay": False}
    listing = run_remote_command(
        manager_host,
        user,
        key_path,
        [
            "docker",
            "node",
            "ls",
            "--format",
            "{{.Hostname}} {{.Status}} {{.Availability}} {{.ManagerStatus}}",
        ],
    )
    if listing.stdout:
        for line in listing.stdout.splitlines():
            parts = line.split()
            summary["nodes"].append(
                {
                    "hostname": parts[0] if parts else "",
                    "status": parts[1] if len(parts) > 1 else "",
                    "availability": parts[2] if len(parts) > 2 else "",
                    "manager": parts[3] if len(parts) > 3 else "",
                }
            )

    network = run_remote_command(
        manager_host,
        user,
        key_path,
        ["docker", "network", "ls", "--filter", "name=cluster_net", "--format", "{{.Name}}"],
    )
    summary["overlay"] = "cluster_net" in [n.strip() for n in network.stdout.splitlines() if n.strip()]

    ready_nodes = [n for n in summary.get("nodes", []) if n.get("status", "").lower() == "ready"]
    target_nodes = expected_nodes or len(summary.get("nodes", []))
    summary["expected_nodes"] = target_nodes
    summary["ready_nodes"] = len(ready_nodes)
    summary["network_check"] = network.returncode == 0
    summary["node_check"] = listing.returncode == 0 and summary["ready_nodes"] >= target_nodes
    summary["advertise_addr"] = advertise_addr
    summary["ok"] = summary["network_check"] and summary["node_check"] and summary["overlay"]
    return summary


def run_onboarding_flow(
    manager_host: str,
    worker_hosts: List[str],
    ssh_user: str,
    key_path: str | None,
    advertise_addr: str | None,
) -> None:
    with onboarding_lock:
        onboarding_logs.clear()
    set_onboarding_state(
        status="running",
        manager=manager_host,
        workers=worker_hosts,
        ssh_user=ssh_user,
        ssh_key_path=key_path,
        advertise_addr=advertise_addr or manager_host,
        started_at=dt.datetime.utcnow().isoformat() + "Z",
        error=None,
        validation=None,
    )

    scripts_dir = REPO_ROOT / "scripts"
    record_onboarding_log(manager_host, "Installing Docker on manager via install-docker.sh")
    result = run_remote_script(manager_host, ssh_user, key_path, scripts_dir / "install-docker.sh")
    if result.returncode != 0:
        set_onboarding_state(status="error", error=result.stderr or "Manager install failed")
        return

    record_onboarding_log(manager_host, "Initializing swarm using init-swarm.sh")
    init_args = [advertise_addr] if advertise_addr else []
    init_result = run_remote_script(manager_host, ssh_user, key_path, scripts_dir / "init-swarm.sh", init_args)
    if init_result.returncode != 0:
        set_onboarding_state(status="error", error=init_result.stderr or "Swarm init failed")
        return

    facts = fetch_remote_swarm_facts(manager_host, ssh_user, key_path)
    tokens = facts.get("tokens", {}) if isinstance(facts.get("tokens"), dict) else {}
    advertise_addr = facts.get("advertise_addr") or advertise_addr or manager_host
    if tokens:
        record_onboarding_log(manager_host, "Captured swarm join tokens from manager")
    persist = _persist_swarm_state(tokens, advertise_addr)
    set_onboarding_state(tokens=persist.get("tokens", {}), advertise_addr=persist.get("advertise_addr"))

    for host in worker_hosts:
        record_onboarding_log(host, "Installing Docker on worker via install-docker.sh")
        worker_install = run_remote_script(host, ssh_user, key_path, scripts_dir / "install-docker.sh")
        if worker_install.returncode != 0:
            set_onboarding_state(status="error", error=worker_install.stderr or f"Install failed on {host}")
            return

        join_token = tokens.get("Worker") if tokens else None
        if not join_token:
            record_onboarding_log(host, "Worker token unavailable; cannot join swarm", level="warn")
            set_onboarding_state(status="error", error="Worker token unavailable")
            return

        record_onboarding_log(host, "Joining swarm via join-swarm.sh")
        join_result = run_remote_script(
            host,
            ssh_user,
            key_path,
            scripts_dir / "join-swarm.sh",
            [advertise_addr, join_token],
        )
        if join_result.returncode != 0:
            set_onboarding_state(status="error", error=join_result.stderr or f"Join failed for {host}")
            return

    # Health checks with retries
    validation: Dict[str, Any] | None = None
    for _ in range(3):
        validation = verify_swarm_health(manager_host, ssh_user, key_path, len(worker_hosts) + 1, advertise_addr)
        set_onboarding_state(validation=validation)
        if validation.get("ok"):
            break
        time.sleep(5)

    final_status = "ready" if validation and validation.get("ok") else "running"
    set_onboarding_state(status=final_status, validation=validation)
    record_onboarding_log(manager_host, "Onboarding complete" if final_status == "ready" else "Health checks pending")


def build_insights(host: Dict[str, Any], nodes: List[Dict[str, Any]], services: List[Dict[str, Any]]) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []

    # Host checks
    if host.get("cpu_percent", 0) > 85:
        messages.append({"level": "warn", "text": "CPU saturation detected on the host."})
    if host.get("mem_percent", 0) > 85:
        messages.append({"level": "warn", "text": "Memory usage is above 85%."})
    if host.get("disk_percent", 0) > 80:
        messages.append({"level": "warn", "text": "Disk usage is trending high."})
    if not messages:
        messages.append({"level": "ok", "text": "Host resources look healthy."})

    # Node checks
    node_summary = summarize_node_health(nodes)
    if node_summary.get("unreachable"):
        messages.append({"level": "warn", "text": f"{node_summary['unreachable']} node(s) unreachable."})
    if node_summary.get("maintenance"):
        messages.append({"level": "info", "text": f"{node_summary['maintenance']} node(s) in maintenance."})
    if not node_summary.get("unreachable") and not node_summary.get("maintenance"):
        messages.append({"level": "ok", "text": "All nodes are reachable."})

    # Service checks
    service_health = evaluate_service_health(services)
    if service_health.get("under_replicated"):
        affected = ", ".join(s.get("name", s.get("id")) for s in service_health["under_replicated"])
        messages.append({"level": "warn", "text": f"Under-replicated services: {affected}."})
    if service_health.get("unstable"):
        affected = ", ".join(s.get("name", s.get("id")) for s in service_health["unstable"])
        messages.append({"level": "warn", "text": f"Services with task failures: {affected}."})
    if not service_health.get("under_replicated") and not service_health.get("unstable"):
        messages.append({"level": "ok", "text": "Service replicas are healthy."})

    version_map = {}
    for node in nodes:
        version = node.get("engine_version")
        if version:
            version_map.setdefault(version, 0)
            version_map[version] += 1
    if len(version_map) > 1:
        versions = ", ".join(f"{v} ({c})" for v, c in version_map.items())
        messages.append({"level": "warn", "text": f"Engine versions differ across nodes: {versions}."})

    return {
        "messages": messages,
        "node_health": node_summary,
        "service_health": service_health,
    }


def remediation_loop() -> None:
    while True:
        if not AUTO_REMEDIATE_ENABLED:
            time.sleep(AUTO_REMEDIATE_INTERVAL)
            continue

        try:
            run_remediation_cycle()
        except Exception as exc:  # noqa: BLE001
            record_remediation_event("controller-error", "controller", "error", str(exc))

        time.sleep(AUTO_REMEDIATE_INTERVAL)


@app.route("/api/summary")
def summary() -> Any:
    info = client.info()
    swarm = info.get("Swarm", {})
    nodes = collect_nodes()
    services = summarize_services()
    host = host_metrics()
    metrics = cluster_metrics(swarm, nodes, services)
    insights = build_insights(host, nodes, services)
    metrics_store = record_service_metrics(services, host)
    forecasts = build_forecast_snapshot(services)
    cfg = controller_config()
    automation_overrides = service_automation_overrides()
    data = {
        "cluster": {
            "node_id": swarm.get("NodeID"),
            "local_node_state": swarm.get("LocalNodeState"),
            "control_available": swarm.get("ControlAvailable"),
            "managers": swarm.get("Managers"),
            "nodes": swarm.get("Nodes"),
            "cluster_info": swarm.get("Cluster", {}),
        },
        "cluster_metrics": metrics,
        "host": host,
        "nodes": nodes,
        "services": services,
        "insights": insights,
        "redis_ping": worker_echo_rates(),
        "llm": texture_llm_status(),
        "forecasts": forecasts,
        "remediation": {
            "config": remediation_config(),
            "events": list(REMEDIATION_EVENTS)[:20],
        },
        "autoscaler": {
            "enabled": cfg.get("auto_scale_enabled"),
            "interval_seconds": cfg.get("auto_scale_interval"),
            "smoothing": cfg.get("auto_scale_smoothing"),
            "min_replicas": cfg.get("auto_scale_min_replicas"),
            "history_limit": cfg.get("auto_scale_history_limit"),
            "failure_weight": cfg.get("auto_scale_failure_weight"),
            "last_metrics": metrics_store.get("last_updated"),
            "events": list(scale_events),
        },
        "service_automation": automation_overrides,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }
    return jsonify(data)


@app.route("/api/admin")
def admin() -> Any:
    stored_swarm = load_swarm_state()
    try:
        swarm = client.api.inspect_swarm()
        join_tokens = swarm.get("JoinTokens", {}) or stored_swarm.get("tokens", {})
        managers = client.info().get("Swarm", {}).get("RemoteManagers", [])
        advertise_addr = managers[0].get("Addr") if managers else stored_swarm.get("advertise_addr")
        _persist_swarm_state(join_tokens, advertise_addr)
    except Exception:  # noqa: BLE001
        swarm = {}
        join_tokens = stored_swarm.get("tokens", {}) if isinstance(stored_swarm, dict) else {}
        advertise_addr = stored_swarm.get("advertise_addr") if isinstance(stored_swarm, dict) else None
    services = summarize_services()
    controller_cfg = controller_config()
    automation_overrides = service_automation_overrides()

    return jsonify(
        {
            "tokens": join_tokens,
            "advertise_addr": advertise_addr,
            "nodes": collect_nodes(),
            "services": services,
            "cluster": swarm,
            "repo": repo_update_status(False),
            "forecasts": build_forecast_snapshot(services),
            "autoscaler_events": list(scale_events),
            "llm": texture_llm_status(),
            "controller_config": controller_cfg,
            "service_automation": automation_overrides,
            "onboarding": onboarding_snapshot(),
        }
    )


@app.get("/api/admin/onboarding/status")
def onboarding_status() -> Any:
    return jsonify(onboarding_snapshot())


@app.post("/api/admin/onboarding/start")
def onboarding_start() -> Any:
    payload = request.get_json(silent=True) or {}
    manager_host = str(payload.get("manager", "")).strip()
    raw_workers = payload.get("workers") or []
    if isinstance(raw_workers, str):
        combined = []
        for part in raw_workers.splitlines():
            combined.extend(segment.strip() for segment in part.split(","))
        raw_workers = combined
    worker_hosts = [str(w).strip() for w in raw_workers if str(w).strip()]
    ssh_user = str(payload.get("ssh_user", "pi") or "pi").strip()
    key_path = str(payload.get("ssh_key_path") or "").strip() or None
    advertise_addr = str(payload.get("advertise_addr") or "").strip() or manager_host

    if not manager_host:
        return jsonify({"error": "manager is required"}), 400

    with onboarding_lock:
        if onboarding_state.get("status") == "running":
            return jsonify({"error": "Onboarding already running"}), 409

    set_onboarding_state(
        status="queued",
        manager=manager_host,
        workers=worker_hosts,
        ssh_user=ssh_user,
        ssh_key_path=key_path,
        advertise_addr=advertise_addr,
        tokens=load_swarm_state().get("tokens", {}),
        error=None,
        validation=None,
    )
    record_onboarding_log(manager_host, "Queued onboarding workflow")
    task_runner.submit(run_onboarding_flow, manager_host, worker_hosts, ssh_user, key_path, advertise_addr)
    return jsonify(onboarding_snapshot())


@app.post("/api/admin/rotate-tokens")
def rotate_tokens() -> Any:
    try:
        client.swarm.update(rotate_manager_token=True, rotate_worker_token=True)
        swarm = client.api.inspect_swarm()
        tokens = swarm.get("JoinTokens", {})
        managers = client.info().get("Swarm", {}).get("RemoteManagers", [])
        advertise_addr = managers[0].get("Addr") if managers else None
        _persist_swarm_state(tokens, advertise_addr)
        return jsonify({"tokens": tokens, "advertise_addr": advertise_addr})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/nodes/<node_id>/availability")
def set_node_availability(node_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    availability = str(payload.get("availability", "")).lower()
    if availability not in {"active", "drain", "pause"}:
        return jsonify({"error": "availability must be one of: active, drain, pause"}), 400

    try:
        node = client.nodes.get(node_id)
        spec = dict(node.attrs.get("Spec", {}))
        spec["Availability"] = availability
        node.update(spec)
        return jsonify({"ok": True, "node_id": node_id, "availability": availability})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/nodes/<node_id>/role")
def set_node_role(node_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    role = str(payload.get("role", "")).lower()
    if role not in {"manager", "worker"}:
        return jsonify({"error": "role must be one of: manager, worker"}), 400

    try:
        node = client.nodes.get(node_id)
        spec = dict(node.attrs.get("Spec", {}))
        spec["Role"] = role
        node.update(spec)
        return jsonify({"ok": True, "node_id": node_id, "role": role})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/nodes/<node_id>/labels")
def set_node_labels(node_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    labels = payload.get("labels", {})
    if not isinstance(labels, dict):
        return jsonify({"error": "labels must be an object"}), 400

    try:
        node = client.nodes.get(node_id)
        spec = dict(node.attrs.get("Spec", {}))
        current_labels = spec.get("Labels", {}) or {}
        current_labels.update({str(k): str(v) for k, v in labels.items()})
        spec["Labels"] = current_labels
        node.update(spec)
        return jsonify({"ok": True, "node_id": node_id, "labels": current_labels})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/nodes/<node_id>/remove")
def remove_node(node_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force"))

    try:
        node = client.nodes.get(node_id)
        spec = dict(node.attrs.get("Spec", {}))
        if not force and spec.get("Availability") != "drain":
            spec["Availability"] = "drain"
            node.update(spec)

        client.api.remove_node(node_id, force=force)
        return jsonify({"ok": True, "node_id": node_id, "force": force})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/scale")
def scale_service(service_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    replicas = payload.get("replicas")
    if replicas is None:
        return jsonify({"error": "replicas is required"}), 400

    try:
        replicas_int = max(0, int(replicas))
    except (TypeError, ValueError):
        return jsonify({"error": "replicas must be an integer"}), 400

    try:
        service = client.services.get(service_id)
        mode = service.attrs.get("Spec", {}).get("Mode", {})
        if "Replicated" not in mode:
            return jsonify({"error": "Only replicated services can be scaled."}), 400

        service.scale(replicas_int)
        return jsonify({"ok": True, "service": service_id, "replicas": replicas_int})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/image")
def update_service_image(service_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    image = str(payload.get("image", "")).strip()
    tag = str(payload.get("tag", "")).strip()

    if not image:
        return jsonify({"error": "image is required"}), 400

    image_ref = f"{image}:{tag}" if tag else image

    try:
        service = client.services.get(service_id)

        def modifier(spec: Dict[str, Any]) -> None:
            container = spec.setdefault("TaskTemplate", {}).setdefault("ContainerSpec", {})
            container["Image"] = image_ref

        summary, error = _update_service_spec(service, modifier)
        if error:
            raise error
        return jsonify({"ok": True, "service": summary})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/env")
def update_service_env(service_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    env = payload.get("env", [])

    if isinstance(env, str):
        env = [line.strip() for line in env.splitlines() if line.strip()]

    if not isinstance(env, list):
        return jsonify({"error": "env must be a list of strings"}), 400

    cleaned_env = [str(item).strip() for item in env if str(item).strip()]

    try:
        service = client.services.get(service_id)

        def modifier(spec: Dict[str, Any]) -> None:
            container = spec.setdefault("TaskTemplate", {}).setdefault("ContainerSpec", {})
            container["Env"] = cleaned_env

        summary, error = _update_service_spec(service, modifier)
        if error:
            raise error
        return jsonify({"ok": True, "service": summary})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/labels")
def update_service_labels(service_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    labels = payload.get("labels", {})

    if isinstance(labels, str):
        parsed: Dict[str, str] = {}
        for line in labels.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                parsed[key] = value
        labels = parsed

    if not isinstance(labels, dict):
        return jsonify({"error": "labels must be an object"}), 400

    normalized = {str(k).strip(): str(v).strip() for k, v in labels.items() if str(k).strip()}

    try:
        service = client.services.get(service_id)

        def modifier(spec: Dict[str, Any]) -> None:
            container = spec.setdefault("TaskTemplate", {}).setdefault("ContainerSpec", {})
            container["Labels"] = normalized
            merged = {**(spec.get("Labels") or {}), **normalized}
            spec["Labels"] = merged

        summary, error = _update_service_spec(service, modifier)
        if error:
            raise error
        return jsonify({"ok": True, "service": summary})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/restart")
def restart_service(service_id: str) -> Any:
    try:
        service = client.services.get(service_id)

        def modifier(spec: Dict[str, Any]) -> None:
            template = spec.setdefault("TaskTemplate", {})
            force_update = int(template.get("ForceUpdate", 0) or 0)
            template["ForceUpdate"] = force_update + 1

        summary, error = _update_service_spec(service, modifier)
        if error:
            raise error
        return jsonify({"ok": True, "service": summary})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/rollback")
def rollback_service(service_id: str) -> Any:
    try:
        service = client.services.get(service_id)
        if hasattr(service, "rollback"):
            service.rollback()
        else:
            client.api.rollback_service(service_id)
        service.reload()
        return jsonify({"ok": True, "service": summarize_service(service)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/services/<service_id>/autoscale")
def set_service_autoscale(service_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))

    data = update_service_automation(service_id, auto_scale=enabled)
    return jsonify(data)


@app.post("/api/services/<service_id>/automation")
def set_service_automation(service_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    auto_scale = payload.get("auto_scale")
    auto_remediate = payload.get("auto_remediate")
    paused = payload.get("paused")

    data = update_service_automation(
        service_id,
        name=str(name) if name else None,
        auto_scale=bool(auto_scale) if auto_scale is not None else None,
        auto_remediate=bool(auto_remediate) if auto_remediate is not None else None,
        automation_paused=bool(paused) if paused is not None else None,
    )
    return jsonify(data)


@app.get("/api/admin/repo-status")
def repo_status() -> Any:
    return jsonify(repo_update_status(False))


@app.get("/api/admin/controller-config")
def get_controller_config() -> Any:
    return jsonify({"config": controller_config()})


@app.post("/api/admin/controller-config")
def set_controller_config() -> Any:
    payload = request.get_json(silent=True) or {}
    updated = _apply_controller_config(payload, persist=True)
    return jsonify({"config": updated})


@app.post("/api/admin/update")
def apply_update() -> Any:
    payload = request.get_json(silent=True) or {}
    apply = bool(payload.get("apply", True))
    status = repo_update_status(apply)
    code = 200 if status.get("error") is None else 500
    return jsonify(status), code


@app.post("/api/admin/ssh-run")
def run_ssh() -> Any:
    payload = request.get_json(silent=True) or {}
    host = str(payload.get("host", "")).strip()
    command = str(payload.get("command", "")).strip()

    nodes = collect_nodes()
    allowed_hosts = [n.get("addr") for n in nodes if n.get("addr")]

    result, status = execute_ssh_command(host, command, allowed_hosts)
    return jsonify(result), status


@app.route("/")
def root() -> Any:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def serve_static(filename: str) -> Any:
    return send_from_directory(app.static_folder, filename)


def ensure_remediation_thread() -> None:
    global remediation_thread_started
    if remediation_thread_started:
        return
    app.logger.info("Starting auto-remediation controller thread")
    remediation_thread = threading.Thread(target=remediation_loop, daemon=True)
    remediation_thread.start()
    remediation_thread_started = True


def ensure_scaler_thread() -> None:
    global scaler_thread_started
    if scaler_thread_started:
        return
    app.logger.info("Starting autoscaler controller thread")
    scaler_thread = threading.Thread(target=scaler_loop, daemon=True)
    scaler_thread.start()
    scaler_thread_started = True


def start_controllers_if_eligible() -> bool:
    if not is_swarm_manager():
        return False

    if not acquire_controller_lock():
        return False

    ensure_remediation_thread()
    ensure_scaler_thread()
    app.logger.info("Controller threads started on swarm manager node")
    return True


def create_app(start_controllers: bool = False) -> Flask:
    if start_controllers:
        start_controllers_if_eligible()
    return app


atexit.register(release_controller_lock)


if __name__ == "__main__":
    create_app(start_controllers=True)
    app.run(host="0.0.0.0", port=8081)
