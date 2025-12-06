from __future__ import annotations

import datetime as dt
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import docker
import psutil
from flask import Flask, jsonify, request, send_from_directory
from subprocess import CalledProcessError, CompletedProcess, run

app = Flask(__name__, static_folder="static", static_url_path="")
client = docker.from_env()
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def summarize_service_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for task in tasks:
        state = task.get("Status", {}).get("State", "unknown")
        summary[state] = summary.get(state, 0) + 1
    return summary


def summarize_services() -> List[Dict[str, Any]]:
    services = []
    for svc in client.services.list():
        attrs = svc.attrs
        spec = attrs.get("Spec", {})
        mode = spec.get("Mode", {})
        replicas = None
        desired = None
        if "Replicated" in mode:
            replicas = mode["Replicated"].get("Replicas")
            desired = replicas
        tasks = svc.tasks()
        running = len([t for t in tasks if t.get("Status", {}).get("State") == "running"])
        desired = desired or len(tasks)

        task_states = summarize_service_tasks(tasks)
        template = spec.get("TaskTemplate", {}).get("ContainerSpec", {})
        services.append(
            {
                "id": svc.id[:12],
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
            }
        )
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


@app.route("/api/summary")
def summary() -> Any:
    info = client.info()
    swarm = info.get("Swarm", {})
    nodes = collect_nodes()
    services = summarize_services()
    host = host_metrics()
    insights = build_insights(host, nodes, services)
    data = {
        "cluster": {
            "node_id": swarm.get("NodeID"),
            "local_node_state": swarm.get("LocalNodeState"),
            "control_available": swarm.get("ControlAvailable"),
            "managers": swarm.get("Managers"),
            "nodes": swarm.get("Nodes"),
            "cluster_info": swarm.get("Cluster", {}),
        },
        "host": host,
        "nodes": nodes,
        "services": services,
        "insights": insights,
        "redis_ping": worker_echo_rates(),
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }
    return jsonify(data)


@app.route("/api/admin")
def admin() -> Any:
    swarm = client.api.inspect_swarm()
    join_tokens = swarm.get("JoinTokens", {})
    managers = client.info().get("Swarm", {}).get("RemoteManagers", [])
    advertise_addr = managers[0].get("Addr") if managers else None

    return jsonify(
        {
            "tokens": join_tokens,
            "advertise_addr": advertise_addr,
            "nodes": collect_nodes(),
            "services": summarize_services(),
            "cluster": swarm,
            "repo": repo_update_status(False),
        }
    )


@app.post("/api/admin/rotate-tokens")
def rotate_tokens() -> Any:
    try:
        client.swarm.update(rotate_manager_token=True, rotate_worker_token=True)
        swarm = client.api.inspect_swarm()
        return jsonify({"tokens": swarm.get("JoinTokens", {})})
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


@app.get("/api/admin/repo-status")
def repo_status() -> Any:
    return jsonify(repo_update_status(False))


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
