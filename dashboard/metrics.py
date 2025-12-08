from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def exponential_smoothing(values: List[float], alpha: float = 0.35) -> Optional[float]:
    if not values:
        return None
    smoothed = float(values[0])
    for value in values[1:]:
        smoothed = alpha * float(value) + (1 - alpha) * smoothed
    return smoothed


class MetricsStore:
    def __init__(self, path: Path, history_limit: int = 200) -> None:
        self.path = path
        self.history_limit = history_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"host": [], "services": {}})

    def _load(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {"host": [], "services": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def record_response_time(self, service_id: str, response_time_ms: float) -> None:
        data = self._load()
        service = data.setdefault("services", {}).setdefault(service_id, {})
        service["latest_response_ms"] = float(response_time_ms)
        self._save(data)

    def record_snapshot(self, host: Dict[str, Any], services: List[Dict[str, Any]]) -> Dict[str, Any]:
        timestamp = dt.datetime.utcnow().isoformat() + "Z"
        data = self._load()

        host_entry = {
            "timestamp": timestamp,
            "cpu_percent": host.get("cpu_percent"),
            "mem_percent": host.get("mem_percent"),
        }
        data.setdefault("host", []).append(host_entry)
        data["host"] = data["host"][-self.history_limit :]

        for svc in services:
            svc_id = svc.get("id")
            svc_name = svc.get("name")
            if not svc_id:
                continue
            svc_data = data.setdefault("services", {}).setdefault(
                svc_id,
                {
                    "name": svc_name,
                    "history": [],
                    "auto_scale": {
                        "enabled": False,
                        "min_replicas": 1,
                        "max_replicas": max(1, svc.get("replicas") or svc.get("desired") or 1),
                        "target_response_ms": 500,
                        "target_cpu_percent": 75,
                        "cooldown_seconds": 90,
                        "last_scaled_at": None,
                    },
                },
            )
            svc_data["name"] = svc_name
            history = svc_data.setdefault("history", [])
            history.append(
                {
                    "timestamp": timestamp,
                    "running": svc.get("running"),
                    "desired": svc.get("desired"),
                    "replicas": svc.get("replicas"),
                    "task_failures": svc.get("task_states", {}).get("failed", 0),
                    "response_time_ms": svc_data.get("latest_response_ms"),
                }
            )
            svc_data["history"] = history[-self.history_limit :]

        self._save(data)
        return data

    def host_forecast(self) -> Dict[str, Optional[float]]:
        data = self._load()
        host_history = data.get("host", [])
        cpu_values = [h.get("cpu_percent") for h in host_history if h.get("cpu_percent") is not None]
        mem_values = [h.get("mem_percent") for h in host_history if h.get("mem_percent") is not None]
        return {
            "cpu_percent": exponential_smoothing(cpu_values) if cpu_values else None,
            "mem_percent": exponential_smoothing(mem_values) if mem_values else None,
        }

    def service_forecast(self, service_id: str) -> Dict[str, Any]:
        data = self._load()
        svc = data.get("services", {}).get(service_id)
        if not svc:
            return {}
        history = svc.get("history", [])
        response_values = [h.get("response_time_ms") for h in history if h.get("response_time_ms") is not None]
        failure_values = [h.get("task_failures") for h in history if h.get("task_failures") is not None]
        running_values = [h.get("running") for h in history if h.get("running") is not None]

        return {
            "service_id": service_id,
            "name": svc.get("name"),
            "auto_scale": svc.get("auto_scale", {}),
            "predicted_response_ms": exponential_smoothing(response_values) if response_values else None,
            "predicted_failures": exponential_smoothing(failure_values) if failure_values else None,
            "predicted_running": exponential_smoothing(running_values) if running_values else None,
            "latest": history[-1] if history else None,
        }

    def update_auto_scale(self, service_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        data = self._load()
        svc = data.setdefault("services", {}).setdefault(service_id, {"history": []})
        existing = svc.get("auto_scale", {})
        existing.update(
            {
                "enabled": bool(config.get("enabled", existing.get("enabled", False))),
                "min_replicas": max(1, int(config.get("min_replicas", existing.get("min_replicas", 1)))),
                "max_replicas": max(1, int(config.get("max_replicas", existing.get("max_replicas", 1)))),
                "target_response_ms": max(1, int(config.get("target_response_ms", existing.get("target_response_ms", 500)))),
                "target_cpu_percent": max(1, int(config.get("target_cpu_percent", existing.get("target_cpu_percent", 75)))),
                "cooldown_seconds": max(15, int(config.get("cooldown_seconds", existing.get("cooldown_seconds", 90)))),
                "last_scaled_at": existing.get("last_scaled_at"),
            }
        )
        svc["auto_scale"] = existing
        self._save(data)
        return existing

    def set_last_scaled(self, service_id: str, timestamp: str) -> None:
        data = self._load()
        svc = data.get("services", {}).get(service_id)
        if not svc:
            return
        auto_scale = svc.setdefault("auto_scale", {})
        auto_scale["last_scaled_at"] = timestamp
        self._save(data)

    def recommended_replicas(
        self,
        service: Dict[str, Any],
        host_forecast: Dict[str, Optional[float]],
    ) -> Optional[int]:
        svc_forecast = self.service_forecast(service.get("id"))
        auto_scale = svc_forecast.get("auto_scale") or {}
        if not auto_scale.get("enabled"):
            return None

        current = service.get("replicas") or service.get("desired") or 1
        min_repl = auto_scale.get("min_replicas", 1)
        max_repl = auto_scale.get("max_replicas", max(1, current))
        target_rt = auto_scale.get("target_response_ms", 500)
        target_cpu = auto_scale.get("target_cpu_percent", 75)

        predicted_rt = svc_forecast.get("predicted_response_ms")
        predicted_failures = svc_forecast.get("predicted_failures")
        predicted_running = svc_forecast.get("predicted_running") or current
        predicted_cpu = host_forecast.get("cpu_percent") if host_forecast else None

        desired = current
        if predicted_rt and predicted_rt > target_rt:
            desired = max(desired, current + 1)
        if predicted_cpu and predicted_cpu > target_cpu:
            desired = max(desired, current + 1)
        if predicted_failures and predicted_failures > 0:
            desired = max(desired, current + 1)

        if predicted_rt and predicted_rt < (0.5 * target_rt) and predicted_cpu and predicted_cpu < (0.5 * target_cpu):
            desired = min(desired, current - 1)

        desired = max(min_repl, min(int(round(desired)), max_repl))
        if desired == predicted_running and desired == current:
            return None
        if desired != current:
            return desired
        return None

    def all_service_forecasts(self) -> List[Dict[str, Any]]:
        data = self._load().get("services", {})
        return [self.service_forecast(svc_id) for svc_id in data.keys()]
