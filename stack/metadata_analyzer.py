import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import redis
import requests
from PIL import Image


class Config:
    def __init__(self) -> None:
        self.redis_url = os.environ.get("REDIS_URL", "redis://redis-bus:6379/0")
        self.input_channel = os.environ.get("INPUT_CHANNEL", "texture:upscaled")
        self.output_channel = os.environ.get("OUTPUT_CHANNEL", "texture:metadata")
        self.shared_path = Path(os.environ.get("SHARED_PATH", "/srv/textures")).resolve()
        self.metadata_suffix = os.environ.get("METADATA_SUFFIX", ".metadata.json")
        self.llm_backend = os.environ.get("LLM_BACKEND", "echo").lower()
        self.llm_endpoint = os.environ.get("LLM_ENDPOINT")
        self.llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.llm_api_key = os.environ.get("LLM_API_KEY")
        self.llm_timeout = float(os.environ.get("LLM_TIMEOUT", "30"))
        self.heartbeat_seconds = int(os.environ.get("HEARTBEAT_SECONDS", "30"))
        self.default_tags = [t for t in os.environ.get("DEFAULT_TAGS", "").split(",") if t]


class MetadataAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.redis = redis.Redis.from_url(self.config.redis_url)
        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(self.config.input_channel)

    def run(self) -> None:
        self.publish_status("online", "llm metadata analyzer ready")
        last_heartbeat = 0.0
        while True:
            message = self.pubsub.get_message(timeout=1.0)
            now = time.time()
            if message and message.get("type") == "message":
                self.handle_message(message.get("data"))
            if now - last_heartbeat >= self.config.heartbeat_seconds:
                self.publish_status("heartbeat", "alive")
                last_heartbeat = now

    def handle_message(self, raw_data: object) -> None:
        received_at = time.time()
        text = self._decode_message(raw_data)
        file_path = self._extract_path(text)
        if not file_path:
            self.publish_error("unrecognized-message", text, received_at)
            return

        info = self._load_texture_info(file_path)
        if not info:
            self.publish_error("missing-texture", text, received_at)
            return

        prompt = self._build_prompt(info)
        llm_started_at = time.time()
        caption, tags = self._call_llm(prompt)
        llm_latency = time.time() - llm_started_at

        metadata = {
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(self.config.shared_path)),
            "caption": caption,
            "tags": tags,
            "width": info.get("width"),
            "height": info.get("height"),
            "size_bytes": info.get("size_bytes"),
            "format": info.get("format"),
            "metrics": {
                "llm_latency_seconds": llm_latency,
                "end_to_end_latency_seconds": time.time() - received_at,
            },
        }

        self._persist_metadata(file_path, metadata)
        self.publish_metadata(metadata)

    def _decode_message(self, raw_data: object) -> str:
        if isinstance(raw_data, bytes):
            return raw_data.decode("utf-8", errors="replace")
        return str(raw_data)

    def _extract_path(self, message: str) -> Optional[Path]:
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                candidate = data.get("path") or data.get("output") or data.get("file")
                if candidate:
                    return self._validate_path(candidate)
        except json.JSONDecodeError:
            pass

        for token in message.split():
            if token.startswith("/"):
                path = self._validate_path(token)
                if path:
                    return path
        return None

    def _validate_path(self, raw_path: str) -> Optional[Path]:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.config.shared_path / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.config.shared_path)
        except ValueError:
            return None
        return candidate if candidate.exists() else None

    def _load_texture_info(self, path: Path) -> Optional[Dict[str, object]]:
        if not path.exists() or not path.is_file():
            return None

        info: Dict[str, object] = {
            "size_bytes": path.stat().st_size,
            "format": path.suffix.lower().lstrip("."),
        }
        try:
            with Image.open(path) as img:
                info["width"], info["height"] = img.size
                info["mode"] = img.mode
        except Exception:
            info["width"] = None
            info["height"] = None
            info["mode"] = None
        return info

    def _build_prompt(self, info: Dict[str, object]) -> str:
        details = [
            f"File: {info.get('format', 'unknown').upper()} texture",
            f"Resolution: {info.get('width')}x{info.get('height')}" if info.get("width") and info.get("height") else "Resolution: unknown",
            f"Approximate size: {info.get('size_bytes')} bytes",
        ]
        base_prompt = (
            "You are an assistant that writes concise captions and 5-10 tags for game textures. "
            "Return JSON with keys 'caption' and 'tags' (array). Keep descriptions succinct."
        )
        return base_prompt + "\n" + " | ".join(details)

    def _call_llm(self, prompt: str) -> Tuple[str, List[str]]:
        if self.config.llm_backend == "echo":
            caption = prompt[:120] + ("..." if len(prompt) > 120 else "")
            tags = self.config.default_tags or ["texture", "auto-generated"]
            return caption, tags

        url = self.config.llm_endpoint or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": "You generate concise captions and tags for game textures."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.config.llm_timeout)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        parsed = json.loads(content)
        caption = parsed.get("caption") or "Automatically generated texture description"
        raw_tags = parsed.get("tags") or []
        tags = [t for t in raw_tags if isinstance(t, str)]
        if self.config.default_tags:
            tags.extend([t for t in self.config.default_tags if t not in tags])
        return caption, tags

    def _persist_metadata(self, file_path: Path, metadata: Dict[str, object]) -> None:
        target = file_path.with_suffix(file_path.suffix + self.config.metadata_suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def publish_metadata(self, metadata: Dict[str, object]) -> None:
        envelope = {
            "event": "metadata",
            "metadata": metadata,
        }
        self.redis.publish(self.config.output_channel, json.dumps(envelope))

    def publish_status(self, status: str, message: str) -> None:
        payload = {
            "event": status,
            "message": message,
            "timestamp": time.time(),
            "worker": "metadata-analyzer",
        }
        self.redis.publish(self.config.output_channel, json.dumps(payload))

    def publish_error(self, code: str, details: str, observed_at: float) -> None:
        payload = {
            "event": "error",
            "code": code,
            "details": details,
            "timestamp": time.time(),
            "latency_seconds": time.time() - observed_at,
        }
        self.redis.publish(self.config.output_channel, json.dumps(payload))


def main() -> None:
    analyzer = MetadataAnalyzer(Config())
    analyzer.run()


if __name__ == "__main__":
    main()
