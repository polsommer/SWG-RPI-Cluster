import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import redis
import requests
from PIL import Image
from functools import lru_cache


# ---------------------------------------------------------
# Logging Layer
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


# ---------------------------------------------------------
# Config Loader
# ---------------------------------------------------------
class Config:
    def __init__(self) -> None:
        self.redis_url: str = os.environ.get("REDIS_URL", "redis://redis-bus:6379/0")
        self.input_channel: str = os.environ.get("INPUT_CHANNEL", "texture:upscaled")
        self.output_channel: str = os.environ.get("OUTPUT_CHANNEL", "texture:metadata")

        self.shared_path: Path = Path(
            os.environ.get("SHARED_PATH", "/srv/textures")
        ).resolve()

        self.metadata_suffix: str = os.environ.get("METADATA_SUFFIX", ".metadata.json")
        if not self.metadata_suffix.startswith("."):
            self.metadata_suffix = "." + self.metadata_suffix

        self.llm_backend: str = os.environ.get("LLM_BACKEND", "echo").lower()
        self.llm_endpoint: Optional[str] = os.environ.get("LLM_ENDPOINT")
        self.llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.llm_api_key: Optional[str] = os.environ.get("LLM_API_KEY")
        self.llm_timeout: float = float(os.environ.get("LLM_TIMEOUT", "30"))

        self.heartbeat_seconds: int = int(os.environ.get("HEARTBEAT_SECONDS", "30"))
        self.default_tags: List[str] = [
            t.strip() for t in os.environ.get("DEFAULT_TAGS", "").split(",") if t.strip()
        ]


# ---------------------------------------------------------
# Metadata Analyzer
# ---------------------------------------------------------
class MetadataAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.redis = redis.Redis.from_url(self.config.redis_url, decode_responses=True)

        # More stable pubsub
        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(self.config.input_channel)

        logging.info(f"Subscribed to: {self.config.input_channel}")
        logging.info(f"Publishing to: {self.config.output_channel}")

    # ---------------------------------------------------------
    # Main Event Loop
    # ---------------------------------------------------------
    def run(self) -> None:
        self.publish_status("online", "metadata analyzer ready")
        last_heartbeat = time.time()

        while True:
            try:
                message = self.pubsub.get_message(timeout=1.5)
                now = time.time()

                if message and message.get("type") == "message":
                    self.handle_message(message.get("data"))

                if now - last_heartbeat >= self.config.heartbeat_seconds:
                    self.publish_status("heartbeat", "alive")
                    last_heartbeat = now

            except Exception as exc:
                logging.exception("Fatal error in main loop")
                self.publish_error("fatal-loop", str(exc), time.time())
                time.sleep(3)

    # ---------------------------------------------------------
    # Message Handling
    # ---------------------------------------------------------
    def handle_message(self, raw_data: object) -> None:
        start = time.time()
        text = self._decode_message(raw_data)

        file_path = self._extract_path(text)
        if not file_path:
            self.publish_error("invalid-message", text, start)
            return

        info = self._load_texture_info(file_path)
        if not info:
            self.publish_error("missing-file", str(file_path), start)
            return

        prompt = self._build_prompt(info)

        caption, tags = self._safe_llm_call(prompt)

        metadata = {
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(self.config.shared_path)),
            "caption": caption,
            "tags": tags,
            "width": info["width"],
            "height": info["height"],
            "size_bytes": info["size_bytes"],
            "format": info["format"],
            "metrics": {
                "llm_latency_seconds": caption.get("llm_latency", None)
                if isinstance(caption, dict)
                else None,
                "end_to_end_latency_seconds": round(time.time() - start, 3),
            },
        }

        self._persist_metadata(file_path, metadata)
        self.publish_metadata(metadata)

    # ---------------------------------------------------------
    # Message Decoding
    # ---------------------------------------------------------
    def _decode_message(self, raw: object) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    # ---------------------------------------------------------
    # Path Extraction
    # ---------------------------------------------------------
    def _extract_path(self, message: str) -> Optional[Path]:
        # Try JSON first
        try:
            obj = json.loads(message)
            if isinstance(obj, dict):
                for key in ("path", "output", "file"):
                    if key in obj:
                        return self._validate_path(obj[key])
        except json.JSONDecodeError:
            pass

        # Fallback: scan tokens
        for token in message.split():
            if token.startswith("/"):
                validated = self._validate_path(token)
                if validated:
                    return validated

        return None

    def _validate_path(self, raw_path: str) -> Optional[Path]:
        p = Path(raw_path)
        if not p.is_absolute():
            p = self.config.shared_path / p

        try:
            p = p.resolve(strict=True)
            _ = p.relative_to(self.config.shared_path)
            return p
        except Exception:
            return None

    # ---------------------------------------------------------
    # Texture Info Loading w/ Cache
    # ---------------------------------------------------------
    @lru_cache(maxsize=4000)
    def _load_texture_info(self, path: Path) -> Optional[Dict[str, object]]:
        if not path.exists():
            return None

        stat = path.stat()
        info: Dict[str, object] = {
            "size_bytes": stat.st_size,
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

    # ---------------------------------------------------------
    # Prompt Builder
    # ---------------------------------------------------------
    def _build_prompt(self, info: Dict[str, object]) -> str:
        parts = [
            f"Format: {info['format'].upper()}",
            f"Resolution: {info['width']}x{info['height']}" if info["width"] else "Resolution: unknown",
            f"Size: {info['size_bytes']} bytes",
        ]

        return (
            "Generate a concise caption and 5–10 simple tags for this game texture. "
            "Return ONLY JSON {caption: string, tags: string[]}.\n"
            + " | ".join(parts)
        )

    # ---------------------------------------------------------
    # LLM Wrapper (with backoff)
    # ---------------------------------------------------------
    def _safe_llm_call(self, prompt: str) -> Tuple[str, List[str]]:
        try:
            return self._call_llm(prompt)
        except Exception as exc:
            logging.warning(f"LLM error: {exc}")
            self.publish_error("llm-failure", str(exc), time.time())
            # Safe fallback
            return prompt[:120] + "...", (self.config.default_tags or ["texture", "auto"])

    def _call_llm(self, prompt: str) -> Tuple[str, List[str]]:
        if self.config.llm_backend == "echo":
            return (
                prompt[:120] + ("..." if len(prompt) > 120 else ""),
                self.config.default_tags or ["texture", "auto-generated"],
            )

        url = self.config.llm_endpoint or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": "Provide JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            url, json=payload, headers=headers, timeout=self.config.llm_timeout
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        parsed = json.loads(content)
        caption = parsed.get("caption", "auto-generated texture description")
        tags = [t for t in parsed.get("tags", []) if isinstance(t, str)]

        # Add default tags uniquely
        for t in self.config.default_tags:
            if t not in tags:
                tags.append(t)

        return caption, tags

    # ---------------------------------------------------------
    # Metadata Persistence
    # ---------------------------------------------------------
    def _persist_metadata(self, file_path: Path, metadata: Dict[str, object]) -> None:
        target = file_path.with_suffix(file_path.suffix + self.config.metadata_suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # ---------------------------------------------------------
    # Redis Publishing Helpers
    # ---------------------------------------------------------
    def publish_metadata(self, metadata: Dict[str, object]) -> None:
        self.redis.publish(
            self.config.output_channel,
            json.dumps({"event": "metadata", "metadata": metadata}),
        )

    def publish_status(self, status: str, message: str) -> None:
        self.redis.publish(
            self.config.output_channel,
            json.dumps(
                {
                    "event": status,
                    "message": message,
                    "timestamp": time.time(),
                    "worker": "metadata-analyzer",
                }
            ),
        )

    def publish_error(self, code: str, details: str, observed_at: float) -> None:
        self.redis.publish(
            self.config.output_channel,
            json.dumps(
                {
                    "event": "error",
                    "code": code,
                    "details": details,
                    "timestamp": time.time(),
                    "latency_seconds": round(time.time() - observed_at, 3),
                }
            ),
        )


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
def main() -> None:
    MetadataAnalyzer(Config()).run()


if __name__ == "__main__":
    main()
