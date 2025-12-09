#!/usr/bin/env bash
set -euo pipefail

TARGET_DOCKER_VERSION=${TARGET_DOCKER_VERSION:-"24.0.9"}
WARN_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warn-only)
      WARN_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed; unable to verify version." >&2
  exit $([[ "$WARN_ONLY" == true ]] && echo 0 || echo 1)
fi

CURRENT_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)
if [[ -z "$CURRENT_VERSION" ]]; then
  echo "Failed to detect Docker Engine version." >&2
  exit $([[ "$WARN_ONLY" == true ]] && echo 0 || echo 2)
fi

if [[ "$CURRENT_VERSION" == "$TARGET_DOCKER_VERSION" ]]; then
  echo "Docker Engine version $CURRENT_VERSION matches target $TARGET_DOCKER_VERSION."
  exit 0
fi

echo "Docker Engine version mismatch: current=$CURRENT_VERSION target=$TARGET_DOCKER_VERSION" >&2
if [[ "$WARN_ONLY" == true ]]; then
  exit 0
fi
exit 3
