#!/usr/bin/env bash
set -euo pipefail

# join-swarm.sh
# Runs on worker nodes to join the Docker Swarm managed by the first Pi.
# Usage: sudo ./join-swarm.sh <manager_ip> <worker_token>

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="${SCRIPT_DIR}/check-docker-version.sh"
ALLOW_DOCKER_VERSION_MISMATCH=${ALLOW_DOCKER_VERSION_MISMATCH:-}
if [[ -x "$CHECK_SCRIPT" ]]; then
  if [[ -n "$ALLOW_DOCKER_VERSION_MISMATCH" ]]; then
    if ! "$CHECK_SCRIPT" --warn-only; then
      echo "Docker version mismatch detected; continuing because ALLOW_DOCKER_VERSION_MISMATCH is set." >&2
    fi
  else
    if ! "$CHECK_SCRIPT"; then
      echo "Docker version check failed. Set ALLOW_DOCKER_VERSION_MISMATCH=1 to skip blocking." >&2
      exit 2
    fi
  fi
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <manager_ip> <worker_token>" >&2
  exit 1
fi

MANAGER_IP=$1
TOKEN=$2

if docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null | grep -q '^active$'; then
  echo "Node already part of a swarm; run 'docker swarm leave --force' before joining another." >&2
  exit 1
fi

if ! ping -c1 -W1 "$MANAGER_IP" >/dev/null 2>&1; then
  echo "Cannot reach manager at $MANAGER_IP. Check networking/DNS." >&2
  exit 2
fi

docker swarm join --token "$TOKEN" "$MANAGER_IP":2377

docker info | grep -E "Swarm:|NodeID"
