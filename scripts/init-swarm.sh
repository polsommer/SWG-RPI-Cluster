#!/usr/bin/env bash
set -euo pipefail

# init-swarm.sh
# Initializes the manager node for the three-node Raspberry Pi cluster and
# creates a shared overlay network for inter-node communication.

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

ADVERTISE_ADDR=${1:-}
if [[ -z "$ADVERTISE_ADDR" ]]; then
  ADVERTISE_ADDR=$(hostname -I | awk '{print $1}')
fi

echo "Using advertise address: $ADVERTISE_ADDR"
if docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null | grep -q '^active$'; then
  echo "Node already part of a swarm; refusing to re-initialize." >&2
  docker info --format 'Current state: {{.Swarm.LocalNodeState}} role={{.Swarm.ControlAvailable}} nodeID={{.Swarm.NodeID}}'
else
  docker swarm init --advertise-addr "$ADVERTISE_ADDR"
fi

echo "Creating attachable overlay network 'cluster_net' if missing..."
docker network create --driver overlay --attachable cluster_net || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="${SCRIPT_DIR}/../dashboard"
DASHBOARD_STACK_FILE="${SCRIPT_DIR}/../stack/dashboard.yml"
DASHBOARD_IMAGE="swg-dashboard:local"
DASHBOARD_STACK_NAME=${DASHBOARD_STACK_NAME:-swg-dashboard}

if [[ -d "$DASHBOARD_DIR" ]]; then
  echo "Building dashboard image (${DASHBOARD_IMAGE}) from ${DASHBOARD_DIR}."
  docker build -t "$DASHBOARD_IMAGE" "$DASHBOARD_DIR"
fi

if [[ -f "$DASHBOARD_STACK_FILE" ]]; then
  echo "Auto-launching the web control module using ${DASHBOARD_STACK_FILE} (stack: ${DASHBOARD_STACK_NAME})."
  docker stack deploy -c "$DASHBOARD_STACK_FILE" "$DASHBOARD_STACK_NAME"
fi

echo "Manager token:"
docker swarm join-token manager

echo "Worker token:"
docker swarm join-token worker
