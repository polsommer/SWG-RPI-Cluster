#!/usr/bin/env bash
set -euo pipefail

# deploy-demo.sh
# Deploys the overlay network and demo stack that exercises inter-node traffic.

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

STACK_FILE=${1:-stack/demo.yml}
STACK_NAME=${2:-swg-demo}

if [[ ! -f "$STACK_FILE" ]]; then
  echo "Stack file not found: $STACK_FILE" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="${SCRIPT_DIR}/../dashboard"
DASHBOARD_IMAGE="swg-dashboard:local"

if ! docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null | grep -q '^active$'; then
  echo "This node is not in an active swarm. Initialize or join first." >&2
  exit 2
fi

if ! docker network ls --format '{{.Name}}' | grep -q '^cluster_net$'; then
  echo "Creating attachable overlay network 'cluster_net'."
  docker network create --driver overlay --attachable cluster_net
fi

if [[ -d "$DASHBOARD_DIR" ]]; then
  echo "Building dashboard image (${DASHBOARD_IMAGE}) from ${DASHBOARD_DIR}".
  docker build -t "$DASHBOARD_IMAGE" "$DASHBOARD_DIR"
else
  echo "Dashboard source missing at ${DASHBOARD_DIR}; skipping build." >&2
fi

echo "Deploying stack ${STACK_NAME} from ${STACK_FILE}"
docker stack deploy -c "$STACK_FILE" "$STACK_NAME"

echo "Current services:"
docker service ls

echo "Tip: watch logs with 'docker service logs -f ${STACK_NAME}_echo-worker'"
