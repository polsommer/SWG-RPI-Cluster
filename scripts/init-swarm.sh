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

echo "Manager token:"
docker swarm join-token manager

echo "Worker token:"
docker swarm join-token worker
