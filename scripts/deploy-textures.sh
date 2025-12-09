#!/usr/bin/env bash
set -euo pipefail

# deploy-textures.sh
# Deploys the texture processing stack and scales esrgan-upscaler-gpu replicas
# to match the number of GPU-labeled swarm nodes.

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

STACK_FILE=${1:-stack/textures.yml}
STACK_NAME=${2:-swg-textures}

if [[ ! -f "$STACK_FILE" ]]; then
  echo "Stack file not found: $STACK_FILE" >&2
  exit 1
fi

if ! docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null | grep -q '^active$'; then
  echo "This node is not in an active swarm. Initialize or join first." >&2
  exit 2
fi

if ! docker network ls --format '{{.Name}}' | grep -q '^cluster_net$'; then
  echo "Creating attachable overlay network 'cluster_net'."
  docker network create --driver overlay --attachable cluster_net
fi

gpu_nodes=$(docker node ls --filter node.label=gpu=true --format '{{.ID}}' | wc -l | tr -d ' ')
export ESRGAN_GPU_REPLICAS=${gpu_nodes:-0}

echo "GPU-labeled nodes (gpu=true): ${ESRGAN_GPU_REPLICAS}"
if [[ ${ESRGAN_GPU_REPLICAS} -eq 0 ]]; then
  echo "No GPU nodes detected; esrgan-upscaler-gpu replicas will be set to 0. CPU upscalers remain available."
fi

echo "Deploying stack ${STACK_NAME} from ${STACK_FILE} with ${ESRGAN_GPU_REPLICAS} GPU replica(s)."
docker stack deploy -c "$STACK_FILE" "$STACK_NAME"

echo "Current services:"
docker service ls
