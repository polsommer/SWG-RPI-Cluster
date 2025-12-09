#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# deploy-textures.sh
# Deploys the distributed texture pipeline and dynamically
# scales GPU upscaler replicas according to swarm node labels.
# -------------------------------------------------------

# -------- Common Logging Helper ------------------------------------
log() {
  local level="$1"; shift
  local ts
  ts=$(date -Iseconds)
  echo "$ts [$level] $*"
}

fail() {
  log "ERROR" "$1"
  exit 1
}

# -------------------------------------------------------
# Require root
# -------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  fail "Please run as root (use sudo)."
fi

# -------------------------------------------------------
# Inputs
# -------------------------------------------------------
STACK_FILE=${1:-stack/textures.yml}
STACK_NAME=${2:-swg-textures}
DRY_RUN=${DRY_RUN:-0}

# -------------------------------------------------------
# Validate stack file
# -------------------------------------------------------
if [[ ! -f "$STACK_FILE" ]]; then
  fail "Stack file not found: $STACK_FILE"
fi

if ! grep -q 'version:' "$STACK_FILE"; then
  log "WARN" "Stack file may not be valid Compose/Swarm format."
fi

# -------------------------------------------------------
# Preflight checks
# -------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "Docker is not installed."

if ! docker info >/dev/null 2>&1; then
  fail "Docker is not running or not accessible."
fi

# Must be a manager to deploy stacks
if ! docker info --format '{{.Swarm.ControlAvailable}}' | grep -qi true; then
  fail "This node is not a Docker Swarm manager. Run on the manager node."
fi

if ! docker info --format '{{.Swarm.LocalNodeState}}' | grep -q '^active$'; then
  fail "This node is not part of an active swarm."
fi

# Ensure NFS mount exists for textures
if [[ ! -d /srv/textures ]]; then
  log "WARN" "/srv/textures does not exist. Creating..."
  mkdir -p /srv/textures
fi

# -------------------------------------------------------
# Ensure overlay network exists
# -------------------------------------------------------
if ! docker network ls --format '{{.Name}}' | grep -q '^cluster_net$'; then
  log "INFO" "Creating overlay network 'cluster_net'..."
  docker network create \
    --driver overlay \
    --attachable \
    --opt encrypted=true \
    cluster_net
  log "INFO" "Network 'cluster_net' created."
else
  log "INFO" "Overlay network 'cluster_net' already exists."
fi

# -------------------------------------------------------
# GPU detection
# -------------------------------------------------------
gpu_nodes=$(
  docker node ls \
    --filter node.label=gpu=true \
    --format '{{.ID}}' | wc -l | tr -d ' '
)

export ESRGAN_GPU_REPLICAS=${gpu_nodes:-0}
log "INFO" "GPU-enabled nodes detected: $ESRGAN_GPU_REPLICAS"

if [[ $ESRGAN_GPU_REPLICAS -eq 0 ]]; then
  log "WARN" "No GPU nodes detected → esrgan-upscaler-gpu replicas will be 0."
  log "INFO" "CPU upscalers remain active."
fi

# -------------------------------------------------------
# Dry run mode
# -------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  log "INFO" "DRY RUN mode — stack WILL NOT be deployed."
  log "INFO" "GPU replicas that would be used: ${ESRGAN_GPU_REPLICAS}"
  exit 0
fi

# -------------------------------------------------------
# Deploy stack with retries
# -------------------------------------------------------
log "INFO" "Deploying stack '$STACK_NAME' from '$STACK_FILE'..."

max_attempts=3
attempt=1
success=0

while (( attempt <= max_attempts )); do
  if docker stack deploy -c "$STACK_FILE" "$STACK_NAME"; then
    success=1
    break
  else
    log "WARN" "Deploy attempt $attempt failed. Retrying in 5s..."
    sleep 5
  fi
  ((attempt++))
done

if [[ $success -ne 1 ]]; then
  fail "Stack deployment failed after ${max_attempts} attempts."
fi

# -------------------------------------------------------
# Report final state
# -------------------------------------------------------
log "INFO" "Stack '$STACK_NAME' deployed successfully."
log "INFO" "Current Swarm services:"
docker service ls

log "INFO" "Use 'docker service ps ${STACK_NAME}_<service>' to inspect service state."
