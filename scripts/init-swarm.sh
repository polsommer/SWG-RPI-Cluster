#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# init-swarm.sh
# Initializes a Docker Swarm manager node and prepares overlay network
# for a Raspberry Pi or mixed-arch cluster. Automatically deploys
# the dashboard stack if present.
# -------------------------------------------------------------------

# ---------------------------- COLORS --------------------------------
supports_color() {
  [[ -t 1 ]] && command -v tput >/dev/null && [[ $(tput colors) -ge 8 ]]
}
if supports_color; then
  RED=$(tput setaf 1)
  GREEN=$(tput setaf 2)
  YELLOW=$(tput setaf 3)
  CYAN=$(tput setaf 6)
  RESET=$(tput sgr0)
else
  RED=""; GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi

timestamp() { date -Iseconds; }
log()   { echo "$(timestamp) ${GREEN}INFO:${RESET} $*"; }
warn()  { echo "$(timestamp) ${YELLOW}WARN:${RESET} $*" >&2; }
fatal() { echo "$(timestamp) ${RED}ERROR:${RESET} $*" >&2; exit 1; }

# ---------------------------- Root Check -----------------------------
[[ $EUID -eq 0 ]] || fatal "Please run as root (sudo)."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------- Arguments ------------------------------
ADVERTISE_ADDR="${1:-}"
FORCE_REINIT="${FORCE_REINIT:-0}"

# ---------------------------- Docker Check ---------------------------
command -v docker >/dev/null 2>&1 || fatal "Docker is not installed."

if ! docker info >/dev/null 2>&1; then
  fatal "Docker daemon is not running."
fi

# ---------------------------- Version Check --------------------------
CHECK_SCRIPT="${SCRIPT_DIR}/check-docker-version.sh"
ALLOW_DOCKER_VERSION_MISMATCH="${ALLOW_DOCKER_VERSION_MISMATCH:-}"

if [[ -x "$CHECK_SCRIPT" ]]; then
  if [[ -n "$ALLOW_DOCKER_VERSION_MISMATCH" ]]; then
    if ! "$CHECK_SCRIPT" --warn-only; then
      warn "Docker version mismatch; continuing due to ALLOW_DOCKER_VERSION_MISMATCH=1"
    fi
  else
    if ! "$CHECK_SCRIPT"; then
      fatal "Docker Engine version check failed.
Set ALLOW_DOCKER_VERSION_MISMATCH=1 to bypass."
    fi
  fi
else
  warn "Version checker missing: $CHECK_SCRIPT"
fi

# ---------------------------- Detect IP ------------------------------
if [[ -z "$ADVERTISE_ADDR" ]]; then
  ADVERTISE_ADDR=$(hostname -I | awk '{print $1}')
fi

if [[ -z "$ADVERTISE_ADDR" ]]; then
  fatal "Could not auto-detect advertise address. Specify manually:
  sudo ./init-swarm.sh <ip>"
fi

log "Using advertise address: ${CYAN}${ADVERTISE_ADDR}${RESET}"

# ---------------------------- Swarm State Check ----------------------
SWARM_STATE=$(docker info --format '{{.Swarm.LocalNodeState}}' || echo "unknown")
IS_MANAGER=$(docker info --format '{{.Swarm.ControlAvailable}}' || echo "false")

if [[ "$SWARM_STATE" == "active" ]]; then
  if [[ "$IS_MANAGER" == "true" && "$FORCE_REINIT" == "1" ]]; then
    warn "This node is already a manager. FORCE_REINIT=1 → resetting swarm."
    docker swarm leave --force
    sleep 2
  else
    fatal "Node already in a swarm. To force reinit:
  FORCE_REINIT=1 sudo ./init-swarm.sh <ip>"
  fi
fi

# ---------------------------- Initialize Swarm -----------------------
log "Initializing Docker Swarm..."
docker swarm init --advertise-addr "$ADVERTISE_ADDR"

log "Swarm initialized successfully."

# ---------------------------- Create Overlay Network -----------------
log "Ensuring overlay network 'cluster_net' exists..."

docker network create \
  --driver overlay \
  --attachable \
  --opt encrypted=true \
  cluster_net \
  >/dev/null 2>&1 || true

log "Overlay network ready (cluster_net)."

# ---------------------------- Deploy Dashboard -----------------------
DASHBOARD_DIR="${REPO_ROOT}/dashboard"
DASHBOARD_STACK_FILE="${REPO_ROOT}/stack/dashboard.yml"
DASHBOARD_IMAGE="swg-dashboard:local"
DASHBOARD_STACK_NAME="${DASHBOARD_STACK_NAME:-swg-dashboard}"

if [[ -d "$DASHBOARD_DIR" ]]; then
  log "Building dashboard image (${CYAN}$DASHBOARD_IMAGE${RESET})..."
  docker build -t "$DASHBOARD_IMAGE" "$DASHBOARD_DIR"
else
  warn "Dashboard directory missing: $DASHBOARD_DIR"
fi

if [[ -f "$DASHBOARD_STACK_FILE" ]]; then
  log "Deploying dashboard stack '${CYAN}${DASHBOARD_STACK_NAME}${RESET}'..."
  docker stack deploy -c "$DASHBOARD_STACK_FILE" "$DASHBOARD_STACK_NAME"
else
  warn "Dashboard stack file missing: $DASHBOARD_STACK_FILE"
fi

# ---------------------------- Show Join Tokens -----------------------
log "Manager join token:"
docker swarm join-token manager

log "Worker join token:"
docker swarm join-token worker

log "Initialization complete."
