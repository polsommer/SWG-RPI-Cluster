#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# join-swarm.sh — Join a worker node to a Docker Swarm cluster
# Usage:
#   sudo ./join-swarm.sh <manager_ip> <worker_token>
# ------------------------------------------------------------

# ---------------------------- COLORS -------------------------
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

log()       { echo "$(timestamp) ${GREEN}INFO:${RESET} $*"; }
warn()      { echo "$(timestamp) ${YELLOW}WARN:${RESET} $*" >&2; }
error()     { echo "$(timestamp) ${RED}ERROR:${RESET} $*" >&2; }
fatal()     { error "$*"; exit 1; }

# ---------------------------- ROOT CHECK ---------------------
[[ $EUID -eq 0 ]] || fatal "Please run as root (sudo)."

# ---------------------------- Arguments ----------------------
if [[ $# -lt 2 ]]; then
  fatal "Usage: $0 <manager_ip> <worker_token>"
fi

MANAGER_IP="$1"
TOKEN="$2"

log "Manager IP: ${CYAN}$MANAGER_IP${RESET}"
log "Worker token length: ${#TOKEN}"

# ---------------------------- Check Swarm State --------------
LOCAL_SWARM_STATE=$(docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null || echo "unknown")

if [[ "$LOCAL_SWARM_STATE" == "active" ]]; then
  fatal "Node is already part of a swarm. Run:
    docker swarm leave --force"
fi

# ---------------------------- Docker Check -------------------
command -v docker >/dev/null 2>&1 || fatal "Docker not installed."

if ! docker info >/dev/null 2>&1; then
  fatal "Docker is installed but not running."
fi

# ---------------------------- Network Checks -----------------
log "Checking connectivity to manager node..."

if ! ping -c1 -W1 "$MANAGER_IP" >/dev/null 2>&1; then
  fatal "Cannot ping manager at $MANAGER_IP. Check LAN/DNS."
fi

log "Testing TCP connection to ${MANAGER_IP}:2377..."
if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/$MANAGER_IP/2377"; then
  fatal "TCP port 2377 is unreachable. Swarm join will fail.
Check firewall or Docker daemon on manager."
fi

# ---------------------------- Token Sanity Check -------------
if [[ ${#TOKEN} -lt 20 ]]; then
  warn "Token looks very short. Did you paste it incorrectly?"
fi

if [[ "$TOKEN" != SWMTKN-* ]]; then
  warn "Worker token does not begin with 'SWMTKN-'.
Double check with:
  docker swarm join-token worker"
fi

# ---------------------------- Time Sync Check ----------------
# Large clock drift breaks raft consensus
if command -v timedatectl >/dev/null 2>&1; then
  if timedatectl show --property=NTPSynchronized | grep -q "no"; then
    warn "Clock is NOT synchronized (NTP disabled). Swarm can behave incorrectly."
  fi
fi

# ---------------------------- Perform Join -------------------
log "Joining swarm at ${CYAN}${MANAGER_IP}:2377${RESET}..."

if docker swarm join --token "$TOKEN" "$MANAGER_IP:2377"; then
  log "Swarm join successful!"
else
  fatal "Swarm join failed. Verify token and manager IP."
fi

# ---------------------------- Final Info ---------------------
log "Node status:"
docker info | grep -E "Swarm:|NodeID"

log "Done."
