#!/usr/bin/env bash
set -euo pipefail

# bootstrap-node.sh
# Helper to prepare a node as a swarm manager or worker in one command.
# Usage:
#   sudo bash scripts/bootstrap-node.sh manager <advertise_ip>
#   sudo bash scripts/bootstrap-node.sh worker <manager_ip> [worker_join_token]

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

log() { echo "${GREEN}INFO:${RESET} $*"; }
warn() { echo "${YELLOW}WARN:${RESET} $*" >&2; }
fatal() { echo "${RED}ERROR:${RESET} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || fatal "Please run as root (sudo)."

if [[ $# -lt 2 ]]; then
  fatal "Usage: $0 <manager|worker> <manager_ip> [worker_join_token]"
fi

ROLE="$1"
MANAGER_IP="$2"
WORKER_TOKEN="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_SCRIPT="${SCRIPT_DIR}/install-docker.sh"
INIT_SCRIPT="${SCRIPT_DIR}/init-swarm.sh"
JOIN_SCRIPT="${SCRIPT_DIR}/join-swarm.sh"
DEPLOY_SCRIPT="${SCRIPT_DIR}/deploy-demo.sh"

[[ -x "$INSTALL_SCRIPT" ]] || fatal "Missing dependency: $INSTALL_SCRIPT"

case "$ROLE" in
  manager)
    log "Bootstrapping manager at advertise IP ${CYAN}${MANAGER_IP}${RESET}"

    before_flags_set=0
    if ! grep -q "cgroup_memory" /boot/cmdline.txt; then
      before_flags_set=1
    fi

    bash "$INSTALL_SCRIPT"

    after_flags_set=0
    if grep -q "cgroup_memory" /boot/cmdline.txt; then
      after_flags_set=1
    fi

    if [[ "$before_flags_set" -eq 1 && "$after_flags_set" -eq 1 ]]; then
      warn "Memory cgroup flags were just added. A reboot is recommended."
      read -r -p "Reboot now? [y/N] " reboot_now || true
      if [[ "$reboot_now" =~ ^[Yy]$ ]]; then
        log "Rebooting to apply kernel flags..."
        reboot
      else
        warn "Continuing without reboot; rerun if Docker fails to start." 
      fi
    fi

    [[ -x "$INIT_SCRIPT" ]] || fatal "Missing dependency: $INIT_SCRIPT"
    bash "$INIT_SCRIPT" "$MANAGER_IP"

    [[ -x "$DEPLOY_SCRIPT" ]] || fatal "Missing dependency: $DEPLOY_SCRIPT"
    bash "$DEPLOY_SCRIPT"
    ;;
  worker)
    log "Bootstrapping worker with manager at ${CYAN}${MANAGER_IP}${RESET}"

    bash "$INSTALL_SCRIPT"

    if command -v docker >/dev/null 2>&1; then
      SWARM_STATE=$(docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null || echo "unknown")
      if [[ "$SWARM_STATE" == "active" ]]; then
        warn "This node already belongs to a swarm."
        read -r -p "Leave existing swarm and join ${MANAGER_IP}? [y/N] " leave_choice || true
        if [[ "$leave_choice" =~ ^[Yy]$ ]]; then
          docker swarm leave --force
        else
          fatal "Aborting; node remains in its current swarm."
        fi
      fi
    fi

    [[ -n "$WORKER_TOKEN" ]] || read -r -p "Enter worker join token: " WORKER_TOKEN
    [[ -n "$WORKER_TOKEN" ]] || fatal "Worker join token is required."

    [[ -x "$JOIN_SCRIPT" ]] || fatal "Missing dependency: $JOIN_SCRIPT"
    bash "$JOIN_SCRIPT" "$MANAGER_IP" "$WORKER_TOKEN"
    ;;
  *)
    fatal "Role must be 'manager' or 'worker'."
    ;;
esac

log "Bootstrap sequence complete."
