#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# install-texture-sync.sh
# Installs and activates the systemd service + timer
# for automated texture export + Git sync.
# -------------------------------------------------------

# ---------------------- Logging -------------------------
timestamp() { date -Iseconds; }

supports_color() {
  [[ -t 1 ]] && command -v tput >/dev/null && [[ $(tput colors) -ge 8 ]]
}

if supports_color; then
  RED=$(tput setaf 1)
  GREEN=$(tput setaf 2)
  YELLOW=$(tput setaf 3)
  RESET=$(tput sgr0)
else
  RED=""; GREEN=""; YELLOW=""; RESET=""
fi

log() {
  echo "$(timestamp) $*"
}

info() { log "${GREEN}INFO:${RESET} $*"; }
warn() { log "${YELLOW}WARN:${RESET} $*"; }
fail() { log "${RED}ERROR:${RESET} $*"; exit 1; }

# ---------------------- Root check -----------------------
if [[ $EUID -ne 0 ]]; then
  fail "Please run as root (sudo)."
fi

# ---------------------- Paths ----------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SYSTEMD_DEST=/etc/systemd/system

SERVICE_SRC="$REPO_ROOT/systemd/texture-sync.service"
TIMER_SRC="$REPO_ROOT/systemd/texture-sync.timer"

SERVICE_DEST="$SYSTEMD_DEST/texture-sync.service"
TIMER_DEST="$SYSTEMD_DEST/texture-sync.timer"

# ---------------------- CLI args -------------------------
UNINSTALL=false
RELOAD_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uninstall)
      UNINSTALL=true
      shift
      ;;
    --reload-only)
      RELOAD_ONLY=true
      shift
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

# ---------------------- Validate environment -------------
command -v systemctl >/dev/null 2>&1 || fail "systemd/systemctl is required."

if [[ ! -d "$SYSTEMD_DEST" ]]; then
  fail "Systemd directory missing: $SYSTEMD_DEST"
fi

# ---------------------- Uninstall mode -------------------
if [[ "$UNINSTALL" == true ]]; then
  info "Stopping texture-sync service + timer..."

  systemctl stop texture-sync.service 2>/dev/null || true
  systemctl stop texture-sync.timer 2>/dev/null || true
  systemctl disable texture-sync.timer 2>/dev/null || true

  info "Removing systemd files..."
  rm -f "$SERVICE_DEST" "$TIMER_DEST"

  systemctl daemon-reload
  info "texture-sync systemd units removed."

  exit 0
fi

# ---------------------- Validate source files ------------
[[ -f "$SERVICE_SRC" ]] || fail "Missing service file: $SERVICE_SRC"
[[ -f "$TIMER_SRC"   ]] || fail "Missing timer file: $TIMER_SRC"

# ---------------------- Reload-only mode -----------------
if [[ "$RELOAD_ONLY" == true ]]; then
  info "Reloading systemd units (no reinstall)..."
  systemctl daemon-reload
  systemctl restart texture-sync.timer || true
  info "Systemd units reloaded."
  exit 0
fi

# ---------------------- Install units --------------------
info "Installing texture-sync.service → $SERVICE_DEST"
install -m 0644 "$SERVICE_SRC" "$SERVICE_DEST"

info "Installing texture-sync.timer   → $TIMER_DEST"
install -m 0644 "$TIMER_SRC" "$TIMER_DEST"

# ---------------------- Enable timer ---------------------
info "Reloading systemd daemon..."
systemctl daemon-reload

info "Enabling + starting texture-sync.timer..."
systemctl enable --now texture-sync.timer

info "Installation complete: texture-sync.timer enabled and running."
