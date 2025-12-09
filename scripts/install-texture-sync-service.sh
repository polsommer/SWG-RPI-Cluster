#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)."
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SYSTEMD_DEST=/etc/systemd/system

install -m 0644 "$REPO_ROOT/systemd/texture-sync.service" "$SYSTEMD_DEST/texture-sync.service"
install -m 0644 "$REPO_ROOT/systemd/texture-sync.timer" "$SYSTEMD_DEST/texture-sync.timer"

systemctl daemon-reload
systemctl enable --now texture-sync.timer

echo "texture-sync.timer enabled and running."
