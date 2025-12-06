#!/usr/bin/env bash
set -euo pipefail

# install-docker.sh
# Prepares a Raspberry Pi node with Docker Engine and common prerequisites
# so it can participate in the SWG three-node cluster.

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" != arm* && "$ARCH" != aarch64 ]]; then
  echo "This script targets Raspberry Pi (ARM). Detected architecture: $ARCH" >&2
fi

if ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y curl
fi

apt-get update
apt-get install -y ca-certificates software-properties-common jq

# Enable cgroup support for container runtimes (Pi OS usually needs this).
if ! grep -q "cgroup_memory" /boot/cmdline.txt; then
  sed -i '1 s/$/ cgroup_memory=1 cgroup_enable=memory/' /boot/cmdline.txt
  echo "Added memory cgroup flags to /boot/cmdline.txt. Reboot is recommended." >&2
fi

# Install Docker using the official convenience script.
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sh /tmp/get-docker.sh
  rm /tmp/get-docker.sh
  if [[ -n "${SUDO_USER:-}" ]]; then
    usermod -aG docker "$SUDO_USER"
  fi
else
  echo "Docker already installed; skipping installation." >&2
fi

systemctl enable docker
systemctl start docker

echo "Docker installed. Log out/in so your user picks up the docker group." >&2
