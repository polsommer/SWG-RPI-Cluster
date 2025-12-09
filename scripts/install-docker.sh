#!/usr/bin/env bash
set -euo pipefail

# install-docker.sh
# Prepares a Raspberry Pi node with Docker Engine and common prerequisites
# so it can participate in the SWG three-node cluster.

TARGET_DOCKER_VERSION=${TARGET_DOCKER_VERSION:-"24.0.9"}
AUTO_ALIGN_DOCKER=${AUTO_ALIGN_DOCKER:-}

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
apt-get install -y ca-certificates software-properties-common jq gnupg

# Enable cgroup support for container runtimes (Pi OS usually needs this).
if ! grep -q "cgroup_memory" /boot/cmdline.txt; then
  sed -i '1 s/$/ cgroup_memory=1 cgroup_enable=memory/' /boot/cmdline.txt
  echo "Added memory cgroup flags to /boot/cmdline.txt. Reboot is recommended." >&2
fi

ensure_docker_repo() {
  . /etc/os-release
  ARCH_DEB=$(dpkg --print-architecture)
  mkdir -p /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
    curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
  fi
  echo "deb [arch=${ARCH_DEB} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
}

find_target_package_version() {
  apt-cache madison docker-ce | awk -v target="$TARGET_DOCKER_VERSION" '$2 ~ target {print $2; exit}'
}

install_target_version() {
  ensure_docker_repo
  apt-get update
  PACKAGE_VERSION=$(find_target_package_version)
  if [[ -z "$PACKAGE_VERSION" ]]; then
    echo "Unable to find docker-ce package matching $TARGET_DOCKER_VERSION. Check TARGET_DOCKER_VERSION." >&2
    exit 3
  fi
  apt-get install -y \
    docker-ce="$PACKAGE_VERSION" \
    docker-ce-cli="$PACKAGE_VERSION" \
    containerd.io docker-buildx-plugin docker-compose-plugin
}

current_version=""
if command -v docker >/dev/null 2>&1; then
  current_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)
fi

if [[ -z "$current_version" ]]; then
  echo "Docker not installed or version unavailable; installing target version ${TARGET_DOCKER_VERSION}."
  install_target_version
else
  if [[ "$current_version" == "$TARGET_DOCKER_VERSION" ]]; then
    echo "Docker already installed at target version ($current_version); skipping reinstall."
  else
    echo "Detected Docker version $current_version but target is $TARGET_DOCKER_VERSION." >&2
    align_response="${AUTO_ALIGN_DOCKER}"
    if [[ -z "$align_response" ]]; then
      read -r -p "Attempt to align Docker to ${TARGET_DOCKER_VERSION}? [y/N] " align_response || true
    fi
    if [[ "$align_response" =~ ^[Yy]$ ]]; then
      install_target_version
    else
      echo "Docker version mismatch left unresolved. Rerun with AUTO_ALIGN_DOCKER=1 to auto-align." >&2
    fi
  fi
fi

if [[ -n "${SUDO_USER:-}" ]]; then
  usermod -aG docker "$SUDO_USER"
fi

systemctl enable docker
systemctl start docker

echo "Docker installed. Log out/in so your user picks up the docker group." >&2
