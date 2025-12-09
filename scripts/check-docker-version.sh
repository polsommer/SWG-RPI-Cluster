#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# Default configuration
# ---------------------------------------------------------
TARGET_DOCKER_VERSION="${TARGET_DOCKER_VERSION:-"24.0.9"}"
COMPARE_MODE="${COMPARE_MODE:-"exact"}"   # exact | min | max
WARN_ONLY=false

# ---------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------
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

fail() {
  log "${RED}ERROR:${RESET} $*"
  exit 1
}

warn() {
  log "${YELLOW}WARN:${RESET} $*"
}

info() {
  log "${GREEN}INFO:${RESET} $*"
}

# ---------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --warn-only)
      WARN_ONLY=true
      shift
      ;;
    --min)
      COMPARE_MODE="min"
      shift
      ;;
    --max)
      COMPARE_MODE="max"
      shift
      ;;
    --exact)
      COMPARE_MODE="exact"
      shift
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

# ---------------------------------------------------------
# Check for Docker installation
# ---------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  warn "Docker is not installed; cannot verify version."
  exit $([[ "$WARN_ONLY" == true ]] && echo 0 || echo 1)
fi

# ---------------------------------------------------------
# Detect installed Docker version
# ---------------------------------------------------------
CURRENT_VERSION_RAW=$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)
CURRENT_VERSION=$(echo "$CURRENT_VERSION_RAW" | sed 's/-.*//' | tr -d ' ')

if [[ -z "$CURRENT_VERSION" ]]; then
  warn "Failed to detect Docker Engine version."
  exit $([[ "$WARN_ONLY" == true ]] && echo 0 || echo 2)
fi

info "Detected Docker Engine version: ${CURRENT_VERSION}"
info "Target version: ${TARGET_DOCKER_VERSION} (mode: ${COMPARE_MODE})"

# ---------------------------------------------------------
# Version comparison function
# ---------------------------------------------------------
# Usage: semver_compare A B
# returns: 0 if A==B, 1 if A>B, 2 if A<B
semver_compare() {
  local a b
  a=(${1//./ })
  b=(${2//./ })
  for i in 0 1 2; do
    local x=${a[i]:-0}
    local y=${b[i]:-0}
    if (( x > y )); then return 1; fi
    if (( x < y )); then return 2; fi
  done
  return 0
}

# ---------------------------------------------------------
# Perform version comparison
# ---------------------------------------------------------
semver_compare "$CURRENT_VERSION" "$TARGET_DOCKER_VERSION"
cmp=$?

is_ok=false

case "$COMPARE_MODE" in
  exact)
    [[ $cmp -eq 0 ]] && is_ok=true
    ;;
  min)
    # current >= target → OK
    [[ $cmp -eq 0 || $cmp -eq 1 ]] && is_ok=true
    ;;
  max)
    # current <= target → OK
    [[ $cmp -eq 0 || $cmp -eq 2 ]] && is_ok=true
    ;;
  *)
    fail "Unknown COMPARE_MODE: $COMPARE_MODE"
    ;;
esac

# ---------------------------------------------------------
# Evaluate result
# ---------------------------------------------------------
if [[ "$is_ok" == true ]]; then
  info "Docker version requirement satisfied."
  exit 0
fi

# Mismatch detected
warn "Docker version mismatch: current=$CURRENT_VERSION target=$TARGET_DOCKER_VERSION"

if [[ "$WARN_ONLY" == true ]]; then
  warn "WARN_ONLY enabled → continuing anyway."
  exit 0
fi

# Non-warn mode = fail with exit code 3
fail "Docker Engine version does not meet the required condition (mode: $COMPARE_MODE)."
