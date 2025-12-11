#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------
# Trap for clean shutdown
# -----------------------------------------------
cleanup() {
  log "INFO" "Shutting down gracefully..."
}
trap cleanup EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# -----------------------------------------------
# Load .env (optional)
# -----------------------------------------------
ENV_FILE=${ENV_FILE:-"$SCRIPT_DIR/.env"}

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# -----------------------------------------------
# Defaults
# -----------------------------------------------
WATCH_DIR=${WATCH_DIR:-"/srv/textures/output"}
EXPORT_DIR=${EXPORT_DIR:-"/srv/textures/export"}
TEMP_DIR="${EXPORT_DIR}.staging"

FILE_PATTERNS=${FILE_PATTERNS:-"*.dds *.png"}
GIT_REMOTE_URL=${GIT_REMOTE_URL:-""}
GIT_BRANCH=${GIT_BRANCH:-"main"}

GIT_SSH_KEY_PATH=${GIT_SSH_KEY_PATH:-""}
GIT_USER_NAME=${GIT_USER_NAME:-"Texture Sync Bot"}
GIT_USER_EMAIL=${GIT_USER_EMAIL:-"textures@example.com"}

SYNC_INTERVAL=${SYNC_INTERVAL:-30}
LOG_FILE=${LOG_FILE:-""}
PUSH_RETRIES=${PUSH_RETRIES:-3}
DRY_RUN=${DRY_RUN:-0}
FALLBACK_ARCHIVE_DIR=${FALLBACK_ARCHIVE_DIR:-"${EXPORT_DIR}/manual_review"}

# -----------------------------------------------
# HDtextureDDS integration
# -----------------------------------------------
HD_DDS_ENABLED=${HD_DDS_ENABLED:-1}  
HD_DDS_REPO="https://github.com/polsommer/HDtextureDDS.git"
HD_DDS_DIR="$SCRIPT_DIR/HDtextureDDS"

ensure_hdtexturedds_repo() {
  if [[ "$HD_DDS_ENABLED" != "1" ]]; then
    log "INFO" "HDtextureDDS integration disabled."
    return
  fi

  if [[ ! -d "$HD_DDS_DIR/.git" ]]; then
    log "INFO" "Cloning HDtextureDDS..."
    git clone "$HD_DDS_REPO" "$HD_DDS_DIR"
  else
    log "INFO" "Updating HDtextureDDS..."
    (cd "$HD_DDS_DIR" && git pull --rebase || true)
  fi

  log "INFO" "HDtextureDDS ready at $HD_DDS_DIR"
}

# Convert files using HDtextureDDS if needed
process_with_hdtexturedds() {
  if [[ "$HD_DDS_ENABLED" != "1" ]]; then return; fi

  log "INFO" "Running HDtextureDDS preprocess step..."

  DDS_SCRIPT="$HD_DDS_DIR/tools/convert_to_dds.sh"

  if [[ ! -x "$DDS_SCRIPT" ]]; then
    log "WARN" "HDtextureDDS conversion script missing or not executable: $DDS_SCRIPT"
    return
  fi

  # Convert any non-DDS textures in WATCH_DIR
  find "$WATCH_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | \
  while read -r file; do
      log "INFO" "Converting to DDS: $file"
      "$DDS_SCRIPT" "$file" || log "WARN" "Conversion failed for: $file"
  done

  log "INFO" "HDtextureDDS processing complete."
}

# -----------------------------------------------
# Logging
# -----------------------------------------------
log() {
  local level="$1"; shift
  local msg="$*"
  local ts
  ts=$(date -Iseconds)
  if [[ -n "$LOG_FILE" ]]; then
    echo "$ts [$level] $msg" | tee -a "$LOG_FILE"
  else
    echo "$ts [$level] $msg"
  fi
}

fail() {
  log "ERROR" "$1"
  exit 1
}

# -----------------------------------------------
# Dependency checks
# -----------------------------------------------
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "$1 is required but not installed. $2"
  fi
}

require_cmd git  "Install git."
require_cmd rsync "Install rsync."

# -----------------------------------------------
# Setup Git repository
# -----------------------------------------------
prepare_git_repo() {
  mkdir -p "$EXPORT_DIR"
  cd "$EXPORT_DIR"

  if [[ ! -d .git ]]; then
    if [[ -n "$GIT_REMOTE_URL" ]]; then
      log "INFO" "Cloning $GIT_REMOTE_URL into $EXPORT_DIR"
      if ! git clone --branch "$GIT_BRANCH" "$GIT_REMOTE_URL" . >/dev/null 2>&1; then
        log "WARN" "Clone failed; initializing empty repo"
        git init
      fi
    else
      log "INFO" "Initializing empty repository at $EXPORT_DIR"
      git init
    fi
  fi

  git config user.name "$GIT_USER_NAME"
  git config user.email "$GIT_USER_EMAIL"

  if [[ -n "$GIT_REMOTE_URL" ]]; then
    if git remote get-url origin &>/dev/null; then
      git remote set-url origin "$GIT_REMOTE_URL"
    else
      git remote add origin "$GIT_REMOTE_URL"
    fi
  fi

  if ! git rev-parse --verify "$GIT_BRANCH" >/dev/null 2>&1; then
    git checkout -b "$GIT_BRANCH"
  else
    git checkout "$GIT_BRANCH"
  fi

  git fetch --all --prune || true
}

# -----------------------------------------------
# Atomic rsync
# -----------------------------------------------
sync_files() {
  mkdir -p "$TEMP_DIR"

  local includes=()
  for pattern in $FILE_PATTERNS; do
    includes+=("--include=$pattern")
  done

  log "INFO" "Syncing textures from $WATCH_DIR → $TEMP_DIR (atomic stage)"

  rsync -av \
    --delete \
    --prune-empty-dirs \
    --exclude='/.git/' \
    --include='*/' \
    "${includes[@]}" \
    --exclude='*' \
    "$WATCH_DIR/" "$TEMP_DIR/" \
    | while read -r line; do log "INFO" "rsync: $line"; done

  rsync -av --delete "$TEMP_DIR/" "$EXPORT_DIR/" >/dev/null
}

# -----------------------------------------------
# Commit + Push
# -----------------------------------------------
commit_and_push() {
  cd "$EXPORT_DIR"

  if git status --porcelain | grep -q .; then
    git add -A
    git commit -m "Sync textures $(date -Iseconds)" || true
  else
    log "INFO" "No changes to commit."
    return
  fi

  if [[ -z "$GIT_REMOTE_URL" ]]; then
    log "WARN" "GIT_REMOTE_URL not set; skipping push."
    archive_for_manual_review "GIT_REMOTE_URL is unset"
    return
  fi

  if [[ -n "$GIT_SSH_KEY_PATH" ]]; then
    export GIT_SSH_COMMAND="ssh -i $GIT_SSH_KEY_PATH -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
  fi

  local attempt=1
  until git push -u origin "$GIT_BRANCH"; do
    log "WARN" "Push failed (attempt $attempt). Retrying..."
    ((attempt++))
    if (( attempt > PUSH_RETRIES )); then
      archive_for_manual_review "Git push failed after $PUSH_RETRIES attempts"
      fail "Push failed after $PUSH_RETRIES attempts."
    fi
    sleep 5
  done

  log "INFO" "Push successful."
}

archive_for_manual_review() {
  local reason="$1"
  local ts
  ts=$(date -Iseconds | tr ':' '-')
  local target_dir="${FALLBACK_ARCHIVE_DIR}/${ts}"

  mkdir -p "$target_dir"

  log "WARN" "GitHub upload unavailable (${reason}). Archiving staged files to: $target_dir"

  rsync -av --exclude='.git/' "$EXPORT_DIR/" "$target_dir/" \
    | while read -r line; do log "INFO" "archive: $line"; done

  log "INFO" "Manual review archive ready at $target_dir. Transfer this directory to the intended repository when connectivity is restored."
}

# -----------------------------------------------
# Full sync cycle (now includes HDtextureDDS)
# -----------------------------------------------
run_sync_cycle() {
  log "INFO" "Running sync cycle..."

  ensure_hdtexturedds_repo
  process_with_hdtexturedds

  prepare_git_repo
  sync_files

  [[ "$DRY_RUN" == "1" ]] && { log "INFO" "DRY RUN MODE: skipping commit/push"; return; }
  commit_and_push
}

# -----------------------------------------------
# Watcher or Polling
# -----------------------------------------------
watch_with_inotify() {
  log "INFO" "Using inotify for change detection."
  require_cmd inotifywait "Install inotify-tools."

  inotifywait -m -r \
    -e close_write,move,create,delete \
    --format '%w%f %e' \
    "$WATCH_DIR" | while read -r file event; do
      case "$file" in
        *.tmp|*.swp) continue ;;
      esac
      log "INFO" "Detected change: $file ($event)"
      run_sync_cycle
    done
}

polling_loop() {
  log "INFO" "Polling every ${SYNC_INTERVAL}s..."
  while true; do
    run_sync_cycle
    sleep "$SYNC_INTERVAL"
  done
}

# -----------------------------------------------
# MAIN
# -----------------------------------------------
main() {
  [[ ! -d "$WATCH_DIR" ]] && {
    log "WARN" "WATCH_DIR missing — creating: $WATCH_DIR"
    mkdir -p "$WATCH_DIR"
  }

  run_sync_cycle

  if command -v inotifywait >/dev/null 2>&1; then
    watch_with_inotify
  else
    log "WARN" "inotifywait missing — using polling mode."
    polling_loop
  fi
}

main "$@"
