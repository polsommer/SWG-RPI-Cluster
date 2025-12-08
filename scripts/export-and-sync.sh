#!/usr/bin/env bash
set -euo pipefail

# Optional: load environment variables from a .env file.
ENV_FILE=${ENV_FILE:-".env"}
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

WATCH_DIR=${WATCH_DIR:-"/srv/textures/output"}
EXPORT_DIR=${EXPORT_DIR:-"/srv/textures/export"}
FILE_PATTERNS=${FILE_PATTERNS:-"*.dds *.png"}
GIT_REMOTE_URL=${GIT_REMOTE_URL:-""}
GIT_BRANCH=${GIT_BRANCH:-"main"}
GIT_SSH_KEY_PATH=${GIT_SSH_KEY_PATH:-""}
GIT_USER_NAME=${GIT_USER_NAME:-"Texture Sync Bot"}
GIT_USER_EMAIL=${GIT_USER_EMAIL:-"textures@example.com"}
SYNC_INTERVAL=${SYNC_INTERVAL:-30}
LOG_FILE=${LOG_FILE:-""}

log() {
  local level="$1"; shift
  local message="$*"
  local timestamp
  timestamp=$(date -Iseconds)
  if [[ -n "$LOG_FILE" ]]; then
    echo "$timestamp [$level] $message" | tee -a "$LOG_FILE"
  else
    echo "$timestamp [$level] $message"
  fi
}

fail() {
  log "ERROR" "$1"
  exit 1
}

prepare_git_repo() {
  mkdir -p "$EXPORT_DIR"
  cd "$EXPORT_DIR"

  if [[ ! -d .git ]]; then
    if [[ -n "$GIT_REMOTE_URL" ]]; then
      log "INFO" "Cloning $GIT_REMOTE_URL into $EXPORT_DIR"
      if git clone --branch "$GIT_BRANCH" "$GIT_REMOTE_URL" . >/dev/null 2>&1; then
        cd "$EXPORT_DIR"
      else
        log "WARN" "Clone failed; falling back to initializing a fresh repo"
        git init
      fi
    else
      log "INFO" "Initializing git repository in $EXPORT_DIR"
      git init
    fi
  fi

  git config user.name "$GIT_USER_NAME"
  git config user.email "$GIT_USER_EMAIL"

  if [[ -n "$GIT_REMOTE_URL" ]]; then
    if git remote get-url origin >/dev/null 2>&1; then
      git remote set-url origin "$GIT_REMOTE_URL"
    else
      git remote add origin "$GIT_REMOTE_URL"
    fi
  fi

  # If the branch does not exist locally, create it.
  if ! git show-ref --verify --quiet "refs/heads/$GIT_BRANCH"; then
    git checkout -b "$GIT_BRANCH" >/dev/null 2>&1 || git checkout -b "$GIT_BRANCH"
  else
    git checkout "$GIT_BRANCH"
  fi
}

sync_files() {
  mkdir -p "$EXPORT_DIR"

  local includes=()
  for pattern in $FILE_PATTERNS; do
    includes+=("--include=$pattern")
  done

  rsync -av --delete --prune-empty-dirs --include='*/' "${includes[@]}" --exclude='*' "$WATCH_DIR/" "$EXPORT_DIR/" \
    | while IFS= read -r line; do log "INFO" "rsync: $line"; done
}

commit_and_push() {
  cd "$EXPORT_DIR"
  if git status --porcelain | grep -q .; then
    git add -A
    git commit -m "Sync textures $(date -Iseconds)" || log "INFO" "No changes to commit"
  else
    log "INFO" "No changes detected; skipping commit"
    return
  fi

  if [[ -n "$GIT_REMOTE_URL" ]]; then
    if [[ -n "$GIT_SSH_KEY_PATH" ]]; then
      export GIT_SSH_COMMAND="ssh -i $GIT_SSH_KEY_PATH -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    fi
    log "INFO" "Pushing to $GIT_REMOTE_URL ($GIT_BRANCH)"
    git push -u origin "$GIT_BRANCH"
  else
    log "WARN" "GIT_REMOTE_URL not set; skipping push"
  fi
}

run_sync_cycle() {
  prepare_git_repo
  sync_files
  commit_and_push
}

watch_with_inotify() {
  log "INFO" "Watching $WATCH_DIR for changes (inotify mode)"
  inotifywait -m -r -e close_write,move,create,delete "$WATCH_DIR" --format '%w%f %e' | while read -r file event; do
    log "INFO" "Change detected: $file ($event)"
    run_sync_cycle
  done
}

polling_loop() {
  log "INFO" "Watching $WATCH_DIR for changes (polling every ${SYNC_INTERVAL}s)"
  while true; do
    run_sync_cycle
    sleep "$SYNC_INTERVAL"
  done
}

main() {
  if [[ ! -d "$WATCH_DIR" ]]; then
    log "WARN" "WATCH_DIR $WATCH_DIR does not exist; creating it"
    mkdir -p "$WATCH_DIR"
  fi

  # Run an initial sync immediately.
  run_sync_cycle

  if command -v inotifywait >/dev/null 2>&1; then
    watch_with_inotify
  else
    polling_loop
  fi
}

main "$@"
