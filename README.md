# SWG-RPI-Cluster

Build and test a **three-node Raspberry Pi Docker Swarm** with a single set of
scripts. The repo installs Docker, bootstraps the swarm, and deploys a demo
stack that proves overlay-network traffic between nodes. Guardrails such as
idempotent installs, token validation, and preflight network checks keep the
process repeatable on fresh or recycled Pis.

> Validated on Raspberry Pi OS 64-bit (Bookworm). Docker and the demo images are
> multi-arch, so the flow works on Pi 3/4/5.

---

## What you need
- Three Raspberry Pi boards with a 64-bit OS, SSH enabled, and static/reserved
  IP addresses (example: manager `192.168.88.7`, workers `192.168.88.5` and
  `192.168.88.8`)
- A shared LAN plus consistent hostnames (examples: `pi-manager`,
  `pi-worker1`, `pi-worker2`)
- `sudo` access on every node
- (Optional) DNS or mDNS so you can visit `http://pi-manager:8080`

## Quickstart
Pick the **one-liner** flow for each node or follow the step-by-step commands
below.

### One-liner bootstrap (per node)
- Manager: `sudo bash scripts/bootstrap-node.sh manager 192.168.88.7`
- Worker: `sudo bash scripts/bootstrap-node.sh worker 192.168.88.7 SWMTKN-1-...`

The helper chains the existing scripts: managers install Docker, initialize the
swarm, and deploy the demo stack; workers install Docker and join using the
provided token (prompted if omitted). If the manager adds cgroup flags to
`/boot/cmdline.txt`, it offers a reboot before continuing.

### Step-by-step flow
Each step is a single command. Follow them in order and run the commands exactly
as shown on the indicated nodes.

### Prep (all nodes)
- Clone this repo onto **every** Pi (or copy `scripts/` and `stack/`).
- Ensure `sudo` works without a password prompt if you plan to run unattended.

### 1) Install Docker (all nodes)
Run on every Pi (manager + workers):

```bash
sudo bash scripts/install-docker.sh
```

Reboot once so the added cgroup flags in `/boot/cmdline.txt` take effect. The
script is safe to rerun whenever you add or rebuild a node.

### 2) Initialize the swarm (manager)
Run **only on `pi-manager`**. Swap in your advertise IP if different (example
below uses `192.168.88.7`):

```bash
sudo bash scripts/init-swarm.sh 192.168.88.7
```

The command prints fresh **manager** and **worker** join tokens plus the
advertise address. It also builds and deploys the web control module so the
dashboard is ready at `http://<manager-ip>:8081` immediately. If the node
already belongs to a swarm, the script exits with the current membership details
instead of reinitializing.

### 3) Join the workers
Run on `pi-worker1` and `pi-worker2` using the worker token from step 2:

```bash
sudo bash scripts/join-swarm.sh 192.168.88.7 SWMTKN-1-...
```

If a worker is already in a swarm, leave first with `docker swarm leave --force`
and rerun the command. Confirm membership from the manager:

```bash
docker node ls
```

### 4) Deploy the demo stack (manager)
Run on the manager to create the overlay network and demo services:

```bash
sudo bash scripts/deploy-demo.sh
```

The compose file (`stack/demo.yml`) launches:
- **hello-app** – `traefik/whoami` with 2 replicas (one per worker) on port 8080.
- **redis-bus** – `redis:7-alpine` pinned to the manager with a named volume for
  durability.
- **echo-worker** – two curl-based workers that continually hit `hello-app`
  across the overlay network, logging responder hostname and latency from the
  worker nodes to avoid loading the manager.
- **swarm-dashboard** – Flask UI on port 8081 (manager-only) that streams swarm
  health, replica counts, and Redis reachability from the Docker API. The new
  Control Center panel surfaces join commands, one-click token rotation, service
  scaling, and node drain/activation switches so you can grow the cluster
  without reaching for the CLI.

App workloads intentionally land on workers so the manager stays available for
control-plane tasks. When you deploy your own services, add a placement
constraint such as `node.role == worker` to keep them off the manager, or adjust
the replica count if you add more nodes to the cluster.

### 5) Verify cross-node traffic (manager or any node)
- Hit the published service: `curl http://192.168.88.7:8080` (or
  `http://pi-manager.local:8080`). The `Hostname` should vary between nodes.
- Open the dashboard at `http://192.168.88.7:8081` for live metrics and overlay
  checks.
- Watch the worker logs from inside the cluster:

```bash
docker service logs -f swg-demo_echo-worker
```

- Confirm service placement:

```bash
docker service ps swg-demo_hello-app
```

- Confirm Redis health within the overlay:

```bash
docker run --rm --network cluster_net redis:7-alpine redis-cli -h redis-bus PING
```

### 6) Tear down (optional)

```bash
docker stack rm swg-demo
```

The overlay network `cluster_net` is created once and reused; remove it with
`docker network rm cluster_net` if you want a full cleanup.

## Deploying the texture processing stack (manager)
Use `stack/textures.yml` to ship a Redis-backed texture pipeline that shares a
single volume across ingestion, upscaling, and LLM metadata analysis workers.
Placement constraints keep replicas on workers, with optional GPU labels for the
Real-ESRGAN nodes.

### Provision shared texture storage (manager + workers)
The stack expects a shared export at `/srv/textures` mounted on every swarm
node. The example below uses an NFS export from the manager; adjust the server
address if you use a NAS or other host.

1. On the chosen NFS server (for example the swarm manager), create the export
   and ensure it is writable by the service users:

   ```bash
   sudo mkdir -p /srv/textures
   sudo chown -R 1000:1000 /srv/textures
   # Add to /etc/exports (example):
   # /srv/textures 192.168.88.0/24(rw,sync,no_subtree_check)
   sudo exportfs -rav
   ```

2. Create the swarm volume on **every** node so the external volume can be
   attached when the stack is deployed. Replace `pi-manager` with your NFS
   server's IP/hostname if different:

   ```bash
   docker volume create \
     --driver local \
     --opt type=nfs \
     --opt o=addr=pi-manager,rw,nfsvers=4 \
     --opt device=:/srv/textures \
     textures-share
   ```

3. Verify the mount works on the manager and each worker before deploying the
   stack:

   ```bash
   docker run --rm -v textures-share:/srv/textures alpine sh -c "touch /srv/textures/.probe && ls -l /srv/textures/.probe"
   ```

   Confirm the probe file appears on all nodes (remove it afterwards if
   desired).

1. (Optional) Label GPU-capable nodes so the Real-ESRGAN workers land where
   hardware acceleration is available:

   ```bash
   docker node update --label-add gpu=true <node-name>
   ```

1. If you are deploying the texture stack without first running the demo stack,
   create the shared overlay network in advance (idempotent and safe to rerun):

   ```bash
   docker network create --driver overlay --attachable --opt encrypted=true cluster_net
   ```

1. Deploy the stack with the helper script, which auto-scales GPU replicas to
   the number of nodes labeled `gpu=true` (or zero if none) and reuses the
   existing `cluster_net` overlay network created by the demo scripts:

   ```bash
   sudo scripts/deploy-textures.sh stack/textures.yml swg-textures
   ```

   The compose file includes:
   - **redis-bus** – shared message queue on the manager for simple channel
     orchestration (`texture:ingest`, `texture:upscaled`, `texture:metadata`).
   - **texture-ingestion** – publishes new work from `/srv/textures/incoming`
     into Redis so the upscalers can pick it up.
   - **esrgan-upscaler-gpu** – Real-ESRGAN workers pinned to GPU-labeled
     workers (replicated across nodes with `max_replicas_per_node: 1`). The
     helper script sets `deploy.replicas` to the number of GPU nodes discovered
     via `docker node ls --filter node.label=gpu=true` so the service scales
     with available hardware.
   - **esrgan-upscaler-cpu** – CPU fallback upscalers for non-GPU workers. When
     no GPU nodes are present, the helper script sets GPU replicas to zero and
     these workers continue handling requests. Listens on `INPUT_CHANNEL` for
     `scan` events, pulls textures from `/srv/textures/incoming`, writes
     CPU-safe upscales into `/srv/textures/output`, and emits status/results on
     `OUTPUT_CHANNEL` (JSON events for online status, completed scans,
     successes, and errors).
   - **metadata-analyzer** – LLM-driven metadata annotator that reads
     upscales, generates captions/tags through a configurable LLM backend, and
     publishes structured metadata on its own Redis channel while writing
     sidecar JSON next to each texture.

   Configure the analyzer with `LLM_BACKEND` (for example `openai`, `local`, or
   the default `echo` mode), `LLM_MODEL`, `LLM_ENDPOINT`, and `LLM_API_KEY` as
   needed. Metadata is persisted alongside each texture with the suffix
   `.metadata.json` and mirrored to the `texture:metadata` channel together with
   latency metrics for observability.

   All services mount the shared `textures-share` volume at `/srv/textures` so
   inputs/outputs stay consistent across nodes. The stack reuses the
   pre-existing `cluster_net` overlay so Redis and the workers can reach each
   other without extra configuration.

## Exporting and syncing rendered textures
Use `scripts/export-and-sync.sh` to watch the shared output volume and push new
textures to a dedicated Git repository (for example, an `HDtextureDDS` repo).

- **What it does:**
  - Watches `WATCH_DIR` (default: `/srv/textures/output`) for new or updated
    files and stages only texture artifacts that match `FILE_PATTERNS`
    (default: `*.dds *.png`).
  - Syncs staged files into `EXPORT_DIR` (default: `/srv/textures/export`) using
    `rsync`, commits them, and pushes to `GIT_REMOTE_URL` on `GIT_BRANCH`.
  - Supports SSH deploy keys via `GIT_SSH_KEY_PATH` or HTTPS tokens embedded in
    `GIT_REMOTE_URL` (e.g., `https://<token>@github.com/org/HDtextureDDS.git`).
  - Logs to stdout and, if `LOG_FILE` is set, appends to the specified file for
    operators to monitor.

- **Dependencies:**
  - `git` for cloning, committing, and pushing updates.
  - `rsync` for mirroring rendered textures into the export workspace.
  - Optional: `inotifywait` (from `inotify-tools`) for event-driven file
    watching; otherwise the script falls back to interval polling.
  - Install the Raspberry Pi OS dependencies with:

    ```bash
    sudo apt-get install -y git rsync inotify-tools
    ```

    `inotify-tools` is optional, so skip it if unavailable.

- **Configuration:** by default, the script loads environment variables from a
  `.env` file next to the script (`ENV_FILE` defaults to `scripts/.env`). You
  can override `ENV_FILE` or set env vars in the service unit to avoid
  hardcoding secrets:

  ```env
  WATCH_DIR=/srv/textures/output
  EXPORT_DIR=/srv/textures/export
  FILE_PATTERNS="*.dds *.png"
  GIT_REMOTE_URL=git@github.com:org/HDtextureDDS.git
  GIT_BRANCH=main
  GIT_SSH_KEY_PATH=/home/pi/.ssh/deploy_key
  GIT_USER_NAME=Texture Sync Bot
  GIT_USER_EMAIL=bot@example.com
  LOG_FILE=/var/log/texture-export.log
  # Optional: override where staged files are archived if pushes can't reach GitHub
  FALLBACK_ARCHIVE_DIR=/srv/textures/export/manual_review
  ```

- **Syncing to the `polsommer/HDtextureDDS` repo:** set
  `GIT_REMOTE_URL=https://github.com/polsommer/HDtextureDDS.git` (or the SSH
  equivalent) and `GIT_BRANCH=main` in your `.env`. On the first run, the script
  clones the existing history into `EXPORT_DIR` before watching for new assets,
  preserving the repo’s commit lineage while automatically pushing subsequent
  renders from the cluster.

- **Running manually:**

  ```bash
  bash scripts/export-and-sync.sh
  ```

- **Scheduling options:**
  - **systemd service + timer (recommended):** bundled units live in
    `systemd/texture-sync.service` and `systemd/texture-sync.timer`, referencing
    `/srv/textures/.env` plus the repo’s `scripts/export-and-sync.sh` path.
    Install and enable them with one command:

    ```bash
    sudo bash scripts/install-texture-sync-service.sh
    ```

    The helper copies the units into `/etc/systemd/system`, reloads the daemon,
    and starts `texture-sync.timer` immediately.

  - **Cron:** run a periodic sync if `inotifywait` isn’t available:

    ```cron
    */5 * * * * /usr/bin/env bash /srv/SWG-RPI-Cluster/scripts/export-and-sync.sh >> /var/log/texture-export.log 2>&1
    ```

  - **Swarm service:** deploy as a sidecar that mounts the shared volume and
    your SSH key/`.env` secret:

    ```bash
    docker service create \
      --name texture-sync \
      --mount type=bind,src=/srv/textures,dst=/srv/textures \
      --secret source=texture-env,target=/srv/textures/.env,mode=0400 \
      --mount type=bind,src=/home/pi/.ssh/id_ed25519,dst=/id_ed25519,ro \
      --env GIT_SSH_KEY_PATH=/id_ed25519 \
      --env ENV_FILE=/srv/textures/.env \
      --restart-condition=any \
      bash:5.2 bash -lc '/workspace/SWG-RPI-Cluster/scripts/export-and-sync.sh'
    ```

- **Failure handling and logging:**
  - Script exits on errors and writes tagged log lines; configure `LOG_FILE` and
    `systemd` journal forwarding to alerting if needed.
  - Pushes are skipped (but not fatal) when `GIT_REMOTE_URL` is unset so local
    staging can proceed. When GitHub upload isn’t available, staged files are
    copied to a timestamped archive in `FALLBACK_ARCHIVE_DIR` (defaults to
    `<EXPORT_DIR>/manual_review`) for manual inspection/upload.
  - `Restart=always` in systemd or `--restart-condition=any` in swarm keeps the
    watcher alive after transient failures.

## Deploying the swarm dashboard by itself
The Flask-based UI can be deployed without the demo services if you only need a
live swarm view. See [dashboard/README.md](dashboard/README.md) for a
step-by-step guide that covers building the image, wiring up `cluster_net`, and
deploying the service with a one-liner stack file.
