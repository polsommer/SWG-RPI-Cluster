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
  IP addresses
- A shared LAN plus consistent hostnames (examples: `pi-manager`,
  `pi-worker1`, `pi-worker2`)
- `sudo` access on every node
- (Optional) DNS or mDNS so you can visit `http://pi-manager:8080`

## Quickstart
Each step is a one-liner. Run them in order.

### 1) Install Docker (all nodes)
Copy this repo to each Pi or download the scripts, then run:

```bash
sudo bash scripts/install-docker.sh
```

Reboot once so the added cgroup flags in `/boot/cmdline.txt` take effect. The
script is safe to rerun whenever you add or rebuild a node.

### 2) Initialize the swarm (manager)
On `pi-manager` (swap in your preferred advertise IP if needed):

```bash
sudo bash scripts/init-swarm.sh 192.168.1.50
```

You’ll see **manager** and **worker** join tokens plus the advertise address
used. If the node already belongs to a swarm, the script exits with the current
membership details instead of reinitializing.

### 3) Join the workers
On `pi-worker1` and `pi-worker2`, supply the worker token and manager IP:

```bash
sudo bash scripts/join-swarm.sh 192.168.1.50 SWMTKN-1-...
```

If a worker is already in a swarm, leave first with `docker swarm leave --force`
and rerun the command. Confirm membership from the manager:

```bash
docker node ls
```

### 4) Deploy the demo stack (manager)
Ship the overlay network and demo services with:

```bash
sudo bash scripts/deploy-demo.sh
```

The compose file (`stack/demo.yml`) launches:
- **hello-app** – `traefik/whoami` with 3 replicas (one per node) on port 8080.
- **redis-bus** – `redis:7-alpine` pinned to the manager with a named volume for
  durability.
- **echo-worker** – two curl-based workers that continually hit `hello-app`
  across the overlay network, logging responder hostname and latency.
- **swarm-dashboard** – Flask UI on port 8081 (manager-only) that streams swarm
  health, replica counts, and Redis reachability from the Docker API.

### 5) Verify cross-node traffic
- Hit the published service: `curl http://192.168.1.50:8080` (or
  `http://pi-manager.local:8080`). The `Hostname` should vary between nodes.
- Open the dashboard at `http://192.168.1.50:8081` for live metrics and overlay
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
