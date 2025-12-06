# SWG-RPI-Cluster

A lightweight recipe for a **three-node Raspberry Pi Docker Swarm**. The scripts
install Docker, bootstrap the swarm, and deploy a small demo stack so the Pis
can talk to each other over an overlay network. Extra checks keep the flow
resilient (idempotent installs, token validation, network preflights) so you can
repeatably rebuild the cluster.

> Tested on Raspberry Pi OS 64-bit (Bookworm); Docker and the chosen images are
> multi-arch so the same steps work on Pi 3/4/5.

## Hardware and network prep
- Three Raspberry Pi boards with 64-bit OS, SSH enabled, and static or reserved
  DHCP addresses
- A shared LAN and consistent hostnames (examples below use `pi-manager`,
  `pi-worker1`, `pi-worker2`)
- `sudo` access to each node
- (Optional) A DNS entry or mDNS so you can `curl http://pi-manager:8080`

## 1) Install Docker on every node
Copy this repo to each Pi or download the scripts, then run:

```bash
sudo bash scripts/install-docker.sh
```

Reboot after the script adds cgroup flags to `/boot/cmdline.txt`. The script is
idempotent, so you can rerun it safely on all nodes if you add more Pis later.

## 2) Initialize the swarm on the manager
On `pi-manager` (replace the IP if you prefer a different advertise address):

```bash
sudo bash scripts/init-swarm.sh 192.168.1.50
```

The script prints **manager** and **worker** join tokens plus a reminder of the
advertise address it used. If the manager already belongs to a swarm, the script
refuses to double-init and shows the existing node info.

## 3) Join the two workers
Run on `pi-worker1` and `pi-worker2`, using the worker token and manager IP:

```bash
sudo bash scripts/join-swarm.sh 192.168.1.50 SWMTKN-1-...
```

If a worker was already part of a swarm, the script refuses to rejoin until you
`docker swarm leave --force`. Verify membership from the manager:

```bash
docker node ls
```

## 4) Deploy the demo stack
From the manager, deploy the overlay network plus a trio of services that prove
cross-node communication:

```bash
sudo bash scripts/deploy-demo.sh
```

What gets deployed (`stack/demo.yml`):
- **hello-app** – `traefik/whoami` with 3 replicas, one per node, published on
  port **8080**.
- **redis-bus** – `redis:7-alpine` pinned to the manager for lightweight
  message passing/storage, with a named volume for durability.
- **echo-worker** – two curl-based workers that continuously hit `hello-app`
  over the overlay network, logging the responding node name and response time.
- **swarm-dashboard** – a Flask web UI on port **8081** (manager-only) that
  streams live swarm info (node status, service replicas, overlay reachability)
  from the Docker API.

## 5) Validate cross-node traffic
- Hit the published service from your laptop: `curl http://192.168.1.50:8080`.
  Successive calls should show different `Hostname` values as the swarm routes
  to different nodes. If you enabled mDNS, you can also try
  `curl http://pi-manager.local:8080`.
- Open the dashboard for a live view of the swarm: `http://192.168.1.50:8081`
  (or `http://pi-manager.local:8081`). It shows CPU/memory, node reachability,
  service replica counts, and a Redis mesh ping.
- Watch the workers streaming responses from inside the cluster:

```bash
docker service logs -f swg-demo_echo-worker
```

- Confirm services are spread across nodes:

```bash
docker service ps swg-demo_hello-app
```

- Confirm Redis is healthy and writable within the network:

```bash
docker run --rm --network cluster_net redis:7-alpine redis-cli -h redis-bus PING
```

## 6) Tear down (optional)

```bash
docker stack rm swg-demo
```

The overlay network `cluster_net` is created once and reused; remove it with
`docker network rm cluster_net` if you want to clean everything.
