# Swarm Dashboard Deployment Guide

The Flask-based dashboard ships with the demo stack so you can see swarm health, overlay network reachability, and worker activity in one page. Use these steps when you want to deploy or iterate on the UI quickly.

The Control Center panel now exposes join commands, one-click token rotation, service scaling, and node drain/activation toggles so you can administer the swarm without leaving the browser.

## Prerequisites
- The node must already belong to the Docker Swarm (manager node recommended because the dashboard needs the Docker socket).
- Docker CLI access with permission to read `/var/run/docker.sock`.
- An attachable overlay network named `cluster_net` (created automatically by the demo deploy script).

## 1) Build the dashboard image
The stack expects a locally tagged image named `swg-dashboard:local`.

```bash
cd /path/to/SWG-RPI-Cluster
sudo docker build -t swg-dashboard:local dashboard
```

> Tip: Rebuild after modifying `dashboard/app.py` or the static assets to pick up your changes.

## 2) Create the overlay network (if missing)
The dashboard talks to Redis and other services over `cluster_net`. Create it once per swarm:

```bash
sudo docker network create --driver overlay --attachable cluster_net
```

If the network already exists, Docker will print an error you can safely ignore.

## 3) Deploy the dashboard service
Deploy just the UI (without the full demo stack) using a small compose file passed via stdin:

```bash
cat <<'STACK' | sudo docker stack deploy -c - swg-dashboard
version: "3.9"
services:
  swarm-dashboard:
    image: swg-dashboard:local
    environment:
      - OVERLAY_NETWORK=cluster_net
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - cluster_net
    ports:
      - "8081:8081"
    deploy:
      placement:
        constraints:
          - node.role == manager
networks:
  cluster_net:
    external: true
STACK
```

Once deployed, open `http://<manager-ip>:8081` for the live UI.

## 4) Verify and maintain
- Check service status:
  ```bash
  sudo docker service ls | grep swarm-dashboard
  ```
- Tail the logs while debugging:
  ```bash
  sudo docker service logs -f swg-dashboard_swarm-dashboard
  ```
- Remove the dashboard if you no longer need it:
  ```bash
  sudo docker stack rm swg-dashboard
  ```

## One-liner demo deploy
If you just want the full demo (hello-app, Redis, echo workers, and dashboard), reuse the existing helper:

```bash
sudo bash scripts/deploy-demo.sh
```

It builds the dashboard image, ensures `cluster_net` exists, and deploys everything as the `swg-demo` stack.
