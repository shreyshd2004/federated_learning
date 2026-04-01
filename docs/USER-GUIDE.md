# FedGuard: User Guide (Recreate & Run the Project)

This guide walks you through **cloning, building, and running** the FedGuard federated learning stack (central server, edge nodes, Streamlit dashboard) from scratch. It includes **links to documentation and open-source projects**, plus **visuals** for the finished system and the build flow.

**Upstream repository:** [github.com/shreyshd2004/federated_learning](https://github.com/shreyshd2004/federated_learning). If you use a fork, substitute your fork’s URL when cloning.

---

## Table of contents

1. [What you are building](#1-what-you-are-building)
2. [Prerequisites](#2-prerequisites)
3. [Get the code](#3-get-the-code)
4. [Build and run (Docker)](#4-build-and-run-docker)
5. [Verify the deployment](#5-verify-the-deployment)
6. [Configuration](#6-configuration)
7. [Optional: develop without Docker](#7-optional-develop-without-docker)
8. [Troubleshooting](#8-troubleshooting)
9. [Resources & references](#9-resources--references)
10. [Visual walkthrough](#10-visual-walkthrough)

---

## 1. What you are building

FedGuard is a **federated learning** demo for collaborative model training: a **FastAPI** server holds the global model; **three node clients** train on **local data shards** (MNIST or NSL-KDD) and upload **only model weights** (not raw data). The server aggregates with **FedAvg** (and optional **median**, **trimmed mean**, **Krum**), with optional **Byzantine detection**, **FedProx**, **DP-SGD (Opacus)**, and **compression**. A **Streamlit** dashboard shows rounds, accuracy, and diagnostics.

---

## 2. Prerequisites

| Requirement | Notes | Official link |
|-------------|--------|----------------|
| **Docker** ≥ 24 | Engine + Compose v2 | [Docker Engine install](https://docs.docker.com/engine/install/) |
| **Docker Compose** | Usually bundled as `docker compose` | [Compose overview](https://docs.docker.com/compose/) |
| **Disk** | ~3 GB+ for PyTorch images and dataset cache | (none) |
| **Git** | To clone the repository | [Git downloads](https://git-scm.com/downloads) |

**Conceptual background (optional reading):**

- [Federated Learning: McMahan et al., “Communication-Efficient Learning…” (FedAvg)](https://arxiv.org/abs/1602.05629)
- [PyTorch](https://pytorch.org/): deep learning framework used by server and nodes
- [FastAPI](https://fastapi.tiangolo.com/): server API
- [Streamlit](https://streamlit.io/): dashboard

---

## 3. Get the code

### Step 1: Clone

```bash
git clone https://github.com/shreyshd2004/federated_learning.git
cd federated_learning
```

If your team uses a fork, replace the URL with your fork’s clone URL.

### Step 2: Update to latest (optional)

```bash
git fetch origin
git checkout main
git pull origin main
```

If your local `main` has diverged from the remote (for example after a force-push), align with the remote only if you intend to discard local-only commits:

```bash
git fetch origin
git reset --hard origin/main
```

---

## 4. Build and run (Docker)

All commands below assume your working directory is the repo root **`federated_learning`**, then **`fedguard`**.

### Step 1: Enter the project folder

```bash
cd fedguard
```

### Step 2: Build and start all services

```bash
docker compose up --build
```

On older Docker installations, you may need:

```bash
docker-compose up --build
```

This builds and runs:

- **server**: FastAPI on port **8000**
- **node1**, **node2**, **node3**: training clients
- **dashboard**: Streamlit on port **8501**

### Step 3: Watch logs (optional)

In another terminal:

```bash
cd fedguard
docker compose logs -f
```

### Step 4: Stop and remove containers

```bash
docker compose down -v
```

The `-v` flag removes the named volume used for MNIST (and dataset) cache so the next run starts with a clean cache if desired.

---

## 5. Verify the deployment

| Step | Action | Success criteria |
|------|--------|------------------|
| 1 | Open **API docs** in a browser | [http://localhost:8000/docs](http://localhost:8000/docs) loads Swagger UI |
| 2 | Open **dashboard** | [http://localhost:8501](http://localhost:8501) shows FedGuard metrics and charts |
| 3 | Check **health** | `curl http://localhost:8000/health` returns a healthy response |

The stack runs a fixed number of federated rounds (default **10**). Nodes poll the server; aggregation runs when enough nodes submit (default **2 of 3**).

---

## 6. Configuration

Tune behavior via **environment variables** in [`fedguard/docker-compose.yml`](../fedguard/docker-compose.yml). Highlights:

| Variable (examples) | Role |
|---------------------|------|
| `DATASET` | `mnist` or `nslkdd` |
| `AGGREGATION_STRATEGY` | `fedavg`, `median`, `trimmed_mean`, `krum` |
| `DIRICHLET_ALPHA` | Non-IID skew (lower = more skewed) |
| `FEDPROX_MU` | FedProx proximal term (`0` ≈ FedAvg only) |
| `ENABLE_DP` | `true` / `false` for DP-SGD (Opacus) on nodes |
| `ENABLE_COMPRESSION` | Top-K style compression on uploads |
| `NODE_POISONED` (e.g. node3) | Simulate Byzantine behavior |
| `BYZANTINE_DETECTION` | Server-side defence flags |
| `MIN_NODES`, `MAX_ROUNDS`, `LOCAL_EPOCHS` | Federation schedule |

After edits, rebuild:

```bash
docker compose up --build
```

---

## 7. Optional: develop without Docker

For quick iteration you can run components locally with Python 3.10+ and virtual environments:

1. **Server:** `cd fedguard/server && pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000`
2. **Node:** set `SERVER_URL=http://localhost:8000`, `cd fedguard/node && pip install -r requirements.txt && python main.py`
3. **Dashboard:** `cd fedguard/dashboard && pip install -r requirements.txt && streamlit run app.py`

You must start the server before nodes and the dashboard. Paths may need `PYTHONPATH` including `fedguard` so `shared/model_def.py` resolves; Dockerfiles set this up for you.

---

## 8. Troubleshooting

| Issue | What to try |
|-------|-------------|
| Port 8000 or 8501 in use | Change host ports in `docker-compose.yml` or stop conflicting apps |
| Nodes stuck / 409 errors | Server uses FL **cycle** gating; ensure only one logical training run; use `/reset` from the dashboard if needed |
| Out of memory | Reduce `LOCAL_EPOCHS` or batch size in code; close other GPU/CPU-heavy apps |
| Stale images after code changes | `docker compose build --no-cache` then `up` |

---

## 9. Resources & references

### This project

- **Repository:** [https://github.com/shreyshd2004/federated_learning](https://github.com/shreyshd2004/federated_learning)
- **Compose file:** [`fedguard/docker-compose.yml`](../fedguard/docker-compose.yml)
- **README:** [`fedguard/README.md`](../fedguard/README.md)

### Core dependencies (open source)

| Package | License (typical) | Link |
|---------|-------------------|------|
| PyTorch | BSD-style | [https://pytorch.org/](https://pytorch.org/) |
| FastAPI | MIT | [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/) |
| Uvicorn | BSD | [https://www.uvicorn.org/](https://www.uvicorn.org/) |
| Streamlit | Apache 2.0 | [https://streamlit.io/](https://streamlit.io/) |
| Opacus (DP) | Apache 2.0 | [https://opacus.ai/](https://opacus.ai/) |
| scikit-learn | BSD | [https://scikit-learn.org/](https://scikit-learn.org/) |

### Learning resources

- [Docker Compose documentation](https://docs.docker.com/compose/)
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (benchmark; downloaded via torchvision)
- [Papers With Code: Federated Learning](https://paperswithcode.com/task/federated-learning)

---





