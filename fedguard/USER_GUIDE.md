# FedGuard User Guide
### Federated Learning System for Network Anomaly Detection

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Repository Structure](#4-repository-structure)
5. [Step-by-Step Setup](#5-step-by-step-setup)
6. [Running the System](#6-running-the-system)
7. [Using the Dashboard](#7-using-the-dashboard)
8. [Running the DLG Attack Lab](#8-running-the-dlg-attack-lab)
9. [Configuration Reference](#9-configuration-reference)
10. [Experiment Scenarios](#10-experiment-scenarios)
11. [Running the Test Suite](#11-running-the-test-suite)
12. [Rebuilding After Code Changes](#12-rebuilding-after-code-changes)
13. [Key Resources and Papers](#13-key-resources-and-papers)

---

## 1. Project Overview

FedGuard is a research-grade federated learning (FL) system for network anomaly detection. Multiple client nodes collaboratively train an intrusion detection model without ever sharing raw data. The server aggregates model updates using configurable strategies, defends against Byzantine attackers, and enforces differential privacy accounting. A separate attack module demonstrates gradient inversion (DLG/iDLG) and shows how differential privacy mitigates it.

**Core features:**
- FedAvg and FedProx local training
- Differential Privacy via DP-SGD (Opacus)
- Byzantine fault tolerance: Krum, cosine screening, L2-norm outlier detection
- Top-K gradient compression with error feedback
- Non-IID data partitioning via Dirichlet distribution
- FL cycle gating (stale upload rejection)
- Deep Leakage from Gradients (DLG / iDLG) attack module
- Real-time Streamlit monitoring dashboard
- Full Docker Compose orchestration

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Network                       │
│                                                         │
│  ┌──────────┐   weights    ┌──────────────────────────┐ │
│  │  node1   │ ──────────►  │                          │ │
│  └──────────┘              │   FL Server (:8000)      │ │
│  ┌──────────┐   weights    │   - FedAvg / FedProx     │ │
│  │  node2   │ ──────────►  │   - Byzantine defence    │ │
│  └──────────┘              │   - Cycle gating         │ │
│  ┌──────────┐   weights    │   - Aggregation noise    │ │
│  │  node3   │ ──────────►  │                          │ │
│  └──────────┘              └──────────────────────────┘ │
│       │                            │                    │
│       │ local data                 │ global model       │
│       ▼                            ▼                    │
│  ┌──────────────┐       ┌────────────────────┐         │
│  │  mnist_cache │       │  Dashboard (:8501) │         │
│  │   (volume)   │       │  Attack Lab        │         │
│  └──────────────┘       └────────────────────┘         │
│                          ┌────────────────────┐         │
│                          │  Attack Svc (:8888)│         │
│                          │  DLG / iDLG        │         │
│                          └────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

**Services:**

| Service   | Port | Description |
|-----------|------|-------------|
| server    | 8000 | FastAPI FL aggregation server |
| node1     | —    | FL client, NODE_ID=1 |
| node2     | —    | FL client, NODE_ID=2 |
| node3     | —    | FL client, NODE_ID=3 (can be Byzantine) |
| dashboard | 8501 | Streamlit monitoring UI |
| attack    | 8888 | FastAPI DLG attack service |

---

## 3. Prerequisites

### Required Software

| Tool | Version | Install |
|------|---------|---------|
| Docker Desktop | 4.x+ | https://www.docker.com/products/docker-desktop |
| Docker Compose | v2 (bundled with Docker Desktop) | included |
| Python | 3.10+ | https://www.python.org/downloads/ (for local testing only) |
| Git | any | https://git-scm.com |

> **Note:** You do NOT need PyTorch, CUDA, or any Python packages installed locally to run the system. Everything runs inside Docker containers.

### Hardware

- **Minimum:** 8 GB RAM, 4-core CPU
- **Recommended:** 16 GB RAM (the MNIST download and model training is CPU-only by default)
- **Disk:** ~4 GB free for Docker images

### Verify Docker is running

```bash
docker --version
docker compose version
docker ps
```

---

## 4. Repository Structure

```
fedguard/
├── docker-compose.yml          # Orchestrates all 5 services
├── Dockerfile.server           # Server image
├── Dockerfile.node             # Node image (shared by node1/2/3)
├── Dockerfile.dashboard        # Streamlit dashboard image
├── Dockerfile.attack           # DLG attack service image
│
├── shared/
│   └── model_def.py            # SimpleMLP (MNIST) + NSLKDDMLP (NSL-KDD)
│
├── server/
│   ├── main.py                 # FastAPI endpoints, FL cycle logic
│   ├── aggregator.py           # FedAvg, median, trimmed mean, Krum
│   ├── defender.py             # Byzantine detection pipeline
│   ├── model.py                # GlobalModel wrapper
│   └── requirements.txt
│
├── node/
│   ├── main.py                 # FL client loop, cycle sync
│   ├── trainer.py              # Standard + DP-SGD training
│   ├── data_loader.py          # MNIST / NSL-KDD + Dirichlet partitioning
│   ├── compressor.py           # Top-K gradient compression
│   └── requirements.txt
│
├── attack/
│   ├── app.py                  # FastAPI attack endpoints
│   ├── dlg_attack.py           # DLG / iDLG gradient inversion
│   └── requirements.txt
│
├── dashboard/
│   ├── app.py                  # Streamlit 10-panel dashboard
│   └── requirements.txt
│
└── tests/
    └── test_all.py             # 81-test suite
```

---

## 5. Step-by-Step Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/shreyshd2004/federated_learning.git
cd federated_learning/fedguard
```

### Step 2: Start Docker Desktop

Open Docker Desktop and wait until the whale icon in your menu bar shows "Docker Desktop is running." You can verify:

```bash
docker info
```

### Step 3: Build all Docker images

This step downloads base images and installs all Python dependencies inside each container. It takes 5–15 minutes the first time.

```bash
docker compose build
```

You should see output like:
```
[+] Building 142.3s (32/32) FINISHED
 => [server] ...
 => [node] ...
 => [dashboard] ...
 => [attack] ...
```

> **Photo/Video:** Record a screen capture of the build output scrolling. This is a good "build process" artifact for your submission.

---

## 6. Running the System

### Step 4: Start all services

```bash
docker compose up
```

This launches the server, 3 nodes, dashboard, and attack service in the correct dependency order. The server health check must pass before nodes start.

You will see interleaved logs from all containers. Look for:

```
fedguard-server   | INFO:     Application startup complete.
fedguard-node1    | [Node 1] Round 1 — downloading model (cycle 0) ...
fedguard-node2    | [Node 2] Round 1 — downloading model (cycle 0) ...
fedguard-node3    | [Node 3] Round 1 — downloading model (cycle 0) ...
fedguard-server   | [Aggregator] Round 1 complete. Accuracy: 0.72
```

### Step 5: Verify the server is healthy

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/status
# {"current_round": 1, "accepting_cycle": 0, "pending_nodes": [...], ...}
```

### Step 6: Watch training progress

Open the dashboard in your browser:

```
http://localhost:8501
```

> **Photo:** Take a screenshot of the dashboard after 3–5 rounds of training. This is the primary visual artifact for your submission.

### Step 7: Stop the system

```bash
# Stop and remove containers (keeps the mnist_cache volume)
docker compose down

# Stop AND delete the cached MNIST data
docker compose down -v
```

---

## 7. Using the Dashboard

The Streamlit dashboard at `http://localhost:8501` has 10 panels:

| Panel | What it shows |
|-------|--------------|
| Global Accuracy | Test accuracy on the held-out MNIST/NSL-KDD set after each round |
| Round History | Table of per-round metrics (accuracy, loss, nodes, strategy) |
| Privacy Budget | Cumulative ε spent per node (only visible when DP is enabled) |
| Local vs Global Accuracy | Comparison of node-local accuracy vs global model accuracy |
| Cosine Similarity Heatmap | Pairwise cosine similarity between node updates (Byzantine nodes appear as outliers) |
| Compression Stats | Sparsity ratio and error feedback magnitude (only when compression is enabled) |
| Node Activity | Which nodes submitted in the last round, how many samples each contributed |
| Aggregation Strategy | Current strategy in use (FedAvg / median / trimmed_mean / Krum) |
| Byzantine Detection Log | Flagged nodes and reason (cosine threshold, norm outlier) |
| Attack Lab | Buttons to trigger DLG / iDLG gradient inversion attacks (see Section 8) |

---

## 8. Running the DLG Attack Lab

The DLG attack reconstructs a training sample from a single gradient update, demonstrating that model updates can leak private data.

### Via the Dashboard

1. Open `http://localhost:8501`
2. Click the **Attack Lab** tab
3. Click **Run Attack** for a single reconstruction at σ=0 (no DP noise)
4. Click **Run Full Comparison** to see reconstructions at σ=0, σ=0.3, and σ=1.1

The dashboard will display:
- Original vs reconstructed image side-by-side
- PSNR (higher = better reconstruction = worse privacy)
- Gradient difference convergence curve

> **Photo:** Screenshot the Attack Lab comparison panel showing three columns (no DP, low DP, high DP). This directly demonstrates the privacy-utility tradeoff.

### Via the API directly

```bash
# Single attack (no DP noise)
curl -X POST http://localhost:8888/run \
  -H "Content-Type: application/json" \
  -d '{"iterations": 300, "lr": 0.1, "noise_multiplier": 0.0}'

# Full comparison (three noise levels)
curl -X POST http://localhost:8888/run_comparison

# Get cached results
curl http://localhost:8888/results
```

Responses include base64-encoded PNG images and convergence data.

---

## 9. Configuration Reference

All configuration is done through environment variables in `docker-compose.yml`. Edit the file, then run `docker compose up` to apply changes.

### Server environment variables

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| DATASET | mnist | mnist, nslkdd | Dataset to train on |
| AGGREGATION_STRATEGY | fedavg | fedavg, median, trimmed_mean, krum | Aggregation algorithm |
| BYZANTINE_DETECTION | true | true, false | Enable cosine + norm screening |
| BYZANTINE_COS_THRESHOLD | 0.0 | float | Cosine similarity cutoff (higher = stricter) |
| BYZANTINE_NORM_SIGMA | 2.0 | float | L2-norm outlier threshold (k-sigma) |
| MIN_NODES | 2 | int | Minimum nodes before aggregation fires |
| TOTAL_NODES | 3 | int | Total expected nodes |
| AGGREGATION_NOISE_STD | 0 | float | Gaussian noise added after aggregation |

### Node environment variables

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| NODE_ID | — | 1, 2, 3 | Unique node identifier |
| DATASET | mnist | mnist, nslkdd | Must match server |
| DIRICHLET_ALPHA | 0.5 | float > 0 | Non-IID skew (0.1=extreme, 100=IID) |
| LOCAL_EPOCHS | 2 | int | Local training epochs per round |
| MAX_ROUNDS | 10 | int | Number of FL rounds to run |
| FEDPROX_MU | 0.01 | float | FedProx proximal coefficient (0=FedAvg) |
| ENABLE_DP | false | true, false | Enable DP-SGD via Opacus |
| DP_EPSILON | 10.0 | float | Target privacy budget ε |
| DP_DELTA | 1e-5 | float | Privacy parameter δ |
| DP_MAX_GRAD_NORM | 1.0 | float | Per-sample gradient clipping norm |
| ENABLE_COMPRESSION | false | true, false | Enable Top-K gradient compression |
| COMPRESSION_TOP_K | 0.1 | float (0–1) | Fraction of gradients to transmit |
| NODE_POISONED | false | true, false | Simulate Byzantine gradient inversion attack |

---

## 10. Experiment Scenarios

These are the key experiments to run and document for your submission.

### Scenario A: Baseline FedAvg (default)

```yaml
# No changes needed from defaults
AGGREGATION_STRATEGY: fedavg
NODE_POISONED: false
ENABLE_DP: false
```

Run 10 rounds and record final test accuracy.

### Scenario B: Byzantine attack + detection

Enable node3 as a Byzantine attacker and observe it being flagged:

```yaml
# In node3's environment:
NODE_POISONED: true

# In server's environment:
BYZANTINE_DETECTION: true
BYZANTINE_COS_THRESHOLD: 0.3
```

Watch the Byzantine Detection Log panel in the dashboard. Node3 should be flagged and excluded from aggregation.

### Scenario C: Differential Privacy

Enable DP on all nodes and compare final accuracy vs ε:

```yaml
ENABLE_DP: true
DP_EPSILON: 10.0   # Try also: 3.0, 1.0
DP_DELTA: 1e-5
DP_MAX_GRAD_NORM: 1.0
```

The Privacy Budget panel in the dashboard shows accumulated ε per round.

### Scenario D: Non-IID data distribution

Make the data highly skewed across nodes:

```yaml
DIRICHLET_ALPHA: 0.1   # Very non-IID (each node mostly sees one class)
# Compare to:
DIRICHLET_ALPHA: 1.0   # Moderate skew
DIRICHLET_ALPHA: 100   # Near-IID
```

### Scenario E: Krum aggregation under Byzantine attack

```yaml
AGGREGATION_STRATEGY: krum
NODE_POISONED: true   # on node3
```

Krum selects the single most-central update, effectively ignoring the Byzantine node.

### Scenario F: Gradient compression

```yaml
ENABLE_COMPRESSION: true
COMPRESSION_TOP_K: 0.1   # Transmit only top 10% of gradients
```

The Compression Stats panel shows how much bandwidth is saved and how error feedback accumulates.

---

## 11. Running the Test Suite

The test suite runs outside Docker against the local Python environment. It covers all major components with 81 tests.

### Install test dependencies locally

```bash
cd fedguard
pip install torch torchvision fastapi uvicorn requests opacus \
            scikit-learn pandas numpy matplotlib httpx pytest
```

### Run all tests

```bash
python -m pytest tests/test_all.py -v
```

Expected output:
```
========================== 81 passed in ~11s ==========================
```

### Run a specific test class

```bash
# Just the aggregator tests
python -m pytest tests/test_all.py::TestAggregator -v

# Just the Byzantine defender tests
python -m pytest tests/test_all.py::TestDefender -v

# Just the DLG attack tests
python -m pytest tests/test_all.py::TestDLGAttack -v
```

### Test classes overview

| Class | Tests | What is verified |
|-------|-------|-----------------|
| TestModelDef | 9 | Architecture shapes, forward pass, parameter counts |
| TestAggregator | 13 | FedAvg, weighted avg, coordinate median, trimmed mean, Krum, noise |
| TestDefender | 12 | Cosine screening accuracy, norm outlier detection, Byzantine rejection |
| TestCompressor | 9 | Top-K sparsity, error feedback convergence, delta computation |
| TestDataLoader | 7 | IID/non-IID splits, disjoint partitions, Dirichlet distribution |
| TestTrainer | 7 | Standard training, FedProx proximal term, DP flag passthrough |
| TestDLGAttack | 10 | Label extraction, reconstruction shape, convergence, DP degradation, PSNR/MSE |
| TestServerEndpoints | 13 | Health/status, model download, cycle gating (409 on stale), aggregation trigger, reset |

---

## 12. Rebuilding After Code Changes

If you edit any Python source file, rebuild the relevant service:

```bash
# Rebuild only the server after editing server/
docker compose build server && docker compose up server

# Rebuild all images after editing shared/model_def.py
docker compose build && docker compose up
```

---

## 13. Key Resources and Papers

### Foundational Papers

1. **FedAvg** — McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
   - https://arxiv.org/abs/1602.05629

2. **Deep Leakage from Gradients (DLG)** — Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019)
   - https://arxiv.org/abs/1906.08935

3. **iDLG** — Zhao et al., "iDLG: Improved Deep Leakage from Gradients" (2020)
   - https://arxiv.org/abs/2001.02610

4. **FedProx** — Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
   - https://arxiv.org/abs/1812.06127

5. **Krum Byzantine Resilience** — Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017)
   - https://arxiv.org/abs/1703.02757

6. **Differential Privacy + FL** — Geyer et al., "Differentially Private Federated Learning: A Client Level Perspective" (2017)
   - https://arxiv.org/abs/1712.07557

### Open-Source Libraries

| Library | Purpose | Link |
|---------|---------|------|
| PyTorch | Neural network training | https://pytorch.org |
| Opacus | DP-SGD differential privacy | https://opacus.ai |
| FastAPI | REST API server/attack service | https://fastapi.tiangolo.com |
| Streamlit | Real-time dashboard | https://streamlit.io |
| Docker Compose | Multi-container orchestration | https://docs.docker.com/compose |
| scikit-learn | NSL-KDD preprocessing | https://scikit-learn.org |

### Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| MNIST | Handwritten digits (used as FL proxy task) | Auto-downloaded via torchvision |
| NSL-KDD | Network intrusion detection benchmark | https://www.unb.ca/cic/datasets/nsl.html |

### GitHub Repository

```
https://github.com/shreyshd2004/federated_learning
```

Branch with all code:
```
claude/fedguard-anomaly-detection-mNdz0
```

---

## Quick Reference: Common Commands

```bash
# Start everything
docker compose up

# Start in background
docker compose up -d

# View logs for a specific service
docker compose logs -f server
docker compose logs -f node1

# Stop everything
docker compose down

# Stop and delete volumes (clears MNIST cache)
docker compose down -v

# Rebuild after code changes
docker compose build

# Run tests locally
python -m pytest tests/test_all.py -v

# Check server health
curl http://localhost:8000/health

# Check FL round status
curl http://localhost:8000/status

# Reset FL state (start new training run)
curl -X POST http://localhost:8000/reset

# Trigger DLG attack manually
curl -X POST http://localhost:8888/run_comparison
```
