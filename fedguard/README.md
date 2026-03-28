# FedGuard — Federated Learning for Network Anomaly Detection

A working MVP federated learning platform built as a senior design capstone.
Three edge nodes collaboratively train a shared model **without ever exchanging raw data**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CENTRAL SERVER                       │
│  - Holds global model                                   │
│  - GET  /get_model    (nodes pull current weights)      │
│  - POST /submit_weights (nodes push local weights)      │
│  - Runs FedAvg once MIN_NODES have reported             │
│  - Tracks rounds, accuracy, participation               │
└─────────────────────┬───────────────────────────────────┘
                      │ REST (binary weights only)
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ NODE 1  │   │ NODE 2  │   │ NODE 3  │
   │ Data: A │   │ Data: B │   │ Data: C │
   └─────────┘   └─────────┘   └─────────┘
   (private, non-overlapping MNIST shards)

   ┌──────────────────────────┐
   │   Streamlit Dashboard    │
   │  :8501 → polls /status   │
   └──────────────────────────┘
```

### Communication
- Nodes **pull** model weights (`GET /get_model`) at the start of every round.
- Nodes **push** only the trained weight tensors (`POST /submit_weights`).
  Raw training data never leaves the node.
- Server runs **Federated Averaging** (FedAvg) once ≥ 2/3 nodes have submitted,
  providing fault-tolerance against a single node failure.

---

## Project Structure

```
fedguard/
├── server/
│   ├── main.py              # FastAPI server — coordinates training rounds
│   ├── aggregator.py        # Federated averaging (FedAvg)
│   ├── model.py             # Global model management + evaluation
│   └── requirements.txt
├── node/
│   ├── main.py              # Node client — polling training loop
│   ├── trainer.py           # Local training (SGD, CrossEntropyLoss)
│   ├── data_loader.py       # MNIST sharding (non-overlapping partitions)
│   └── requirements.txt
├── dashboard/
│   ├── app.py               # Streamlit live dashboard
│   └── requirements.txt
├── shared/
│   └── model_def.py         # Shared SimpleMLP architecture
├── Dockerfile.server
├── Dockerfile.node
├── Dockerfile.dashboard
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites
- Docker ≥ 24 with Docker Compose v2
- ~3 GB disk for PyTorch images + MNIST cache

### Run

```bash
cd fedguard

# Build & launch everything
docker-compose up --build

# Watch logs from all containers
docker-compose logs -f

# Dashboard → http://localhost:8501
# Server API → http://localhost:8000/docs
```

The system will:
1. Start the central server and wait for it to be healthy.
2. Launch 3 node containers, each with its own private MNIST shard.
3. Complete 10 federated rounds, printing accuracy after each round.
4. Display live progress on the Streamlit dashboard.

### Tear down

```bash
docker-compose down -v
```

---

## Configuration

All tuneable parameters are set via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `NODE_ID` | — | 1-indexed node identifier |
| `SERVER_URL` | `http://server:8000` | Central server URL |
| `TOTAL_NODES` | `3` | Total nodes in federation |
| `LOCAL_EPOCHS` | `2` | Local training epochs per round |
| `MAX_ROUNDS` | `10` | How many federated rounds to run |
| `ROUND_POLL_INTERVAL` | `5` | Seconds between rounds |

---

## Model

**SimpleMLP** — a 2-layer fully-connected network:

```
Input (784) → Linear(784→128) → ReLU → Linear(128→10) → logits
```

Trained with SGD (lr=0.01, momentum=0.9) and CrossEntropyLoss on MNIST.
Typical accuracy after 10 rounds: **~90–93%**.

---

## Fault Tolerance

The server aggregates as soon as **2 of 3** nodes have submitted weights
(`MIN_NODES_FOR_AGGREGATION = 2`).  A single dead node does not stall training.

---

## Federated Averaging

```python
avg_weights[key] = mean( node1_weights[key],
                         node2_weights[key],
                         node3_weights[key] )
```

All nodes are weighted equally (uniform FedAvg).

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Round, history, node participation |
| `GET` | `/get_model` | Download current global weights (binary) |
| `POST` | `/submit_weights` | Upload node weights (multipart form) |
| `POST` | `/reset` | Reset training state (dev utility) |

Interactive docs: `http://localhost:8000/docs`

---

## Stretch Goals

- **Differential privacy** — add Gaussian noise to weights before upload
- **Secure aggregation** — encrypt weights so server learns only the average
- **Non-IID data** — assign class-skewed shards to simulate heterogeneous nodes
- **gRPC transport** — replace REST with gRPC for lower serialisation overhead
