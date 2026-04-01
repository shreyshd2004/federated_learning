# FedGuard: Federated Learning for Network Anomaly Detection

A working MVP federated learning platform built as a senior design capstone.
Three edge nodes collaboratively train a shared model **without ever exchanging raw data**.

**How to clone, build, run, configure, and troubleshoot:** see **[`docs/USER-GUIDE.md`](../docs/USER-GUIDE.md)** (the project user guide).

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

## Project structure

```
fedguard/
├── server/
│   ├── main.py              # FastAPI server (coordinates training rounds)
│   ├── aggregator.py        # Federated averaging (FedAvg)
│   ├── model.py             # Global model management + evaluation
│   └── requirements.txt
├── node/
│   ├── main.py              # Node client (polling training loop)
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

## API (summary)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Round, history, node participation |
| `GET` | `/get_model` | Download current global weights (binary) |
| `POST` | `/submit_weights` | Upload node weights (multipart form) |
| `POST` | `/reset` | Reset training state (dev utility) |

Interactive docs when the server is running: `http://localhost:8000/docs`

---

## Stretch goals

- **Differential privacy**: add Gaussian noise to weights before upload
- **Secure aggregation**: encrypt weights so server learns only the average
- **Non-IID data**: assign class-skewed shards to simulate heterogeneous nodes
- **gRPC transport**: replace REST with gRPC for lower serialisation overhead
