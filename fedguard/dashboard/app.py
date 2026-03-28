"""
FedGuard Streamlit Dashboard

Shows:
- Current training round
- Per-node participation status
- Global accuracy over rounds (line chart)
- Round-by-round metrics table
- Live auto-refresh every 5 seconds
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
REFRESH_INTERVAL = 5  # seconds

st.set_page_config(
    page_title="FedGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ FedGuard — Federated Learning Dashboard")
st.caption("Real-time training monitor for distributed anomaly detection")

# ---------------------------------------------------------------------------
# Fetch status from server
# ---------------------------------------------------------------------------

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_status():
    try:
        r = requests.get(f"{SERVER_URL}/status", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


status, error = fetch_status()

if error:
    st.error(f"Cannot reach server at `{SERVER_URL}`: {error}")
    st.info("Make sure the FedGuard server is running (`docker-compose up server`)")
    st.stop()

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
current_round = status.get("round", 0)
history = status.get("history", [])
known_nodes = status.get("known_nodes", [])
pending_nodes = status.get("pending_nodes", [])
total_expected = status.get("total_expected_nodes", 3)
min_needed = status.get("min_nodes_for_aggregation", 2)

latest_acc = history[-1]["accuracy"] if history else None
best_acc = max((h["accuracy"] for h in history), default=None)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Round", current_round)
col2.metric(
    "Latest Accuracy",
    f"{latest_acc:.2%}" if latest_acc is not None else "—",
)
col3.metric(
    "Best Accuracy",
    f"{best_acc:.2%}" if best_acc is not None else "—",
)
col4.metric(
    "Active Nodes",
    f"{len(known_nodes)} / {total_expected}",
)

st.divider()

# ---------------------------------------------------------------------------
# Node participation
# ---------------------------------------------------------------------------
st.subheader("Node Participation")
node_cols = st.columns(total_expected)
expected_ids = [str(i) for i in range(1, total_expected + 1)]

for i, node_id in enumerate(expected_ids):
    with node_cols[i]:
        if node_id in pending_nodes:
            st.success(f"Node {node_id}\n\n⏳ Weights pending")
        elif node_id in known_nodes:
            st.info(f"Node {node_id}\n\n✅ Submitted this round")
        else:
            st.warning(f"Node {node_id}\n\n💤 Not yet seen")

if current_round == 0 and not known_nodes:
    st.info(f"Waiting for nodes to start training... (need {min_needed}/{total_expected})")

st.divider()

# ---------------------------------------------------------------------------
# Accuracy chart
# ---------------------------------------------------------------------------
st.subheader("Global Accuracy Over Rounds")

if history:
    df = pd.DataFrame(history)
    df["round"] = df["round"].astype(int)
    df["accuracy_pct"] = df["accuracy"] * 100

    # Line chart
    chart_data = df.set_index("round")[["accuracy_pct"]]
    chart_data.columns = ["Accuracy (%)"]
    st.line_chart(chart_data, use_container_width=True)

    # Improvement callout
    if len(history) >= 2:
        delta = history[-1]["accuracy"] - history[0]["accuracy"]
        st.caption(
            f"Accuracy improved by **{delta:+.2%}** from round 1 to round {current_round}"
        )
else:
    st.info("No rounds completed yet — accuracy chart will appear once training begins.")

st.divider()

# ---------------------------------------------------------------------------
# Round history table
# ---------------------------------------------------------------------------
st.subheader("Round History")

if history:
    table_df = pd.DataFrame(history)
    table_df["accuracy"] = table_df["accuracy"].apply(lambda x: f"{x:.2%}")
    table_df["timestamp"] = pd.to_datetime(table_df["timestamp"], unit="s").dt.strftime(
        "%H:%M:%S"
    )
    table_df.columns = ["Round", "Accuracy", "Nodes", "Time"]
    st.dataframe(table_df[::-1], use_container_width=True, hide_index=True)
else:
    st.caption("No rounds completed yet.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
st.divider()
refresh_col, _ = st.columns([1, 3])
with refresh_col:
    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s · Server: `{SERVER_URL}`")
time.sleep(REFRESH_INTERVAL)
st.rerun()
