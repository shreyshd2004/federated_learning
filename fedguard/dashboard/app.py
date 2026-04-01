"""
FedGuard: Advanced Streamlit Dashboard

Panels
------
1. Header metrics: round, accuracy, nodes, privacy budget
2. Node status: per-node participation + Byzantine flags
3. Accuracy chart: global accuracy over rounds
4. Cosine similarity heatmap: Byzantine detection per round
5. Privacy budget: cumulative ε per round (DP-SGD)
6. Local vs global accuracy: per-node training quality
7. Compression ratio: bandwidth savings per round
8. Round details table: full round history
9. Config sidebar: live system configuration

Auto-refreshes every 5 seconds.
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

SERVER_URL       = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
REFRESH_INTERVAL = 5

st.set_page_config(
    page_title="FedGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_status():
    try:
        r = requests.get(f"{SERVER_URL}/status", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


status, error = fetch_status()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: config
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ FedGuard")
    st.caption("Federated Learning for Anomaly Detection")
    st.divider()

    if status:
        cfg = status.get("config", {})
        st.subheader("System Config")
        st.json(cfg)
        st.divider()

    st.subheader("Controls")
    if st.button("Reset Training", type="primary"):
        try:
            requests.post(f"{SERVER_URL}/reset", timeout=5)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Reset failed: {exc}")

    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Server: `{SERVER_URL}`")
    st.caption(f"Auto-refresh: {REFRESH_INTERVAL}s")

# ─────────────────────────────────────────────────────────────────────────────
# Error state
# ─────────────────────────────────────────────────────────────────────────────
if error:
    st.error(f"Cannot reach server: {error}")
    st.info("Start the system with `docker-compose up`")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Extract state
# ─────────────────────────────────────────────────────────────────────────────
current_round  = status.get("round", 0)
accepting_cycle = status.get("accepting_cycle", status.get("config", {}).get("accepting_cycle", 0))
history        = status.get("history", [])
known_nodes    = status.get("known_nodes", [])
pending_nodes  = status.get("pending_nodes", [])
cfg            = status.get("config", {})
total_nodes    = cfg.get("total_nodes", 3)
byz_enabled    = cfg.get("byzantine_detection", False)

latest         = history[-1] if history else {}
latest_acc     = latest.get("accuracy")
best_acc       = max((h["accuracy"] for h in history), default=None)
latest_eps     = latest.get("avg_epsilon")
flagged_ever   = set(nid for h in history for nid in h.get("flagged_nodes", []))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Header metrics
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ FedGuard: Federated Learning Monitor")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Round", current_round)
c2.metric("FL cycle", accepting_cycle)
c3.metric("Latest Accuracy",  f"{latest_acc:.2%}"  if latest_acc  is not None else "N/A")
c4.metric("Best Accuracy",    f"{best_acc:.2%}"    if best_acc    is not None else "N/A")
c5.metric("Active Nodes",     f"{len(known_nodes)} / {total_nodes}")
c6.metric("Privacy Budget ε", f"{latest_eps:.3f}"  if latest_eps  is not None else "off")

if flagged_ever:
    st.warning(f"Byzantine alert: nodes {sorted(flagged_ever)} flagged in at least one round")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Node status row
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Node Participation")
node_cols = st.columns(total_nodes)
expected_ids = [str(i) for i in range(1, total_nodes + 1)]

for i, nid in enumerate(expected_ids):
    with node_cols[i]:
        ever_flagged = nid in flagged_ever
        if nid in pending_nodes:
            st.warning(f"**Node {nid}**\n\n⏳ Pending")
        elif nid in known_nodes and not ever_flagged:
            st.success(f"**Node {nid}**\n\n✅ Active")
        elif ever_flagged:
            st.error(f"**Node {nid}**\n\n⚠️ Flagged")
        else:
            st.info(f"**Node {nid}**\n\n💤 Waiting")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Accuracy chart + 5. Privacy budget
# ─────────────────────────────────────────────────────────────────────────────
col_acc, col_eps = st.columns(2)

with col_acc:
    st.subheader("Global Accuracy Over Rounds")
    if history:
        df_acc = pd.DataFrame([
            {"Round": h["round"], "Accuracy (%)": h["accuracy"] * 100}
            for h in history
        ]).set_index("Round")
        st.line_chart(df_acc, use_container_width=True)
        if len(history) >= 2:
            delta = history[-1]["accuracy"] - history[0]["accuracy"]
            st.caption(f"Δ accuracy from round 1→{current_round}: **{delta:+.2%}**")
    else:
        st.info("Waiting for first completed round…")

with col_eps:
    st.subheader("Cumulative Privacy Budget (ε)")
    eps_rows = [
        {"Round": h["round"], "ε spent": h["avg_epsilon"]}
        for h in history if h.get("avg_epsilon") is not None
    ]
    if eps_rows:
        df_eps = pd.DataFrame(eps_rows).set_index("Round")
        st.line_chart(df_eps, use_container_width=True)
        st.caption("Lower ε = stronger privacy guarantee (DP-SGD)")
    else:
        st.info("Differential privacy is **disabled**.\nSet `ENABLE_DP=true` to enable.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cosine similarity heatmap (Byzantine detection)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Byzantine Detection: Cosine Similarity per Round")

cos_rows = []
for h in history:
    cos_dict = h.get("cosine_similarities", {})
    if cos_dict:
        row = {"Round": h["round"]}
        row.update({f"Node {k}": v for k, v in cos_dict.items()})
        cos_rows.append(row)

if cos_rows:
    df_cos = pd.DataFrame(cos_rows).set_index("Round")
    # Colour: green = high similarity (trustworthy), red = low (suspicious)
    styled = df_cos.style.background_gradient(
        cmap="RdYlGn", vmin=-1.0, vmax=1.0, axis=None
    ).format("{:.3f}")
    st.dataframe(styled, use_container_width=True)
    st.caption(
        "Values near **+1.0** = aligned with global mean (trusted).  "
        "Values near **−1.0** = inverted update (Byzantine alert)."
    )
elif byz_enabled:
    st.info("No rounds with cosine data yet.")
else:
    st.info("Byzantine detection is **disabled**. Set `BYZANTINE_DETECTION=true`.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Local vs global accuracy
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Local Training Accuracy per Node")

local_rows = []
for h in history:
    la = h.get("local_accuracies", {})
    if la:
        row = {"Round": h["round"], "Global": h["accuracy"]}
        row.update({f"Node {k}": v for k, v in la.items() if v is not None})
        local_rows.append(row)

if local_rows:
    df_local = pd.DataFrame(local_rows).set_index("Round")
    st.line_chart(df_local, use_container_width=True)
    st.caption(
        "Gap between node local accuracy and global accuracy reveals non-IID drift."
    )
else:
    st.info("Local accuracy data will appear after the first round.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Compression ratio
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Gradient Compression (Top-K)")

comp_rows = [
    {"Round": h["round"], "Compression ratio": h["avg_compression"]}
    for h in history if h.get("avg_compression") is not None
]
if comp_rows:
    df_comp = pd.DataFrame(comp_rows).set_index("Round")
    st.bar_chart(df_comp, use_container_width=True)
    ratio = comp_rows[-1]["Compression ratio"]
    st.caption(
        f"Latest ratio: **{ratio:.1%}** of weights transmitted "
        f"({1 - ratio:.1%} bandwidth saved)."
    )
else:
    st.info("Compression is **disabled**. Set `ENABLE_COMPRESSION=true`.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Round history table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Round History")
if history:
    table_rows = []
    for h in history:
        table_rows.append({
            "Round":        h["round"],
            "FL cycle":     h.get("fl_cycle", "N/A"),
            "Accuracy":     f"{h['accuracy']:.2%}",
            "Nodes":        h["num_nodes"],
            "Strategy":     h.get("aggregation", "N/A"),
            "Flagged":      ", ".join(h.get("flagged_nodes", [])) or "N/A",
            "ε":            f"{h['avg_epsilon']:.3f}" if h.get("avg_epsilon") else "N/A",
            "Compression":  f"{h['avg_compression']:.1%}" if h.get("avg_compression") else "N/A",
            "Time":         pd.to_datetime(h["timestamp"], unit="s").strftime("%H:%M:%S"),
        })
    st.dataframe(pd.DataFrame(table_rows[::-1]), use_container_width=True, hide_index=True)
else:
    st.caption("No rounds completed yet.")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
time.sleep(REFRESH_INTERVAL)
st.rerun()
