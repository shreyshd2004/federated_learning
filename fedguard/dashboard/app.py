"""
FedGuard — Advanced Streamlit Dashboard

Panels
------
1. Header metrics   — round, accuracy, nodes, privacy budget
2. Node status      — per-node participation + Byzantine flags
3. Accuracy chart   — global accuracy over rounds
4. Cosine similarity heatmap — Byzantine detection per round
5. Privacy budget   — cumulative ε per round (DP-SGD)
6. Local vs global accuracy — per-node training quality
7. Compression ratio — bandwidth savings per round
8. Round details table — full round history
9. Config sidebar   — live system configuration
10. Attack Lab      — DLG gradient leakage attack + DP defence demo

Auto-refreshes every 5 seconds.
"""
import base64
import os
import time

import pandas as pd
import requests
import streamlit as st

SERVER_URL       = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
ATTACK_URL       = os.environ.get("ATTACK_URL", "http://attack:8888").rstrip("/")
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
# Sidebar — config
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
st.title("🛡️ FedGuard — Federated Learning Monitor")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Round", current_round)
c2.metric("FL cycle", accepting_cycle)
c3.metric("Latest Accuracy",  f"{latest_acc:.2%}"  if latest_acc  is not None else "—")
c4.metric("Best Accuracy",    f"{best_acc:.2%}"    if best_acc    is not None else "—")
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
st.subheader("Byzantine Detection — Cosine Similarity per Round")

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
            "FL cycle":     h.get("fl_cycle", "—"),
            "Accuracy":     f"{h['accuracy']:.2%}",
            "Nodes":        h["num_nodes"],
            "Strategy":     h.get("aggregation", "—"),
            "Flagged":      ", ".join(h.get("flagged_nodes", [])) or "—",
            "ε":            f"{h['avg_epsilon']:.3f}" if h.get("avg_epsilon") else "—",
            "Compression":  f"{h['avg_compression']:.1%}" if h.get("avg_compression") else "—",
            "Time":         pd.to_datetime(h["timestamp"], unit="s").strftime("%H:%M:%S"),
        })
    st.dataframe(pd.DataFrame(table_rows[::-1]), use_container_width=True, hide_index=True)
else:
    st.caption("No rounds completed yet.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Attack Lab — DLG Gradient Leakage
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Attack Lab — Deep Leakage from Gradients (DLG)")

st.markdown(
    """
    **Threat:** An honest-but-curious server can reconstruct a node's *private training images*
    from the gradient upload alone — without ever seeing the raw data.

    **Defence:** DP-SGD adds calibrated Gaussian noise before upload, degrading reconstruction.

    > *Zhu et al., "Deep Leakage from Gradients" NeurIPS 2019*
    """
)

# Check if attack server is reachable
attack_online = False
try:
    resp = requests.get(f"{ATTACK_URL}/health", timeout=3)
    attack_online = resp.status_code == 200
except Exception:
    pass

if not attack_online:
    st.warning(
        "Attack server is offline. Add `attack` service to docker-compose and rebuild.\n\n"
        "Set `ATTACK_URL=http://attack:8888` in the dashboard environment."
    )
else:
    atk_col1, atk_col2 = st.columns([1, 2])

    with atk_col1:
        st.markdown("**Attack parameters**")
        img_idx   = st.number_input("MNIST test image index", 0, 9999, 0, 1)
        iters     = st.slider("L-BFGS iterations", 50, 500, 300, 50)
        use_idlg  = st.checkbox("Use iDLG (analytical label extraction)", value=True)
        tv_weight = st.select_slider(
            "TV regularisation weight",
            options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
        )

        st.markdown("---")
        st.markdown("**Run single noise level**")
        noise_mult = st.select_slider(
            "DP noise multiplier (σ)",
            options=[0.0, 0.1, 0.3, 0.6, 1.1, 2.0],
            value=0.0,
            format_func=lambda x: {
                0.0: "0.0 (No DP, ε=∞)",
                0.1: "0.1 (ε≈50)",
                0.3: "0.3 (ε≈10)",
                0.6: "0.6 (ε≈3)",
                1.1: "1.1 (ε≈1)",
                2.0: "2.0 (ε<1)",
            }.get(x, str(x)),
        )

        run_single = st.button("Run Attack", type="primary")
        run_compare = st.button("Run Full Comparison (3 noise levels)")

    with atk_col2:
        def _show_image(b64: str, caption: str):
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, caption=caption, width=140)

        # ── Single attack ────────────────────────────────────────────────────
        if run_single:
            with st.spinner("Running DLG reconstruction…"):
                try:
                    r = requests.post(
                        f"{ATTACK_URL}/run",
                        json={
                            "image_index":     img_idx,
                            "iterations":      iters,
                            "use_idlg":        use_idlg,
                            "tv_weight":       tv_weight,
                            "noise_multiplier": noise_mult,
                        },
                        timeout=300,
                    )
                    r.raise_for_status()
                    res = r.json()

                    m = res["metrics"]
                    p = res["params"]

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("PSNR", f"{m['psnr_db']} dB",
                               help="Higher = better reconstruction")
                    mc2.metric("MSE", f"{m['mse']:.6f}",
                               help="Lower = better reconstruction")
                    mc3.metric("Label recovered",
                               f"{p['predicted_label']} ({'✓' if p['label_correct'] else '✗'})",
                               help="True label: " + str(p["true_label"]))

                    ic1, ic2, ic3 = st.columns(3)
                    with ic1:
                        _show_image(res["images"]["original"], "Original (private)")
                    with ic2:
                        _show_image(res["images"]["reconstructed"],
                                    f"Reconstructed (σ={noise_mult})")
                    with ic3:
                        conv_bytes = base64.b64decode(res["images"]["convergence"])
                        st.image(conv_bytes, caption="Convergence curve", width=280)

                except Exception as exc:
                    st.error(f"Attack failed: {exc}")

        # ── Comparison (3 noise levels) ──────────────────────────────────────
        if run_compare:
            with st.spinner("Running 3-level DP comparison… (~3× longer)"):
                try:
                    r = requests.post(
                        f"{ATTACK_URL}/run_comparison",
                        params={"image_index": img_idx, "iterations": iters},
                        timeout=900,
                    )
                    r.raise_for_status()
                    res = r.json()

                    st.markdown("**True label:** `{}`".format(res["true_label"]))

                    # Header row
                    hcols = st.columns(len(res["comparisons"]) + 1)
                    with hcols[0]:
                        _show_image(res["original_b64"], "Original")

                    for col, comp in zip(hcols[1:], res["comparisons"]):
                        with col:
                            _show_image(comp["reconstructed_b64"], comp["label"])
                            st.caption(
                                f"PSNR: {comp['psnr_db']} dB\n"
                                f"MSE: {comp['mse']:.5f}\n"
                                f"Label: {comp['predicted_label']} "
                                f"({'✓' if comp['label_correct'] else '✗'})"
                            )

                    # Metrics table
                    df_comp = pd.DataFrame([
                        {
                            "Privacy level": c["label"],
                            "σ (noise)":     c["noise_multiplier"],
                            "PSNR (dB)":     c["psnr_db"],
                            "MSE":           c["mse"],
                            "Label correct": "Yes" if c["label_correct"] else "No",
                        }
                        for c in res["comparisons"]
                    ])
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    st.caption(
                        "As DP noise increases (σ↑, ε↓), PSNR drops and MSE rises — "
                        "the server can no longer reconstruct the private training image."
                    )

                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

        # ── Cached results ───────────────────────────────────────────────────
        if not run_single and not run_compare:
            try:
                cached = requests.get(f"{ATTACK_URL}/results", timeout=3).json()
                if cached.get("status") != "no_results":
                    st.info("Showing cached results from last attack run.")
                    if "original_b64" in cached:
                        # Comparison result
                        hcols = st.columns(len(cached["comparisons"]) + 1)
                        with hcols[0]:
                            _show_image(cached["original_b64"], "Original")
                        for col, comp in zip(hcols[1:], cached["comparisons"]):
                            with col:
                                _show_image(comp["reconstructed_b64"], comp["label"])
                                st.caption(f"PSNR: {comp['psnr_db']} dB")
                    elif "images" in cached:
                        # Single result
                        ic1, ic2 = st.columns(2)
                        with ic1:
                            _show_image(cached["images"]["original"], "Original")
                        with ic2:
                            _show_image(cached["images"]["reconstructed"], "Reconstructed")
            except Exception:
                st.caption("Click **Run Attack** or **Run Full Comparison** to start.")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
time.sleep(REFRESH_INTERVAL)
st.rerun()
