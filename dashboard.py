import time
from collections import deque

import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="metrics_refresh")
except Exception:
    pass

st.set_page_config(
    page_title="OptiServe Telemetry",
    layout="wide",
)

METRICS_URL = "http://localhost:8000/metrics"

if "history" not in st.session_state:
    st.session_state.history = {
        "timestamps": deque(maxlen=120),
        "queue_depth": deque(maxlen=120),
        "decode_batch": deque(maxlen=120),
        "throughput": deque(maxlen=120),
        "last_poll_time": time.time(),
    }


def fetch_metrics():
    try:
        r = requests.get(METRICS_URL, timeout=1)
        return r.json()
    except Exception:
        return None


data = fetch_metrics()

st.title("OptiServe Telemetry")

if not data:
    st.error("Engine offline or metrics endpoint unreachable.")
    st.stop()

sched = data.get("scheduler", {})
alloc = data.get("allocator", {})
perf = data.get("performance", {})
gpu = data.get("gpu", {})
debug_requests = data.get("debug_requests", [])

h = st.session_state.history

now = time.time()
h["last_poll_time"] = now

h["timestamps"].append(time.strftime("%H:%M:%S"))
h["queue_depth"].append(sched.get("waiting", 0))
h["decode_batch"].append(perf.get("avg_decode_batch_size", 0))
h["throughput"].append(perf.get("tokens_per_second", 0))

k1, k2, k3 = st.columns(3)

k1.metric(
    "Throughput",
    f"{perf.get('tokens_per_second', 0):.1f} tok/s",
)

k2.metric(
    "Cache Hit",
    f"{perf.get('cache_hit_rate_pct', 0):.1f}%",
)

k3.metric(
    "TTFT",
    f"{perf.get('avg_ttft_ms', 0):.0f} ms",
)

k5, k6, k7 = st.columns(3)

k5.metric(
    "TPOT",
    f"{perf.get('avg_tpot_ms', 0):.0f} ms",
)

k6.metric(
    "Decode Batch",
    perf.get("avg_decode_batch_size", 0),
)

k7.metric(
    "GPU Alloc",
    f"{gpu.get('allocated_gb', 0):.2f} GB",
)

adm = data.get("admission", {})
cap = data.get("capacity", {})

k8, k9, k10 = st.columns(3)
k8.metric("Admitted", adm.get("total_admitted", 0))
k9.metric("Rejected", adm.get("total_rejected", 0))
k10.metric("Preemptions", adm.get("total_preemptions", 0))

st.progress(
    min(cap.get("free_kv_tokens", 0) / max(cap.get("total_kv_tokens", 1), 1), 1),
    text=f"KV token headroom: {cap.get('free_kv_tokens', 0)}/{cap.get('total_kv_tokens', 0)}",
)

left, right = st.columns([1, 2])

with left:
    st.subheader("Memory")

    util = alloc.get("utilization_pct", 0)

    st.progress(
        min(max(util / 100, 0), 1),
        text=f"KV Utilization {util:.1f}%",
    )

    st.markdown(
        f"""
**Blocks**

- Active: {alloc.get('active_blocks', 0)}
- Free: {alloc.get('free_blocks', 0)}
- Total: {alloc.get('total_blocks', 0)}

**Scheduler**

- Running: {sched.get('running', 0)}
- Waiting: {sched.get('waiting', 0)}
"""
    )

with right:
    st.subheader("System Dynamics")

    df = pd.DataFrame(
        {
            "Queue Depth": list(h["queue_depth"]),
            "Decode Batch": list(h["decode_batch"]),
            "Throughput": list(h["throughput"]),
        },
        index=list(h["timestamps"]),
    )

    st.line_chart(df)

    st.subheader("Debug Requests")

    if debug_requests:
        st.dataframe(
            pd.DataFrame(debug_requests),
            use_container_width=True,
        )
    else:
        st.info("No active requests.")