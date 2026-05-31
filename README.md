# LLM-Inference-Runtime

An LLM inference engine built from scratch in Python, inspired by vLLM. Built to understand how modern inference systems work under the hood — paged KV caching, continuous batching, prefix caching, and GPU memory management.

> **Scope:** Supports LLaMA-style architectures only. 


---

## Performance

| Metric                                        | Value                                       |
| --------------------------------------------- | ------------------------------------------- |
| Peak throughput                               | 414 tok/s                                   |
| Sustained throughput (50 concurrent requests) | 250–280 tok/s                               |
| Sustained request rate                        | ~4.8 requests/s                             |
| Stability test                                | 10,000+ requests, no throughput degradation |
| Hardware                                      | RTX 3050 6GB (laptop)                       |

Benchmark workload: ShareGPT short prompts, mixed output lengths (20–60 tokens), 50 concurrent clients.

---

## Features

### Inference Engine

* Async inference engine with background processing loop
* OpenAI-compatible `/v1/chat/completions` API
* Streaming token generation via Server-Sent Events (SSE)
* Concurrent request handling and lifecycle tracking

### Scheduler

* Continuous batching
* Chunked prefill
* Dynamic batch admission
* Token-budget-aware scheduling
* Separate prefill and decode scheduling
* Request waiting queue management

### KV Cache

* Paged KV cache with fixed-size blocks
* Reference-counted block allocator
* Block sharing across requests
* Dynamic block allocation and reclamation
* GPU memory utilization tracking
* Demand-driven memory recovery

### Prefix Caching

* Radix-tree prefix cache
* Prefix matching and reuse
* Cached prompt skipping
* Shared KV block reuse
* Cache hit accounting
* LRU eviction
* Reference-safe block reclamation

### FlashInfer Integration

* FlashInfer paged attention
* FlashInfer prefill kernels
* FlashInfer decode kernels
* Monkey-patched LLaMA attention layers
* FlashInfer KV page management
* FlashInfer batch planning

### Metrics & Observability

Per-request metrics:

* Queue latency
* Time to First Token (TTFT)
* Time per Output Token (TPOT)
* Request throughput
* Prompt and generation token counts

System metrics:

* Tokens/sec
* Cache hit rate
* Cache pressure
* KV utilization
* Decode batch size
* Queue depth
* Preemption count
* GPU memory usage

### Dashboard

* Live Streamlit telemetry dashboard
* Throughput monitoring
* TTFT / TPOT tracking
* Cache hit rate visualization
* Cache pressure monitoring
* Decode batch visualization
* GPU memory monitoring
* Active request inspection
* Time-series performance charts

---

## Installation

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/harsh-vsingh/LLM-Inference-Runtime
cd LLM-Inference-Runtime
uv sync
```

---

## Usage

### Start the inference server

```bash
uv run python -m src.api.server
```

### Start the telemetry dashboard

```bash
uv run streamlit run dashboard.py
```

The server exposes an OpenAI-compatible API at:

```text
http://localhost:8000/v1
```

Example client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(
            chunk.choices[0].delta.content,
            end="",
            flush=True,
        )
```

---

## Architecture Overview

```text
Request
  └── FastAPI Server
        └── AsyncInferenceEngine
              ├── Scheduler
              ├── Sequence Manager
              ├── BlockAllocator
              ├── RadixCache
              └── FlashInfer Attention
```

### Core Components

**Scheduler**

* Handles admission control and continuous batching
* Separates prefill and decode execution
* Enforces token and sequence budgets

**BlockAllocator**

* Manages paged GPU KV memory
* Tracks ownership using reference counts
* Supports block sharing and reclamation

**RadixCache**

* Stores reusable prompt prefixes
* Reuses existing KV blocks across requests
* Uses LRU eviction under memory pressure

**FlashInfer**

* Executes paged attention for prefill and decode
* Reads KV pages directly from the allocator
* Accelerates batched generation

---

## Benchmarks

A benchmarking suite is included under `benchmarks/`.

Included tests:

* Burst tests
* Decode scaling tests
* Fairness tests
* Mixed workload tests
* Stress tests
* Long-running stability tests
* Cache reuse benchmarks
* Randomized workload tests

Example:

```bash
uv run python benchmarks/stability_test.py
```

---

## Dashboard

Live telemetry includes:

* Throughput
* TTFT
* TPOT
* Cache hit rate
* Cache pressure
* Decode batch size
* GPU memory utilization
* Active request inspection