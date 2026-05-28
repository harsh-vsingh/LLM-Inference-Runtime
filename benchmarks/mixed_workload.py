import asyncio
import aiohttp
import time
import json
import numpy as np

API_URL = "http://localhost:8000/v1/chat/completions"

WORKLOADS = [
    # 1. Heavy Prefill (Tries to hog the GPU)
    {"type": "Heavy", "prompt": "Summarize this historical text: " + "Rome fell because of economic instability. " * 300, "max_tokens": 20},
    # 2. Heavy Decode (Tries to hog the batch generation)
    {"type": "Long", "prompt": "Write a detailed, 5-paragraph essay on the history of AI.", "max_tokens": 300},
    # 3. Light Chat (Should return instantly if Chunked Prefill is fair)
    {"type": "Light", "prompt": "What is 2+2?", "max_tokens": 10}
] * 10 # 30 total mixed requests

async def fire_mixed_request(session, workload):
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": workload["prompt"]}],
        "stream": True,
        "max_tokens": workload["max_tokens"]
    }
    
    internal_metrics = {}
    tokens = 0
    
    try:
        async with session.post(API_URL, json=payload) as response:
            async for line in response.content:
                if line and b'"object": "chat.completion.chunk"' in line:
                    data = json.loads(line.decode("utf-8")[6:].strip())
                    if data["choices"][0]["delta"].get("content"):
                        tokens += 1
                    if "usage" in data and "optiserve_metrics" in data["usage"]:
                        internal_metrics = data["usage"]["optiserve_metrics"]
    except Exception:
        return None
        
    if not internal_metrics: return None
        
    return {
        "type": workload["type"],
        "tokens": tokens,
        "ttft": internal_metrics.get("ttft", 0.0),
        "tpot": internal_metrics.get("decode_time", 0.0) / max(tokens, 1)
    }

async def run_mixed():
    print("=== OPTISERVE MIXED WORKLOAD (Fairness Test) ===")
    print("Firing heavy documents, long essays, and light chats simultaneously...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [fire_mixed_request(session, w) for w in WORKLOADS]
        results = await asyncio.gather(*tasks)
        
    valid = [r for r in results if r is not None]
    
    # Group results by type
    metrics_by_type = {"Heavy": [], "Long": [], "Light": []}
    for r in valid:
        metrics_by_type[r["type"]].append(r)
        
    print("\n" + "="*60)
    print(f"{'Workload Type':<15} | {'Mean TTFT':<12} | {'Mean TPOT':<12} | {'Count'}")
    print("-" * 60)
    
    for t in ["Heavy", "Long", "Light"]:
        group = metrics_by_type[t]
        if not group: continue
        
        avg_ttft = np.mean([r["ttft"] for r in group])
        avg_tpot = np.mean([r["tpot"] for r in group]) * 1000
        print(f"{t:<15} | {avg_ttft:<10.2f}s | {avg_tpot:<9.2f}ms | {len(group)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(run_mixed())