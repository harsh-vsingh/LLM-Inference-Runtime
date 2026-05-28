import asyncio
import aiohttp
import time
import json
import numpy as np

API_URL = "http://localhost:8000/v1/chat/completions"
CONCURRENCY = 50  # Massive spike

async def fire_request(session, req_id):
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": f"Write a 3 sentence poem about a robot named {req_id}."}],
        "stream": True,
        "max_tokens": 50
    }
    
    start_time = time.time()
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
        "tokens": tokens,
        "internal": internal_metrics
    }

async def run_burst():
    print(f"=== OPTISERVE BURST TEST ({CONCURRENCY} Concurrent Requests) ===")
    print("Firing massive traffic spike...")
    global_start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fire_request(session, i) for i in range(CONCURRENCY)]
        results = await asyncio.gather(*tasks)
        
    global_end = time.time()
    valid = [r for r in results if r is not None]
    
    total_time = global_end - global_start
    total_tokens = sum(r["tokens"] for r in valid)
    q_latencies = [r["internal"]["queue_latency"] for r in valid]
    tpots = [r["internal"]["decode_time"] / max(r["tokens"], 1) for r in valid]
    avg_batches = [r["internal"].get("avg_decode_batch_size", 1.0) for r in valid]
    
    print("\n" + "="*50)
    print(f"Success Rate   : {len(valid)}/{CONCURRENCY} survived without OOM")
    print(f"Total Time     : {total_time:.2f}s")
    print(f"Throughput     : {total_tokens / total_time:.2f} tokens/s")
    print("-" * 50)
    print(f"Peak Batch Size: {max(avg_batches):.1f} sequences simultaneously on GPU")
    print(f"Queue Wait P99 : {np.percentile(q_latencies, 99):.2f}s (Max wait time)")
    print(f"TPOT Mean      : {np.mean(tpots)*1000:.2f}ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_burst())