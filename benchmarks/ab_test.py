import asyncio
import aiohttp
import time
import json
import numpy as np

API_URL = "http://localhost:8000/v1/chat/completions"
ADMIN_URL = "http://localhost:8000/v1/admin/config"

PROMPT = "You are a historian. " + "Here is a massive text about Rome: " * 50 + " Explain it."
CONCURRENCY = 20

async def set_engine_config(session, config_dict):
    """Hits the admin endpoint to toggle features and clear the cache."""
    payload = {**config_dict, "CLEAR_CACHE": True}
    async with session.post(ADMIN_URL, json=payload) as resp:
        await resp.json()
    await asyncio.sleep(0.5)

async def fire_request(session, is_warmup=False):
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": PROMPT}],
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
    except Exception as e:
        return None
        
    if not internal_metrics: return None
        
    return {
        "total_time": time.time() - start_time,
        "tokens": tokens,
        "internal": internal_metrics
    }

async def run_scenario(session, name, config):
    print(f"Running Scenario: {name}...")
    await set_engine_config(session, config)
    
    global_start = time.time()
    
    # 1. Warmup
    warmup_res = await fire_request(session, is_warmup=True)
    if not warmup_res:
        return None
        
    # 2. Thundering Herd
    tasks = [fire_request(session) for _ in range(CONCURRENCY)]
    results = await asyncio.gather(*tasks)
    
    global_end = time.time()
    
    valid = [r for r in results if r is not None]
    valid.append(warmup_res)
    
    total_time = global_end - global_start
    total_tokens = sum(r["tokens"] for r in valid)
    
    # Safely pull the true TTFT (Requires the routes.py update from the previous step!)
    ttfts = [r["internal"].get("ttft", 0.01) for r in valid]
    
    # --- DYNAMIC CACHE HIT PERCENTAGE ---
    prefix_depths = [r["internal"].get("prefix_depth", 0) for r in valid]
    max_depth = max(prefix_depths) if prefix_depths else 0
    
    if max_depth > 0:
        total_possible_depth = max_depth * len(valid)
        actual_depth = sum(prefix_depths)
        cache_hit_pct = f"{(actual_depth / total_possible_depth) * 100:.1f}%"
    else:
        cache_hit_pct = "0.0%"
    # ------------------------------------
    
    return {
        "Name": name,
        "TTFT_P99": f"{np.percentile(ttfts, 99):.2f}s",
        "Throughput": f"{total_tokens / total_time:.2f} t/s",
        "Cache_Hit": cache_hit_pct
    }

async def main():
    configs = [
        {
            "name": "Naive (All OFF)",
            "config": {
                "ENABLE_CONTINUOUS_BATCHING": False,
                "ENABLE_CHUNKED_PREFILL": False,
                "ENABLE_PREFIX_CACHE": False
            }
        },
        {
            "name": "Batching ON",
            "config": {
                "ENABLE_CONTINUOUS_BATCHING": True,
                "ENABLE_CHUNKED_PREFILL": False,
                "ENABLE_PREFIX_CACHE": False
            }
        },
        {
            "name": "OptiServe Full (All ON)",
            "config": {
                "ENABLE_CONTINUOUS_BATCHING": True,
                "ENABLE_CHUNKED_PREFILL": True,
                "ENABLE_PREFIX_CACHE": True
            }
        }
    ]

    results = []
    async with aiohttp.ClientSession() as session:
        for c in configs:
            res = await run_scenario(session, c["name"], c["config"])
            if res:
                results.append(res)
    
    print("\n" + "="*65)
    print(f"{'Configuration':<25} | {'P99 TTFT':<10} | {'Throughput':<12} | {'Cache Hit %'}")
    print("-" * 65)
    for r in results:
        print(f"{r['Name']:<25} | {r['TTFT_P99']:<10} | {r['Throughput']:<12} | {r['Cache_Hit']}")
    print("="*65 + "\n")

if __name__ == "__main__":
    asyncio.run(main())