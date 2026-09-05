import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp

API_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"  # was 8080 - wrong port, fetch_metrics was silently failing every call

PROMPT_POOL_SIZE = 500
TARGET_RATE_RPS = 20          # sustained requests/sec - tune this to find the engine's breaking point
RUN_DURATION_SECONDS = 120

# Bimodal generation length: most requests are short, a meaningful
# minority are long. Long+short competing for the same KV budget
# concurrently is what actually creates preemption pressure - an
# all-short workload just cycles through quickly with low peak KV usage,
# never forcing eviction/preemption decisions.
SHORT_MAX_TOKENS_RANGE = (20, 60)
LONG_MAX_TOKENS_RANGE = (200, 500)
LONG_REQUEST_FRACTION = 0.15

# Bimodal prompt length - same reasoning, applied to the prompt side.
SHORT_PROMPT_MAX_WORDS = 30
LONG_PROMPT_MAX_WORDS = 300
LONG_PROMPT_FRACTION = 0.15


@dataclass
class Stats:
    completed: int = 0
    rejected_retryable: int = 0
    rejected_permanent: int = 0
    connection_error: int = 0
    other_errors: int = 0
    total_ttft: float = 0.0
    total_tpot: float = 0.0
    total_generated_tokens: int = 0
    fired: int = 0
    # Rolling window of recent (ttft, tpot) samples, mirroring the
    # engine's own EngineMetrics._recent (maxlen=128) - see the note in
    # reporter() for why this matters. cumulative total_ttft/total_tpot
    # above are kept too (lifetime averages), but are NOT directly
    # comparable to the engine's avg_ttft_ms/avg_tpot_ms.
    recent_samples: deque = field(default_factory=lambda: deque(maxlen=128))

    def outcome_summary(self) -> str:
        return (
            f"fired={self.fired} ok={self.completed} "
            f"rejected(retry)={self.rejected_retryable} rejected(perm)={self.rejected_permanent} "
            f"conn_err={self.connection_errors} other_err={self.other_errors}"
        )

    def recent_avg_ttft_ms(self) -> float:
        if not self.recent_samples:
            return 0.0
        return sum(s[0] for s in self.recent_samples) / len(self.recent_samples) * 1000

    def recent_avg_tpot_ms(self) -> float:
        if not self.recent_samples:
            return 0.0
        return sum(s[1] for s in self.recent_samples) / len(self.recent_samples) * 1000


def load_prompts(file_path: str, count: int = PROMPT_POOL_SIZE):
    with open(file_path, "r") as f:
        data = json.load(f)

    short_prompts, long_prompts = [], []

    for entry in data:
        conversations = entry.get("conversations")
        if not conversations:
            continue

        content = conversations[0]["value"]
        word_count = len(content.split())

        if word_count <= SHORT_PROMPT_MAX_WORDS and len(short_prompts) < count:
            short_prompts.append(content)
        elif word_count <= LONG_PROMPT_MAX_WORDS and len(long_prompts) < count // 4:
            long_prompts.append(content)

        if len(short_prompts) >= count and len(long_prompts) >= count // 4:
            break

    print(f"Loaded {len(short_prompts)} short prompts, {len(long_prompts)} long prompts")
    return short_prompts, long_prompts


def build_payload(short_prompts, long_prompts) -> dict:
    is_long_prompt = random.random() < LONG_PROMPT_FRACTION
    prompt = (
        random.choice(long_prompts) if is_long_prompt and long_prompts
        else random.choice(short_prompts)
    )

    is_long_gen = random.random() < LONG_REQUEST_FRACTION
    max_tokens = (
        random.randint(*LONG_MAX_TOKENS_RANGE) if is_long_gen
        else random.randint(*SHORT_MAX_TOKENS_RANGE)
    )

    return {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }


async def fire_request(session: aiohttp.ClientSession, payload: dict, stats: Stats):
    stats.fired += 1
    start = time.time()
    first_token_time = None
    generated_tokens = 0

    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 503:
                stats.rejected_retryable += 1
                return
            if response.status == 400:
                stats.rejected_permanent += 1
                return
            if response.status >= 400:
                stats.other_errors += 1
                return

            async for raw_chunk in response.content:
                chunk = raw_chunk.decode("utf-8", errors="ignore").strip()
                if not chunk:
                    continue
                chunk = chunk.removeprefix("data: ")
                if chunk == "[DONE]":
                    break

                try:
                    data = json.loads(chunk)
                    delta = data["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        generated_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        end = time.time()
        ttft = (first_token_time - start) if first_token_time else 0.0
        generation_time = (end - first_token_time) if first_token_time else 0.0
        tpot = generation_time / max(generated_tokens - 1, 1)

        stats.completed += 1
        stats.total_ttft += ttft
        stats.total_tpot += tpot
        stats.total_generated_tokens += generated_tokens
        stats.recent_samples.append((ttft, tpot))

    except aiohttp.ClientError:
        stats.connection_errors += 1
    except Exception as e:
        stats.other_errors += 1
        print(f"Unexpected error: {e}")


async def fetch_engine_metrics(session: aiohttp.ClientSession) -> dict:
    try:
        async with session.get(METRICS_URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status != 200:
                print(f"metrics fetch returned status {response.status}")
                return {}
            return await response.json()
    except Exception as e:
        # Previously a bare `except Exception: return {}` - silently
        # swallowed the actual failure reason. Under sustained load this
        # was almost certainly timing out or contending for connections
        # with the hundreds of in-flight streaming POSTs on the same
        # session, but there was no way to tell that from a bare {}.
        print(f"metrics fetch failed: {type(e).__name__}: {e}")
        return {}


async def reporter(session: aiohttp.ClientSession, stats: Stats, start_time: float):
    while True:
        await asyncio.sleep(5)
        elapsed = time.time() - start_time

        metrics = await fetch_engine_metrics(session)
        admission = metrics.get("admission", {})
        capacity = metrics.get("capacity", {})
        perf = metrics.get("performance", {})

        # Two different TTFT/TPOT numbers are reported here, and they are
        # NOT expected to match - see the comment on Stats.recent_samples.
        # lifetime_ttft/lifetime_tpot average over EVERY request completed
        # since this script started; under sustained/growing load this
        # climbs over time as the system saturates and later requests
        # queue longer. recent_ttft/recent_tpot is a rolling window
        # (last 128 completions), directly comparable to the engine's own
        # avg_ttft_ms/avg_tpot_ms (also a 128-sample rolling window) - if
        # these two "recent" numbers disagree significantly, THAT indicates
        # a real problem (e.g. client-side queuing/contention distinct
        # from the engine's own processing time). The lifetime numbers
        # diverging from either is expected under sustained load, not a bug.
        lifetime_ttft = stats.total_ttft / max(stats.completed, 1)
        lifetime_tpot = stats.total_tpot / max(stats.completed, 1)
        recent_ttft = stats.recent_avg_ttft_ms()
        recent_tpot = stats.recent_avg_tpot_ms()
        avg_tps = stats.total_generated_tokens / max(elapsed, 1e-6)

        print(
            f"[{elapsed:6.1f}s] {stats.outcome_summary()} | "
            f"client_tps={avg_tps:.1f} | "
            f"ttft(lifetime={lifetime_ttft*1000:.0f}ms recent={recent_ttft:.0f}ms) "
            f"tpot(lifetime={lifetime_tpot*1000:.0f}ms recent={recent_tpot:.0f}ms) || "
            f"engine: waiting={metrics.get('scheduler', {}).get('waiting', '?')} "
            f"free_kv_tokens={capacity.get('free_kv_tokens', '?')} "
            f"admitted={admission.get('total_admitted', '?')} "
            f"rejected={admission.get('total_rejected', '?')} "
            f"preemptions={admission.get('total_preemptions', '?')} "
            f"engine_tps={perf.get('tokens_per_second', '?')} "
            f"engine_ttft={perf.get('avg_ttft_ms', '?')} "
            f"engine_tpot={perf.get('avg_tpot_ms', '?')}"
        )


async def load_generator(session: aiohttp.ClientSession, short_prompts, long_prompts, stats: Stats):
    """
    Fixed-rate open-loop arrival: fires a new request every 1/TARGET_RATE_RPS
    seconds regardless of whether earlier requests have completed. This is
    what creates real backpressure - a concurrency-pool model
    (N-in-flight, fire next on completion) self-throttles to the engine's
    own service rate and can never actually exceed its capacity, so it
    would never trigger QUEUE_FULL/preemption/proactive eviction. A fixed
    rate keeps arriving even if the engine falls behind, which is the only
    way to observe admission control actually doing its job.
    """
    interval = 1.0 / TARGET_RATE_RPS
    next_fire_time = time.time()

    while True:
        payload = build_payload(short_prompts, long_prompts)
        asyncio.create_task(fire_request(session, payload, stats))

        next_fire_time += interval
        sleep_time = next_fire_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        # If sleep_time <= 0, the client is falling behind the target
        # rate - intentionally NOT catching up by bursting, just fires
        # the next request immediately and continues, which still
        # sustains pressure without an artificial burst spike.


async def main():
    short_prompts, long_prompts = load_prompts("ShareGPT_V3_unfiltered_cleaned_split.json")

    stats = Stats()
    start_time = time.time()

    # Separate session for metrics polling, distinct from the load
    # session used for the hundreds of in-flight streaming POSTs. Sharing
    # one session meant the metrics GET was competing for connections
    # against the load itself, which was almost certainly why it was
    # timing out and silently returning {} under sustained pressure.
    async with aiohttp.ClientSession() as load_session, aiohttp.ClientSession() as metrics_session:
        gen_task = asyncio.create_task(
            load_generator(load_session, short_prompts, long_prompts, stats)
        )
        report_task = asyncio.create_task(reporter(metrics_session, stats, start_time))

        await asyncio.sleep(RUN_DURATION_SECONDS)

        gen_task.cancel()
        report_task.cancel()
        for t in (gen_task, report_task):
            try:
                await t
            except asyncio.CancelledError:
                pass

        print("\n=== FINAL SUMMARY ===")
        elapsed = time.time() - start_time
        print(stats.outcome_summary())
        print(f"duration={elapsed:.1f}s")
        print(f"client_tps={stats.total_generated_tokens / max(elapsed, 1e-6):.1f}")
        print(f"avg_ttft={stats.total_ttft / max(stats.completed, 1) * 1000:.0f}ms")
        print(f"avg_tpot={stats.total_tpot / max(stats.completed, 1) * 1000:.0f}ms")

        final_metrics = await fetch_engine_metrics(metrics_session)
        print(f"final engine metrics: {json.dumps(final_metrics, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())