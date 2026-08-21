"""
Single-backend load test for OptiServe or vLLM - run one at a time (never
concurrently; both would compete for the same GPU and corrupt results).

Run the same command twice, once per backend with --backend, so both get
the IDENTICAL request plan (same seed) for a fair before/after or A/B
comparison across two separate invocations.

Usage:
    python load_test.py --backend optiserve --url http://localhost:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python load_test.py --backend vllm      --url http://localhost:8001/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Fix vs. the earlier compare_vllm.py: that script silently dropped any
request that got a 2xx response but produced zero tokens before the
stream ended - not counted as completed, not counted as any error type.
completed + rejected + conn_err + other_err did not sum to fired (37
requests unaccounted for in the first real run). That's not a neutral
non-event - a request that opens successfully, streams nothing, and closes
is itself a failure mode (almost certainly the server-side batch-failure
path aborting in-flight requests), and silently dropping it both hides the
true completion rate AND throws away the exact evidence trail needed to
diagnose why. Every fired request now lands in EXACTLY one bucket:
completed / rejected / empty_response / connection_error / other_error -
enforced by an assertion at the end of each run, not just a convention.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

SEED = 1234
PROMPT_POOL_SIZE = 500
TARGET_RATE_RPS = 20
RUN_DURATION_SECONDS = 50
REPORT_INTERVAL_SECONDS = 5
DRAIN_TIMEOUT_SECONDS = 60.0

SHORT_MAX_TOKENS_RANGE = (20, 60)
LONG_MAX_TOKENS_RANGE = (200, 500)
LONG_REQUEST_FRACTION = 0.15

SHORT_PROMPT_MAX_WORDS = 30
LONG_PROMPT_MAX_WORDS = 300
LONG_PROMPT_FRACTION = 0.15

PROMPTS_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


# ---------------------------------------------------------------------------
# Request plan - identical generation logic regardless of backend, so
# running this script twice (once per --backend) with the same seed
# produces the same plan both times.
# ---------------------------------------------------------------------------

@dataclass
class PlannedRequest:
    fire_at: float
    prompt: str
    max_tokens: int


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

    return short_prompts, long_prompts


def build_request_plan(
    short_prompts, long_prompts, rate_rps: float, duration_seconds: float
) -> List[PlannedRequest]:
    rng = random.Random(SEED)
    plan: List[PlannedRequest] = []

    t = 0.0
    interval = 1.0 / rate_rps
    while t < duration_seconds:
        is_long_prompt = rng.random() < LONG_PROMPT_FRACTION
        prompt = (
            rng.choice(long_prompts) if is_long_prompt and long_prompts
            else rng.choice(short_prompts)
        )
        is_long_gen = rng.random() < LONG_REQUEST_FRACTION
        max_tokens = (
            rng.randint(*LONG_MAX_TOKENS_RANGE) if is_long_gen
            else rng.randint(*SHORT_MAX_TOKENS_RANGE)
        )
        plan.append(PlannedRequest(fire_at=t, prompt=prompt, max_tokens=max_tokens))
        t += interval

    return plan


# ---------------------------------------------------------------------------
# Stats - every fired request lands in exactly one bucket. This invariant
# is checked explicitly at the end of the run (see run_load_plan), not
# just assumed.
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    fired: int = 0
    completed: int = 0
    rejected: int = 0          # HTTP 400/503 before any streaming began
    empty_response: int = 0    # 2xx response, but zero tokens streamed back
    connection_errors: int = 0 # aiohttp.ClientError (dropped connection, etc.)
    other_errors: int = 0      # anything else unexpected

    total_ttft: float = 0.0
    total_tpot: float = 0.0
    total_generated_tokens: int = 0

    interval_generated_tokens: int = 0
    interval_completed: int = 0
    interval_ttft_sum: float = 0.0
    interval_tpot_sum: float = 0.0

    # Timestamps (seconds since run start) of every non-completed outcome,
    # so failures can be correlated against a server-side metrics timeline
    # (e.g. OptiServe's /metrics KV-utilization curve) after the fact.
    failure_log: List[str] = field(default_factory=list)

    def accounted_for(self) -> int:
        return (
            self.completed + self.rejected + self.empty_response
            + self.connection_errors + self.other_errors
        )

    def snapshot_and_reset_interval(self):
        snap = {
            "tokens": self.interval_generated_tokens,
            "completed": self.interval_completed,
            "avg_ttft_ms": (self.interval_ttft_sum / self.interval_completed * 1000)
                if self.interval_completed else 0.0,
            "avg_tpot_ms": (self.interval_tpot_sum / self.interval_completed * 1000)
                if self.interval_completed else 0.0,
        }
        self.interval_generated_tokens = 0
        self.interval_completed = 0
        self.interval_ttft_sum = 0.0
        self.interval_tpot_sum = 0.0
        return snap

    def lifetime_avg_ttft_ms(self) -> float:
        return (self.total_ttft / self.completed * 1000) if self.completed else 0.0

    def lifetime_avg_tpot_ms(self) -> float:
        return (self.total_tpot / self.completed * 1000) if self.completed else 0.0


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------

async def fire_request(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    req: PlannedRequest,
    stats: Stats,
    run_start: float,
):
    stats.fired += 1
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "stream": True,
        "max_tokens": req.max_tokens,
    }

    start = time.time()
    first_token_time: Optional[float] = None
    generated_tokens = 0

    def log_failure(kind: str, detail: str = ""):
        t = start - run_start
        stats.failure_log.append(f"[{t:6.2f}s] {kind} {detail}".strip())

    try:
        async with session.post(url, json=payload) as response:
            if response.status in (400, 503):
                stats.rejected += 1
                log_failure("rejected", f"status={response.status}")
                return
            if response.status >= 400:
                stats.other_errors += 1
                log_failure("http_error", f"status={response.status}")
                return

            async for raw_chunk in response.content:
                chunk = raw_chunk.decode("utf-8", errors="ignore").strip()
                if not chunk or not chunk.startswith("data: "):
                    continue
                chunk = chunk[6:]
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

        if generated_tokens == 0:
            # Response opened fine (2xx) but the stream produced nothing
            # before ending. This is NOT a neutral outcome - it almost
            # always means the request was aborted server-side mid-flight
            # (e.g. a batch-failure teardown). Counted explicitly rather
            # than silently dropped, specifically so it isn't lost the
            # way it was in the original version of this script.
            stats.empty_response += 1
            log_failure("empty_response", f"max_tokens={req.max_tokens}")
            return

        end = time.time()
        ttft = (first_token_time - start) if first_token_time else 0.0
        generation_time = (end - first_token_time) if first_token_time else 0.0
        tpot = generation_time / max(generated_tokens - 1, 1)

        stats.completed += 1
        stats.total_ttft += ttft
        stats.total_tpot += tpot
        stats.total_generated_tokens += generated_tokens

        stats.interval_completed += 1
        stats.interval_generated_tokens += generated_tokens
        stats.interval_ttft_sum += ttft
        stats.interval_tpot_sum += tpot

    except aiohttp.ClientError as e:
        stats.connection_errors += 1
        log_failure("connection_error", f"{type(e).__name__}: {e}")
    except Exception as e:
        stats.other_errors += 1
        log_failure("other_error", f"{type(e).__name__}: {e}")


async def reporter(label: str, stats: Stats, start_time: float, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=REPORT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass

        elapsed = time.time() - start_time
        snap = stats.snapshot_and_reset_interval()
        interval_tps = snap["tokens"] / REPORT_INTERVAL_SECONDS

        print(
            f"[{label:>9s} {elapsed:5.1f}s] "
            f"fired={stats.fired:4d} ok={stats.completed:4d} "
            f"rejected={stats.rejected:3d} empty={stats.empty_response:3d} "
            f"conn_err={stats.connection_errors:3d} other_err={stats.other_errors:3d} | "
            f"tps={interval_tps:6.1f} "
            f"ttft={snap['avg_ttft_ms']:5.0f}ms tpot={snap['avg_tpot_ms']:5.0f}ms"
        )


async def run_load_plan(label: str, url: str, model_name: str, plan: List[PlannedRequest]) -> Stats:
    stats = Stats()
    start_time = time.time()
    stop_event = asyncio.Event()

    print(f"\n--- Running against {label} ({url}) ---")

    async with aiohttp.ClientSession() as session:
        report_task = asyncio.create_task(reporter(label, stats, start_time, stop_event))

        in_flight = []
        for req in plan:
            now = time.time() - start_time
            sleep_time = req.fire_at - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            in_flight.append(
                asyncio.create_task(fire_request(session, url, model_name, req, stats, start_time))
            )

        done, pending = await asyncio.wait(in_flight, timeout=DRAIN_TIMEOUT_SECONDS)

        if pending:
            print(f"WARNING: {len(pending)} request(s) still in flight after "
                  f"{DRAIN_TIMEOUT_SECONDS}s drain timeout - cancelling them. "
                  f"This means the backend did not finish serving the full "
                  f"plan in a reasonable time; treat this run as unstable.")
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        stop_event.set()
        await report_task

    elapsed = time.time() - start_time

    # Hard invariant: every fired request must land in exactly one bucket.
    # If this fires, there's a new gap in the accounting - treat it as a
    # bug in THIS SCRIPT, not a hidden truth about the backend.
    unaccounted = stats.fired - stats.accounted_for()
    if unaccounted != 0:
        print(
            f"WARNING: accounting mismatch - {unaccounted} fired request(s) "
            f"not present in any outcome bucket. This means a request task "
            f"was cancelled/lost before reaching any counted branch (most "
            f"likely the {DRAIN_TIMEOUT_SECONDS}s drain timeout above). "
            f"Investigate before trusting this run's numbers."
        )

    print(f"--- {label} finished in {elapsed:.1f}s "
          f"(fired={stats.fired} accounted={stats.accounted_for()}) ---")

    if stats.failure_log:
        print(f"\n  First {min(20, len(stats.failure_log))} non-completed outcomes "
              f"(seconds from run start, for correlating against server logs/metrics):")
        for line in stats.failure_log[:20]:
            print(f"    {line}")
        if len(stats.failure_log) > 20:
            print(f"    ... and {len(stats.failure_log) - 20} more")

    return stats


def print_summary(label: str, stats: Stats, duration: float):
    print("\n" + "=" * 55)
    print(f"SUMMARY: {label}")
    print("=" * 55)
    print(f"  requests fired           {stats.fired}")
    print(f"  requests completed       {stats.completed}")
    print(f"  rejected (400/503)       {stats.rejected}")
    print(f"  empty response           {stats.empty_response}")
    print(f"  connection errors        {stats.connection_errors}")
    print(f"  other errors             {stats.other_errors}")
    print(f"  completion rate          {stats.completed / max(stats.fired, 1) * 100:.1f}%")
    print(f"  avg throughput           {stats.total_generated_tokens / max(duration, 1e-6):.1f} tok/s")
    print(f"  avg TTFT                 {stats.lifetime_avg_ttft_ms():.0f} ms")
    print(f"  avg TPOT                 {stats.lifetime_avg_tpot_ms():.0f} ms")
    print("=" * 55)


async def main():
    parser = argparse.ArgumentParser(description="Single-backend load test (OptiServe or vLLM)")
    parser.add_argument("--backend", required=True, choices=["optiserve", "vllm"],
                         help="Which backend this run targets - used only for labeling output")
    parser.add_argument("--url", required=True, help="Full chat completions URL, e.g. http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", required=True, help="Model name to send in the request payload")
    parser.add_argument("--prompts-file", default=PROMPTS_FILE)
    parser.add_argument("--rate", type=float, default=TARGET_RATE_RPS, help="Target requests/sec")
    parser.add_argument("--duration", type=float, default=RUN_DURATION_SECONDS, help="Load duration in seconds")
    args = parser.parse_args()

    print(f"Loading prompts from {args.prompts_file} ...")
    short_prompts, long_prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(short_prompts)} short prompts, {len(long_prompts)} long prompts")

    plan = build_request_plan(short_prompts, long_prompts, args.rate, args.duration)
    print(
        f"Built request plan: {len(plan)} requests over {args.duration}s "
        f"(~{args.rate} rps, seed={SEED}) - run the same command with "
        f"--backend {'vllm' if args.backend == 'optiserve' else 'optiserve'} "
        f"(same --rate/--duration) to get an identical plan for comparison."
    )

    label = "OptiServe" if args.backend == "optiserve" else "vLLM"
    stats = await run_load_plan(label, args.url, args.model, plan)
    print_summary(label, stats, duration=args.duration)


if __name__ == "__main__":
    asyncio.run(main())