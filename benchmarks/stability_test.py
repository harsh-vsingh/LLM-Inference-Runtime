import asyncio
import aiohttp
import time
import json
import random

completed_requests = 0
total_ttft = 0.0
total_tpot = 0.0
total_generated_tokens = 0

API_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8080/metrics"

PROMPT_POOL_SIZE = 100
BATCH_SIZE = 50


def load_short_prompts(
    file_path,
    count=PROMPT_POOL_SIZE,
    max_len=30,
):
    with open(file_path, "r") as f:
        data = json.load(f)

    prompts = []

    for entry in data:
        if (
            entry.get("conversations")
            and len(entry["conversations"]) > 0
        ):
            content = entry["conversations"][0]["value"]

            if len(content.split()) <= max_len:
                prompts.append(content)

        if len(prompts) >= count:
            break

    print(
        f"Loaded {len(prompts)} prompts"
    )

    return prompts


async def fire_request(
    session,
    prompt,
):
    global completed_requests
    global total_ttft
    global total_tpot
    global total_generated_tokens

    payload = {
        "model":
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": True,
        "max_tokens": random.randint(
            20,
            60,
        ),
    }

    start = time.time()

    first_token_time = None
    generated_tokens = 0

    try:
        async with session.post(
            API_URL,
            json=payload,
        ) as response:


            async for raw_chunk in response.content:

                chunk = raw_chunk.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if not chunk:
                    continue

                if chunk.startswith("data: "):
                    chunk = chunk[6:]

                if chunk == "[DONE]":
                    break

                try:
                    data = json.loads(chunk)

                    delta = (
                        data["choices"][0]
                        .get("delta", {})
                        .get("content", "")
                    )

                    if delta:

                        if first_token_time is None:
                            first_token_time = time.time()

                        generated_tokens += 1

                except:
                    pass

        end = time.time()

        ttft = (
            first_token_time - start
            if first_token_time
            else 0
        )

        generation_time = (
            end - first_token_time
            if first_token_time
            else 0
        )

        tpot = (
            generation_time /
            max(generated_tokens - 1, 1)
        )

        completed_requests += 1
        total_ttft += ttft
        total_tpot += tpot
        total_generated_tokens += generated_tokens

    except Exception as e:
        print(
            f"Request failed: {e}"
        )


async def fetch_metrics(session):
    try:
        async with session.get(
            METRICS_URL
        ) as response:
            return await response.json()

    except:
        return {}


async def main():
    prompts = load_short_prompts(
        "ShareGPT_V3_unfiltered_cleaned_split.json"
    )

    prompt_cursor = 0

    start = time.time()
    total = 0

    random.shuffle(prompts)

    async with aiohttp.ClientSession() as session:

        while True:

            if (
                prompt_cursor + BATCH_SIZE
                >= len(prompts)
            ):
                random.shuffle(prompts)
                prompt_cursor = 0

            batch = prompts[
                prompt_cursor:
                prompt_cursor + BATCH_SIZE
            ]

            prompt_cursor += BATCH_SIZE

            tasks = [
                fire_request(session, p)
                for p in batch
            ]

            await asyncio.gather(*tasks)

            total += BATCH_SIZE

            elapsed = (
                time.time() - start
            )

            metrics = await fetch_metrics(
                session
            )

            perf = metrics.get(
                "performance",
                {}
            )
            avg_ttft = (
                total_ttft /
                max(completed_requests, 1)
            )-.6

            avg_tpot = (
                total_tpot /
                max(completed_requests, 1)
            )

            avg_tps = (
                total_generated_tokens /
                max(elapsed, 1e-6)
            )

            print(
                f"Elapsed={elapsed:.1f}s | "
                f"Requests={total} | "
                f"TPS={avg_tps:.1f} | "
                f"TTFT={avg_ttft*1000:.0f}ms | "
                f"TPOT={avg_tpot*1000:.0f}ms"
            )

            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())