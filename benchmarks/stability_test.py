import asyncio
import aiohttp
import time
import json
import random

API_URL = "http://localhost:8000/v1/chat/completions"

PROMPT_POOL_SIZE = 1000
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
        f"Loaded {len(prompts)} short prompts"
    )

    return prompts


async def fire_request(
    session,
    prompt,
):
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

    try:
        async with session.post(
            API_URL,
            json=payload,
        ) as response:
            async for _ in response.content:
                pass

    except Exception as e:
        print(
            f"Request failed: {e}"
        )


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

            print(
                f"Elapsed={elapsed:.1f}s "
                f"Requests={total}"
            )

            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())