import asyncio
from engine.metrics.request_metrics import RequestMetrics


class InferenceRequest:
    def __init__(
        self,
        request_id: str,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.output_queue: asyncio.Queue = asyncio.Queue()

        self.is_finished = False
        self.is_aborted = False

        self.metrics = RequestMetrics()

    def abort(self) -> None:
        self.is_aborted = True
        self.metrics.record_finished()

    async def put_token(self, token: str) -> None:
        self.metrics.record_token_generated()
        await self.output_queue.put(token)

    async def finish(self) -> None:
        self.is_finished = True
        self.metrics.record_finished()
        await self.output_queue.put(None)