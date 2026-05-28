import asyncio
from typing import Optional
from engine.request import InferenceRequest
from engine.generator import generate_tokens

class AsyncInferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.request_queue = asyncio.Queue()
        self.background_task: Optional[asyncio.Task] = None

    def start(self):
        self.background_task = asyncio.create_task(self._process_loop())

    async def add_request(self, request: InferenceRequest):
        await self.request_queue.put(request)

    async def _process_loop(self):
        def _get_next(gen):
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            request: InferenceRequest = await self.request_queue.get()
            
            try:
                gen = generate_tokens(
                    self.model, 
                    self.tokenizer, 
                    request.prompt, 
                    request.max_new_tokens
                )
                
                while not request.is_cancelled:
                    token = await asyncio.to_thread(_get_next, gen)
                    if token is None:
                        break
                    await request.put_token(token)
            except Exception as e:
                print(f"Generation error: {e}")
            finally:
                await request.finish()
                self.request_queue.task_done()