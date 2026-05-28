import asyncio

class InferenceRequest:
    def __init__(self, request_id: str, prompt: str, max_new_tokens: int):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        
        self.output_queue = asyncio.Queue()
        
        self.is_finished = False
        self.is_cancelled = False

    async def put_token(self, token: str):
        await self.output_queue.put(token)
        
    async def finish(self):
        self.is_finished = True
        await self.output_queue.put(None) 