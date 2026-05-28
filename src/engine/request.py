import asyncio
import time
from typing import Dict, Any

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
        
        self.output_queue = asyncio.Queue()
        
        self.is_finished = False
        self.is_aborted = False

        # --- ADVANCED TELEMETRY ---
        self.metrics: Dict[str, Any] = {
            "created_at": time.time(),
            "admitted_at": 0.0,
            "prefill_start": 0.0,
            "first_token_time": 0.0,
            "finished_at": 0.0,
            "tokens_generated": 0,
            
            "prefix_depth_tokens": 0,
            "cache_blocks_reused": 0,
            
            "decode_steps": 0,
            "sum_decode_batch_size": 0,
        }

    def abort(self):
        self.is_aborted = True
        self.metrics["finished_at"] = time.time()

    async def put_token(self, token: str):
        if self.metrics["first_token_time"] == 0.0:
            self.metrics["first_token_time"] = time.time()
            
        self.metrics["tokens_generated"] += 1
        await self.output_queue.put(token)
        
    async def finish(self):
        self.is_finished = True
        self.metrics["finished_at"] = time.time()
        await self.output_queue.put(None)