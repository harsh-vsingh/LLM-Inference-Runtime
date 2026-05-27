from typing import List, Any
import asyncio
from collections import deque

class Scheduler:
    def __init__(self):
        self.queue = deque()
        self.active_batches = []
        self.max_batch_size = 8
        self.current_batch = []

    def add_request(self, request: Any):
        self.queue.append(request)
        self.schedule_batches()

    def schedule_batches(self):
        while len(self.current_batch) < self.max_batch_size and self.queue:
            self.current_batch.append(self.queue.popleft())
        
        if self.current_batch:
            self.process_batch(self.current_batch)
            self.current_batch = []

    def process_batch(self, batch: List[Any]):
        asyncio.create_task(self.run_inference(batch))

    async def run_inference(self, batch: List[Any]):
        await asyncio.sleep(1)
        print(f"Processed batch: {batch}")

    def clear_batches(self):
        self.active_batches.clear()
        self.current_batch.clear()

    def get_queue_length(self) -> int:
        return len(self.queue)

    def get_active_batches(self) -> List[List[Any]]:
        return self.active_batches
