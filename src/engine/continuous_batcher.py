from collections import deque
import asyncio

class ContinuousBatcher:
    def __init__(self, max_batch_size=8, max_wait_time=0.5):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.waiting_queue = deque()
        self.active_batch = []
        self.batch_lock = asyncio.Lock()
        self.batch_event = asyncio.Event()

    async def add_request(self, request):
        async with self.batch_lock:
            self.waiting_queue.append(request)
            if len(self.active_batch) < self.max_batch_size:
                self.active_batch.append(request)
                if len(self.active_batch) == self.max_batch_size:
                    self.batch_event.set()
            else:
                self.batch_event.clear()

        await self.process_batch()

    async def process_batch(self):
        while True:
            await self.batch_event.wait()
            async with self.batch_lock:
                if not self.active_batch:
                    self.active_batch.extend(self.waiting_queue.popleft() for _ in range(min(self.max_batch_size, len(self.waiting_queue))))
                if len(self.active_batch) >= self.max_batch_size:
                    self.batch_event.set()
                else:
                    self.batch_event.clear()

            await self.decode_batch(self.active_batch)
            self.active_batch.clear()

    async def decode_batch(self, batch):
        await asyncio.sleep(self.max_wait_time)
        print(f"Decoded batch: {batch}")