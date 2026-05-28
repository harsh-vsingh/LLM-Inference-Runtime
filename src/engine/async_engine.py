import asyncio
import torch
import time
from typing import Optional
from engine.request import InferenceRequest
from engine.sequence import Sequence, SequenceStatus
from engine.scheduler import Scheduler

class AsyncInferenceEngine:
    def __init__(self, model, tokenizer, max_batch_size: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.wakeup_event = asyncio.Event()
        self.background_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None

    def start(self):
        self.background_task = asyncio.create_task(self._process_loop())
        self.metrics_task = asyncio.create_task(self._log_system_metrics())

    async def add_request(self, request: InferenceRequest):
        prompt_ids = self.tokenizer(request.prompt).input_ids
        seq = Sequence(
            request=request,
            prompt_token_ids=prompt_ids,
            eos_token_id=self.tokenizer.eos_token_id
        )
        seq.prev_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
        self.scheduler.add_sequence(seq)
        self.wakeup_event.set()

    async def _log_system_metrics(self):
        while True:
            await asyncio.sleep(5)
            if self.scheduler.has_unfinished_sequences():
                active_batch = len(self.scheduler.running)
                waiting_queue = len(self.scheduler.waiting)
                
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_res = torch.cuda.memory_reserved() / (1024**3)
                
                print(f"[METRICS] Batch Size: {active_batch}/{self.scheduler.max_batch_size} | "
                      f"Queue: {waiting_queue} | "
                      f"GPU VRAM: Allc {mem_alloc:.2f}GB, Rsvd {mem_res:.2f}GB | "
                      f"Cache Hit Rate: 0.0% (Pending Phase 5)")

    async def _process_loop(self):
        while True:
            await self.wakeup_event.wait()
            
            while self.scheduler.has_unfinished_sequences():
                step_outputs = await asyncio.to_thread(self._pytorch_step)
                
                for seq, new_text in step_outputs:
                    await seq.request.put_token(new_text)
                    
                    if seq.is_finished():
                        seq.finish_time = time.time()
                        self._calculate_and_log_request_metrics(seq)
                        await seq.request.finish()
                        
                await asyncio.sleep(0)
                
            self.wakeup_event.clear()

    def _calculate_and_log_request_metrics(self, seq: Sequence):
        queue_latency = seq.start_time - seq.arrival_time
        ttft = seq.first_token_time - seq.arrival_time
        
        num_generated = len(seq.generated_token_ids)
        if num_generated > 1:
            tpot = (seq.finish_time - seq.first_token_time) / (num_generated - 1)
        else:
            tpot = 0.0

        throughput = num_generated / (seq.finish_time - seq.start_time)
        
        print(f"\n[REQUEST FINISHED] ID: {seq.request.request_id[:8]}...")
        print(f"  Queue Latency : {queue_latency:.4f}s")
        print(f"  TTFT          : {ttft:.4f}s")
        print(f"  TPOT          : {tpot*1000:.2f}ms/token")
        print(f"  Throughput    : {throughput:.2f} tokens/s")
        print(f"  Tokens        : {num_generated} generated, {len(seq.prompt_token_ids)} prompt")

    @torch.inference_mode()
    def _pytorch_step(self):
        now = time.time()
        for seq in self.scheduler.running:
            if seq.start_time == 0.0:
                seq.start_time = now

        prefill_seqs, decode_seqs = self.scheduler.step()
        outputs = []

        for seq in prefill_seqs:
            input_ids = torch.tensor([seq.prompt_token_ids], device=self.model.device)
            out = self.model(input_ids=input_ids, use_cache=True)
            next_token_id = int(torch.argmax(out.logits[0, -1, :], dim=-1).item())
            
            seq.past_key_values = out.past_key_values
            seq.generated_token_ids.append(next_token_id)
            seq.first_token_time = time.time()
            
            full_text = self.tokenizer.decode(
                seq.prompt_token_ids + seq.generated_token_ids, 
                skip_special_tokens=True
            )
            new_text = full_text[len(seq.prev_text):]
            seq.prev_text = full_text
            outputs.append((seq, new_text))

        for seq in decode_seqs:
            input_ids = torch.tensor([[seq.generated_token_ids[-1]]], device=self.model.device)
            out = self.model(
                input_ids=input_ids, 
                past_key_values=seq.past_key_values, 
                use_cache=True
            )
            next_token_id = int(torch.argmax(out.logits[0, -1, :], dim=-1).item())
            
            seq.past_key_values = out.past_key_values
            seq.generated_token_ids.append(next_token_id)
            
            full_text = self.tokenizer.decode(
                seq.prompt_token_ids + seq.generated_token_ids, 
                skip_special_tokens=True
            )
            new_text = full_text[len(seq.prev_text):]
            seq.prev_text = full_text
            outputs.append((seq, new_text))

        return outputs