import asyncio
import math
import time
import torch
from typing import Optional
from engine.radix_cache import RadixCache
from engine.request import InferenceRequest
from engine.sequence import Sequence
from engine.scheduler import Scheduler
from engine.memory import BlockAllocator

from models.llama import (
    FlashInferState,
    init_flashinfer_state,
    patch_llama_model,
    plan_prefill,
    plan_decode,
    build_batch_indices_positions,
)


class AsyncInferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size
        )
        config = model.config

        self.block_size = 16

        self.radix_cache = RadixCache(
            block_size=self.block_size
        )

        self.num_layers = config.num_hidden_layers
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = (
            config.hidden_size //
            config.num_attention_heads
        )

        self.allocator = BlockAllocator(
            num_blocks=2048,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=model.dtype,
            device=model.device,
        )

        patch_llama_model(model)

        init_flashinfer_state(
            kv_pool=self.allocator.kv_pool,
            page_size=self.block_size,
            device=model.device,
        )

        self.wakeup_event = asyncio.Event()
        
        self.total_prompt_tokens = 0
        self.cached_prompt_tokens = 0

        self.background_task: Optional[
            asyncio.Task
        ] = None

    def start(self):
        self.background_task = asyncio.create_task(
            self._process_loop()
        )
        self.metrics_task = asyncio.create_task(
            self._log_system_metrics()
        )
    async def add_request(
        self,
        request: InferenceRequest,
    ):
        prompt_ids = self.tokenizer(
            request.prompt
        ).input_ids

        seq = Sequence(
            request=request,
            prompt_token_ids=prompt_ids,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        matched_tokens, matched_blocks = self.radix_cache.match_prefix(prompt_ids)
        seq.block_table = list(matched_blocks)
        seq.cached_prefix_len = len(matched_tokens)
        
        self.cached_prompt_tokens += len(matched_tokens)

        seq.prev_text = self.tokenizer.decode(
            prompt_ids,
            skip_special_tokens=True,
        )

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
                
                hit_rate = (
                    (self.cached_prompt_tokens / self.total_prompt_tokens * 100) 
                    if self.total_prompt_tokens > 0 
                    else 0.0
                )
                
                print(f"[METRICS] Batch Size: {active_batch}/{self.scheduler.max_batch_size} | "
                      f"Queue: {waiting_queue} | "
                      f"GPU VRAM: Allc {mem_alloc:.2f}GB, Rsvd {mem_res:.2f}GB | "
                      f"Cache Hit Rate: {hit_rate:.2f}%")

    async def _process_loop(self):
        while True:
            await self.wakeup_event.wait()
            
            while self.scheduler.has_unfinished_sequences():
                try:
                    step_outputs = await asyncio.to_thread(self._pytorch_step)
                    
                    for seq, new_text in step_outputs:
                        await seq.request.put_token(new_text)
                        
                        if seq.is_finished():
                            seq.finish_time = time.time()
                            self._calculate_and_log_request_metrics(seq)
                            
                            self.allocator.decref(seq.block_table)
                            seq.block_table.clear()
                            
                            await seq.request.finish()
                            
                except Exception as e:
                    print(f"Engine Loop Error: {e}")
                    await asyncio.sleep(0.1)
                               
            self.wakeup_event.clear()

    def _calculate_and_log_request_metrics(self, seq: Sequence):
        queue_latency = seq.start_time - seq.arrival_time
        ttft = seq.first_token_time - seq.arrival_time
        
        num_generated = len(seq.generated_token_ids)
        if num_generated > 1:
            tpot = (seq.finish_time - seq.first_token_time) / (num_generated - 1)
        else:
            tpot = 0.0

        throughput = num_generated / max(seq.finish_time - seq.start_time, 1e-6)
        
        print(f"\n[REQUEST FINISHED] ID: {seq.request.request_id[:8]}...")
        print(f"  Queue Latency : {queue_latency:.4f}s")
        print(f"  TTFT          : {ttft:.4f}s")
        print(f"  TPOT          : {tpot*1000:.2f}ms/token")
        print(f"  Throughput    : {throughput:.2f} tokens/s")
        print(f"  Tokens        : {num_generated} generated, {len(seq.prompt_token_ids)} prompt")

    def _allocate_blocks_if_needed(
        self,
        seq: Sequence,
    ):
        logical_blocks = math.ceil(
            seq.get_len / self.block_size
        )

        current_blocks = len(seq.block_table)

        if logical_blocks > current_blocks:
            new_blocks = self.allocator.allocate(
                logical_blocks - current_blocks
            )

            seq.block_table.extend(new_blocks)


    @torch.inference_mode()
    def _pytorch_step(self):
        now = time.time()
        for seq in self.scheduler.running:
            if seq.start_time == 0.0:
                seq.start_time = now

        prefill_seqs, decode_seqs = (
            self.scheduler.step()
        )

        outputs = []

        for seq in prefill_seqs:
            self.total_prompt_tokens += len(seq.prompt_token_ids)
            
            self._allocate_blocks_if_needed(seq)

            kv_indptr = torch.tensor(
                [0, len(seq.block_table)],
                dtype=torch.int32,
                device=self.model.device,
            )

            kv_indices = torch.tensor(
                seq.block_table,
                dtype=torch.int32,
                device=self.model.device,
            )

            total_len = len(seq.prompt_token_ids)

            last_len = (
                total_len %
                self.block_size
            )

            kv_last_page_len = torch.tensor(
                [
                    last_len
                    if last_len > 0
                    else self.block_size
                ],
                dtype=torch.int32,
                device=self.model.device,
            )

            qo_indptr = torch.tensor(
                [0, len(seq.uncached_token_ids)],
                dtype=torch.int32,
                device=self.model.device,
            )

            build_batch_indices_positions(
                qo_indptr,
                torch.tensor(
                    [len(seq.uncached_token_ids)],
                    dtype=torch.int32,
                    device=self.model.device,
                ),
            )

            plan_prefill(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )

            input_ids = torch.tensor(
                [seq.uncached_token_ids],
                device=self.model.device,
            )

            prefix_len = seq.cached_prefix_len
            position_ids = torch.arange(
                prefix_len,
                prefix_len + len(seq.uncached_token_ids),
                device=self.model.device,
            ).unsqueeze(0)

            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

            next_token = int(
                torch.argmax(
                    out.logits[0, -1],
                    dim=-1,
                ).item()
            )

            seq.generated_token_ids.append(
                next_token
            )

            if seq.first_token_time == 0.0:
                seq.first_token_time = time.time()

            full_text = self.tokenizer.decode(
                seq.prompt_token_ids +
                seq.generated_token_ids,
                skip_special_tokens=True,
            )

            delta = full_text[
                len(seq.prev_text):
            ]

            seq.prev_text = full_text

            outputs.append((seq, delta))
        
        if prefill_seqs:
            for seq in prefill_seqs:
                full_blocks = ( len(seq.prompt_token_ids) // self.block_size )

                if full_blocks > 0:
                    newly_cached = self.radix_cache.insert(
                        seq.prompt_token_ids[:full_blocks * self.block_size],
                        seq.block_table[:full_blocks],
                    )
                    if newly_cached:
                        self.allocator.incref(newly_cached)

        if decode_seqs:
            for seq in decode_seqs:
                self._allocate_blocks_if_needed(
                    seq
                )

            kv_indptr_list = [0]
            kv_indices_list = []
            kv_last_page_len_list = []

            for seq in decode_seqs:
                kv_indices_list.extend(
                    seq.block_table
                )

                kv_indptr_list.append(
                    len(kv_indices_list)
                )

                last_len = (
                    seq.get_len %
                    self.block_size
                )

                kv_last_page_len_list.append(
                    last_len
                    if last_len > 0
                    else self.block_size
                )

            kv_indptr = torch.tensor(
                kv_indptr_list,
                dtype=torch.int32,
                device=self.model.device,
            )

            kv_indices = torch.tensor(
                kv_indices_list,
                dtype=torch.int32,
                device=self.model.device,
            )

            kv_last_page_len = torch.tensor(
                kv_last_page_len_list,
                dtype=torch.int32,
                device=self.model.device,
            )

            decode_seq_lens = torch.tensor(
                [
                    seq.get_len
                    for seq in decode_seqs
                ],
                dtype=torch.int32,
                device=self.model.device,
            )

            decode_append_indptr = torch.arange(
                len(decode_seqs) + 1,
                dtype=torch.int32,
                device=self.model.device,
            )

            build_batch_indices_positions(
                decode_append_indptr,
                decode_seq_lens,
            )

            plan_decode(
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )

            input_ids = torch.tensor(
                [
                    [
                        seq.generated_token_ids[-1]
                    ]
                    for seq in decode_seqs
                ],
                device=self.model.device,
            )

            position_ids = torch.tensor(
                [
                    [
                        seq.get_len - 1
                    ]
                    for seq in decode_seqs
                ],
                device=self.model.device,
            )

            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

            next_tokens = torch.argmax(
                out.logits[:, -1],
                dim=-1,
            ).tolist()

            for i, seq in enumerate(
                decode_seqs
            ):
                seq.generated_token_ids.append(
                    next_tokens[i]
                )

                full_text = self.tokenizer.decode(
                    seq.prompt_token_ids +
                    seq.generated_token_ids,
                    skip_special_tokens=True,
                )

                delta = full_text[
                    len(seq.prev_text):
                ]

                seq.prev_text = full_text

                outputs.append((seq, delta))

        return outputs
