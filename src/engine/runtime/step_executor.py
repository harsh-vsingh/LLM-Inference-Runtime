import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from engine.errors import BlockAllocationError
from engine.memory.allocator import BlockAllocator
from engine.memory.radix_cache import RadixCache
from engine.runtime.detokenizer import IncrementalDetokenizer
from engine.sequence import Sequence
from models.llama import (
    FlashInferState,
    build_batch_indices_positions,
    plan_decode,
    plan_prefill,
)


@dataclass
class StepResult:
    outputs: List[Tuple[Sequence, str]] = field(default_factory=list)
    is_prefill: bool = False
    preempted: List[Sequence] = field(default_factory=list)


class StepExecutor:
    def __init__(
        self,
        model,
        tokenizer,
        allocator: BlockAllocator,
        radix_cache: RadixCache,
        block_size: int,
        max_chunk_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sampler,
        allocate_blocks_fn,
        preempt_fn,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.allocator = allocator
        self.radix_cache = radix_cache
        self.block_size = block_size
        self.max_chunk_size = max_chunk_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._sample = sampler
        self._allocate_blocks_if_needed = allocate_blocks_fn
        self._preempt_sequence = preempt_fn

        self._detokenizers: Dict[str, IncrementalDetokenizer] = {}

    def _get_detokenizer(self, seq: Sequence) -> IncrementalDetokenizer:
        key = seq.request.request_id
        dt = self._detokenizers.get(key)
        if dt is None:
            dt = IncrementalDetokenizer(self.tokenizer, seq.prompt_token_ids)
            self._detokenizers[key] = dt
        return dt

    def drop_detokenizer(self, seq: Sequence) -> None:
        self._detokenizers.pop(seq.request.request_id, None)

    def run_step(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
        chunked_prefill_enabled: bool,
    ) -> StepResult:
        result = StepResult(is_prefill=bool(prefill_seqs))

        admitted_prefill_seqs = self._admit_or_preempt(prefill_seqs, result)
        if admitted_prefill_seqs:
            result.outputs.extend(self._run_prefill(admitted_prefill_seqs, chunked_prefill_enabled))

        if not decode_seqs:
            return result

        admitted_decode_seqs = self._admit_or_preempt(decode_seqs, result)
        if admitted_decode_seqs:
            result.outputs.extend(self._run_decode(admitted_decode_seqs))

        return result

    def _admit_or_preempt(self, seqs: List[Sequence], result: StepResult) -> List[Sequence]:
        admitted = []
        for seq in seqs:
            try:
                allocated = self._allocate_blocks_if_needed(seq)
            except BlockAllocationError:
                allocated = False

            if not allocated:
                self._preempt_sequence(seq)
                result.preempted.append(seq)
                continue

            admitted.append(seq)
        return admitted

    def _run_prefill(
        self, prefill_seqs: List[Sequence], chunked_prefill_enabled: bool
    ) -> List[Tuple[Sequence, str]]:
        max_chunk = self.max_chunk_size if chunked_prefill_enabled else 10 ** 9
        outputs: List[Tuple[Sequence, str]] = []

        for seq in prefill_seqs:
            remaining_len = len(seq.prompt_token_ids) - seq.computed_len
            chunk_len = min(remaining_len, max_chunk)
            new_computed_len = seq.computed_len + chunk_len

            chunk_ids = seq.prompt_token_ids[seq.computed_len:new_computed_len]
            logical_blocks_needed = math.ceil(new_computed_len / self.block_size)
            blocks_to_use = seq.block_table[:logical_blocks_needed]

            kv_indptr = torch.tensor([0, len(blocks_to_use)], dtype=torch.int32, device=self.model.device)
            kv_indices = torch.tensor(blocks_to_use, dtype=torch.int32, device=self.model.device)

            last_len = new_computed_len % self.block_size
            kv_last_page_len = torch.tensor(
                [last_len if last_len > 0 else self.block_size],
                dtype=torch.int32,
                device=self.model.device,
            )

            qo_indptr = torch.tensor([0, chunk_len], dtype=torch.int32, device=self.model.device)

            build_batch_indices_positions(
                qo_indptr,
                torch.tensor([chunk_len], dtype=torch.int32, device=self.model.device),
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

            input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.model.device)
            position_ids = torch.arange(
                seq.computed_len, new_computed_len, dtype=torch.long, device=self.model.device
            ).unsqueeze(0)

            out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

            seq.computed_len = new_computed_len

            if new_computed_len == len(seq.prompt_token_ids):
                seq.request.metrics.record_prefill_start()

                next_token = self._sample(out.logits[0, -1].unsqueeze(0), [seq])[0]
                seq.generated_token_ids.append(next_token)

                dt = self._get_detokenizer(seq)
                delta = dt.append(next_token)
                outputs.append((seq, delta))

                if seq.cached_prefix_len < len(seq.prompt_token_ids):
                    prompt_blocks = math.ceil(len(seq.prompt_token_ids) / self.block_size)
                    blocks = seq.block_table[:prompt_blocks]
                    self.radix_cache.insert(seq.prompt_token_ids, blocks)

        return outputs

    def _run_decode(self, decode_seqs: List[Sequence]) -> List[Tuple[Sequence, str]]:
        current_decode_batch_size = len(decode_seqs)
        for seq in decode_seqs:
            seq.request.metrics.record_decode_step(current_decode_batch_size)

        kv_indptr_list = [0]
        kv_indices_list: List[int] = []
        kv_last_page_len_list: List[int] = []

        for seq in decode_seqs:
            kv_indices_list.extend(seq.block_table)
            kv_indptr_list.append(len(kv_indices_list))

            total_len = seq.get_len
            last_len = total_len % self.block_size
            kv_last_page_len_list.append(last_len if last_len > 0 else self.block_size)

        kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device=self.model.device)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=self.model.device)
        kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=self.model.device)

        FlashInferState.batch_indices = torch.arange(
            len(decode_seqs), dtype=torch.int32, device=self.model.device
        )
        FlashInferState.positions = torch.tensor(
            [seq.get_len - 1 for seq in decode_seqs], dtype=torch.int32, device=self.model.device
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
            [[seq.generated_token_ids[-1]] for seq in decode_seqs],
            dtype=torch.long,
            device=self.model.device,
        )
        position_ids = torch.tensor(
            [[seq.get_len - 1] for seq in decode_seqs], dtype=torch.long, device=self.model.device
        )

        out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        next_tokens = self._sample(out.logits[:, -1, :], decode_seqs)

        outputs: List[Tuple[Sequence, str]] = []
        for i, seq in enumerate(decode_seqs):
            seq.generated_token_ids.append(next_tokens[i])

            dt = self._get_detokenizer(seq)
            delta = dt.append(next_tokens[i])
            outputs.append((seq, delta))

        return outputs