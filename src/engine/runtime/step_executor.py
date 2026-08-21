import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

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
    # (seq, prompt_token_ids, blocks) for sequences that finished their
    # prefill this step and need inserting into the radix cache.
    finished_prefills: List[Tuple[Sequence, List[int], List[int]]] = field(default_factory=list)


class StepExecutor:
    def __init__(  
        self,
        model,
        tokenizer,
        block_size: int,
        max_chunk_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sampler,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_chunk_size = max_chunk_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._sample = sampler

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
        prefill_chunk_lens: Dict[int, int],
        decode_seqs: List[Sequence],
    ) -> StepResult:
        """
        prefill_seqs/decode_seqs are assumed already admitted - i.e.
        AdmissionController.step() has already ensured each seq's
        block_table has enough blocks for this step's tokens.
        """
        result = StepResult(is_prefill=bool(prefill_seqs))

        if prefill_seqs:
            result.outputs.extend(
                self._run_prefill(prefill_seqs, prefill_chunk_lens, result)
            )

        if decode_seqs:
            result.outputs.extend(self._run_decode(decode_seqs))

        return result

    def _run_prefill(
        self,
        prefill_seqs: List[Sequence],
        prefill_chunk_lens: Dict[int, int],
        result: StepResult,
    ) -> List[Tuple[Sequence, str]]:
        device = self.model.device

        active_seqs: List[Sequence] = []
        chunk_ids_list: List[List[int]] = []
        chunk_lens: List[int] = []
        new_computed_lens: List[int] = []
        blocks_per_seq: List[List[int]] = []
        last_page_lens: List[int] = []
        position_id_ranges: List[Tuple[int, int]] = []

        for seq in prefill_seqs:
            remaining_len = len(seq.prompt_token_ids) - seq.computed_len

            if id(seq) not in prefill_chunk_lens:
                raise ValueError(
                    f"StepExecutor._run_prefill received seq (request_id="
                    f"{seq.request.request_id}) with no entry in "
                    f"prefill_chunk_lens - AdmissionController.step() must "
                    f"authorize a chunk length for every prefill_seq it returns."
                )
            chunk_len = min(prefill_chunk_lens[id(seq)], remaining_len)
            if chunk_len <= 0:
                self._zero_chunk_skips = getattr(self, "_zero_chunk_skips", 0) + 1
                continue

            new_computed_len = seq.computed_len + chunk_len

            chunk_ids = seq.prompt_token_ids[seq.computed_len:new_computed_len]
            logical_blocks_needed = math.ceil(new_computed_len / self.block_size)
            blocks_to_use = seq.block_table[:logical_blocks_needed]

            last_len = new_computed_len % self.block_size

            active_seqs.append(seq)
            chunk_ids_list.append(chunk_ids)
            chunk_lens.append(chunk_len)
            new_computed_lens.append(new_computed_len)
            blocks_per_seq.append(blocks_to_use)
            last_page_lens.append(last_len if last_len > 0 else self.block_size)
            position_id_ranges.append((seq.computed_len, new_computed_len))

        if not active_seqs:
            return []

        qo_indptr_list = [0]
        for cl in chunk_lens:
            qo_indptr_list.append(qo_indptr_list[-1] + cl)
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=device)

        kv_indptr_list = [0]
        kv_indices_list: List[int] = []
        for blocks in blocks_per_seq:
            kv_indices_list.extend(blocks)
            kv_indptr_list.append(len(kv_indices_list))
        kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device=device)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=device)
        kv_last_page_len = torch.tensor(last_page_lens, dtype=torch.int32, device=device)

        flat_input_ids = [tok for chunk in chunk_ids_list for tok in chunk]
        input_ids = torch.tensor([flat_input_ids], dtype=torch.long, device=device)

        flat_positions = [
            pos for start, end in position_id_ranges for pos in range(start, end)
        ]
        position_ids = torch.tensor([flat_positions], dtype=torch.long, device=device)

        seq_lens_cumulative = torch.tensor(new_computed_lens, dtype=torch.int32, device=device)
        build_batch_indices_positions(qo_indptr, seq_lens_cumulative)

        plan_prefill(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

        outputs: List[Tuple[Sequence, str]] = []

        for i, seq in enumerate(active_seqs):
            seq.computed_len = new_computed_lens[i]

            if new_computed_lens[i] == len(seq.prompt_token_ids):
                seq.request.metrics.record_prefill_start()

                last_token_idx = qo_indptr_list[i + 1] - 1
                next_token = self._sample(
                    out.logits[0, last_token_idx].unsqueeze(0), [seq]
                )[0]
                seq.generated_token_ids.append(next_token)

                dt = self._get_detokenizer(seq)
                delta = dt.append(next_token)
                outputs.append((seq, delta))

                if seq.cached_prefix_len < len(seq.prompt_token_ids):
                    prompt_blocks = math.ceil(len(seq.prompt_token_ids) / self.block_size)
                    blocks = seq.block_table[:prompt_blocks]
                    # Defer the actual radix_cache.insert to the caller
                    # (ProcessLoop, after the asyncio.to_thread hop) -
                    # see StepResult.finished_prefills docstring.
                    result.finished_prefills.append(
                        (seq, list(seq.prompt_token_ids), blocks)
                    )

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