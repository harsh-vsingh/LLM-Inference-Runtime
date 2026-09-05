import math
import time

from engine.config import EngineConfig
from engine.errors import AdmissionError, AdmissionRejectReason
from engine.memory.allocator import BlockAllocator
from engine.memory.prefix_match import match_prefix_for_new_run
from engine.memory.radix_cache import RadixCache
from engine.runtime.preemption_policy import select_preemption_victims
from engine.sequence import Sequence, SequenceStatus


class AdmissionController:
    """
    Owns the waiting/running queues and all admission decisions,
    including block reservation and preemption. Both the one-time entry
    gate (add_sequence) and the per-step decode/prefill plan (step) live
    here, along with the mechanics of preempting a running sequence.

    KV space is the primary admission driver. step() finalizes
    every block reservation for the step.

    Per-step planning follows a two-phase global reservation order:
      1. Reserve blocks for everything already RUNNING first - ongoing
         decodes and in-progress prefill chunks - evicting cache and
         preempting other running sequences if short.
      2. Leftover token budget admitting new prefill from `waiting`,
         following the same evict-then-preempt reservation order per
         candidate.
    """

    def __init__(
        self,
        config: EngineConfig,
        allocator: BlockAllocator,
        radix_cache: RadixCache,
        block_size: int,
        max_num_batched_tokens: int,
        on_reject=None,
        drop_detokenizer_fn=None,
        engine_metrics=None,
    ):
        self.config = config
        self.allocator = allocator
        self.radix_cache = radix_cache
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self._on_reject = on_reject

        self._drop_detokenizer_fn = drop_detokenizer_fn
        self._engine_metrics = engine_metrics

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

        self._preempted_this_step: list[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        self._check_admission(seq)
        self.waiting.append(seq)

    def _check_admission(self, seq: Sequence) -> None:
        if len(self.waiting) + len(self.running) >= self.config.max_queue_depth:
            self._reject(AdmissionRejectReason.QUEUE_FULL)

        total_kv_tokens = self.allocator.num_blocks * self.block_size
        if seq.original_prompt_len > total_kv_tokens:
            self._reject(
                AdmissionRejectReason.PROMPT_EXCEEDS_CAPACITY,
                f"prompt has {seq.original_prompt_len} tokens, exceeds total "
                f"KV capacity of {total_kv_tokens} tokens on this engine",
            )

    def _reject(self, reason: AdmissionRejectReason, message: str | None = None) -> None:
        if self._on_reject is not None:
            self._on_reject(reason)
        raise AdmissionError(reason, message)

    def expire_timed_out_waiting(self) -> list[Sequence]:
        """
        Removes and returns waiting sequences that have exceeded
        max_queue_wait_seconds, measured from the request's
        original arrival time(not reset on premption)
        """
        if not self.waiting:
            return []

        now = time.time()
        timeout = self.config.max_queue_wait_seconds
        expired = [
            seq for seq in self.waiting
            if (now - seq.request.metrics.created_at) > timeout
        ]
        if expired:
            expired_ids = {id(s) for s in expired}
            self.waiting = [s for s in self.waiting if id(s) not in expired_ids]
        return expired

    def step(self) -> tuple[list[Sequence], dict[int, int], list[Sequence], list[Sequence]]:
        """
        Returns (prefill_seqs, chunk_lens, decode_seqs, preempted).
        Preempted lists every sequence this call preempted.
        """
        self._preempted_this_step = []

        finished = [seq for seq in self.running if seq.is_finished()]
        for seq in finished:
            seq.status = SequenceStatus.FINISHED
            self.running.remove(seq)

        self._run_proactive_eviction()

        decode_seqs = [
            seq for seq in self.running
            if seq.computed_len >= len(seq.prompt_token_ids)
        ]
        in_progress_prefill = [
            seq for seq in self.running
            if seq.computed_len < len(seq.prompt_token_ids)
        ]

        decode_token_cost = len(decode_seqs)
        remaining_budget = self.max_num_batched_tokens - decode_token_cost

        max_chunk = self.config.max_chunk_size if self.config.enable_chunked_prefill else 10 ** 9

        chunk_lens: dict[int, int] = {}
        for seq in in_progress_prefill:
            chunk_len = min(
                len(seq.prompt_token_ids) - seq.computed_len,
                max_chunk,
                max(remaining_budget, 0),
            )
            chunk_lens[id(seq)] = chunk_len
            remaining_budget -= chunk_len

        failed_decode = self._reserve_blocks_for_running_decode(decode_seqs)
        if failed_decode:
            failed_ids = {id(s) for s in failed_decode}
            decode_seqs = [s for s in decode_seqs if id(s) not in failed_ids]

        failed_prefill_ids = set()
        for seq in in_progress_prefill:
            chunk_len = chunk_lens.get(id(seq), 0)
            if not self._reserve_blocks_for_prefill_chunk(seq, chunk_len):

                self._preempt(seq)
                failed_prefill_ids.add(id(seq))

        prefill_seqs: list[Sequence] = [
            seq for seq in in_progress_prefill if id(seq) not in failed_prefill_ids
        ]
        seen_ids = {id(s) for s in prefill_seqs}

        if remaining_budget > 0:
            admitted, admitted_chunk_lens = self._admit_prefill(remaining_budget)
            for seq in admitted:
                if id(seq) in seen_ids:
                    continue
                prefill_seqs.append(seq)
                chunk_lens[id(seq)] = admitted_chunk_lens[id(seq)]

        return prefill_seqs, chunk_lens, decode_seqs, self._preempted_this_step

    def _run_proactive_eviction(self) -> int:
        """
        Evicts cache only blocks down to a low water
        mark, checked once per step

        Threshold = min(eviction_low_water_blocks, eviction_low_water_pct
        * total_blocks)
        """
        total_blocks = self.allocator.num_blocks
        low_water = min(
            self.config.eviction_low_water_blocks,
            int(self.config.eviction_low_water_pct * total_blocks),
        )

        free_blocks = self.allocator.get_available_blocks()
        if free_blocks >= low_water:
            return 0

        deficit = low_water - free_blocks
        evicted = self.radix_cache.evict_lru(deficit)
        return len(evicted)

    def _admit_prefill(self, budget: int) -> tuple[list[Sequence], dict[int, int]]:
        """
        Iterates self.waiting by sequence identity - reservation can
        trigger preemption of a running sequence, which inserts that
        sequence back into self.waiting at index 0.
        """
        max_chunk = self.config.max_chunk_size if self.config.enable_chunked_prefill else 10 ** 9

        admitted: list[Sequence] = []
        chunk_lens: dict[int, int] = {}
        used = 0

        candidates = list(self.waiting)
        for seq in candidates:
            if seq not in self.waiting:
                continue

            remaining_budget = budget - used
            if remaining_budget <= 0:
                break

            next_chunk_len = min(
                seq.get_num_uncomputed_tokens(),
                max_chunk,
                remaining_budget,
            )

            if next_chunk_len <= 0:
                continue

            if not self._reserve_blocks_for_prefill_chunk(seq, next_chunk_len):
                continue

            if seq not in self.waiting:
                continue

            self.waiting.remove(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            admitted.append(seq)
            chunk_lens[id(seq)] = next_chunk_len
            used += next_chunk_len

        return admitted, chunk_lens

    def _reserve_blocks_for_prefill_chunk(self, seq: Sequence, chunk_len: int) -> bool:
        """
        Ensures blocks are available AND ASSIGNED for the next
        chunk_len tokens of seq.
        """
        logical_blocks_needed = (
            math.ceil((seq.computed_len + chunk_len) / self.block_size) - len(seq.block_table)
        )
        if logical_blocks_needed <= 0:
            return True

        if not self.allocator.try_allocate(logical_blocks_needed, self.radix_cache):
            if not self.running:
                return False

            still_needed = logical_blocks_needed - self.allocator.get_available_blocks()
            victims = select_preemption_victims(
                running_seqs=self.running,
                blocks_needed=still_needed,
                allocator=self.allocator,
                completion_protection_threshold=self.config.completion_protection_threshold,
            )
            for victim in victims:
                self._preempt(victim)

            if self.allocator.get_available_blocks() < logical_blocks_needed:
                return False

        seq.block_table.extend(self.allocator.allocate(logical_blocks_needed))
        return True

    def _reserve_blocks_for_running_decode(self, decode_seqs: list[Sequence]) -> list[Sequence]:
        """
        Reserves, in a single batched pass, whatever new blocks the
        decode batch needs for this step's token

        Preempts running sequences if short on blocks after eviction.
        Returns the subset of decode_seqs that still couldn't get a
        block; those have already been preempted by this call
        """
        needs: dict[int, int] = {}
        total_needed = 0
        for seq in decode_seqs:
            required_blocks = math.ceil((seq.get_len + 1) / self.block_size)
            extra = required_blocks - len(seq.block_table)
            if extra > 0:
                needs[id(seq)] = extra
                total_needed += extra

        if total_needed == 0:
            return []

        if not self.allocator.try_allocate(total_needed, self.radix_cache):
            if self.running:
                still_needed = total_needed - self.allocator.get_available_blocks()
                victims = select_preemption_victims(
                    running_seqs=self.running,
                    blocks_needed=still_needed,
                    allocator=self.allocator,
                    completion_protection_threshold=self.config.completion_protection_threshold,
                )
                for victim in victims:
                    self._preempt(victim)
                    needs.pop(id(victim), None)

        failed: list[Sequence] = []
        for seq in decode_seqs:
            need = needs.get(id(seq))
            if not need:
                continue
            if self.allocator.get_available_blocks() < need:
                self._preempt(seq)
                failed.append(seq)
                continue
            seq.block_table.extend(self.allocator.allocate(need))

        return failed

    def _preempt(self, seq: Sequence) -> None:

        seq.request.metrics.record_preemption()
        self._preempted_this_step.append(seq)
        if self._engine_metrics is not None:
            self._engine_metrics.record_preemption()

        if seq.request.metrics.preemption_count > self.config.max_preemption_retries:
            self._fail_preempted_sequence(seq)
            return

        if seq.block_table:
            self.allocator.decref(seq.block_table)
            seq.block_table.clear()

        seq.prompt_token_ids.extend(seq.generated_token_ids)
        seq.generated_token_ids.clear()

        if self._drop_detokenizer_fn is not None:
            self._drop_detokenizer_fn(seq)

        matched_tokens, matched_blocks = match_prefix_for_new_run(
            self.radix_cache, self.allocator, seq.prompt_token_ids
        )

        seq.cached_prefix_len = len(matched_tokens)
        seq.computed_len = seq.cached_prefix_len
        seq.block_table = list(matched_blocks)
        seq.status = SequenceStatus.WAITING

        if seq in self.running:
            self.running.remove(seq)

        self.waiting.insert(0, seq)

    def _fail_preempted_sequence(self, seq: Sequence) -> None:
        """
        A sequence that has exhausted its preemption retry budget is
        aborted. Bypasses the normal ProcessLoop._finish_sequence
        so does the equivalent bookkeeping itself:
        """
        if seq.block_table:
            self.allocator.decref(seq.block_table)
            seq.block_table.clear()

        seq.status = SequenceStatus.FINISHED

        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)

        seq.request.is_aborted = True

        if self._drop_detokenizer_fn is not None:
            self._drop_detokenizer_fn(seq)

        if self._engine_metrics is not None:
            self._engine_metrics.record_aborted_request()

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running)