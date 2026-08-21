import math
import time
from typing import Dict, List, Optional, Tuple

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

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

        self._preempted_this_step: List[Sequence] = []

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

    def _reject(self, reason: AdmissionRejectReason, message: Optional[str] = None) -> None:
        if self._on_reject is not None:
            self._on_reject(reason)
        raise AdmissionError(reason, message)

    def expire_timed_out_waiting(self) -> List[Sequence]:
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

    def step(self) -> Tuple[List[Sequence], Dict[int, int], List[Sequence], List[Sequence]]:
        """
        Returns (prefill_seqs, chunk_lens, decode_seqs, preempted).

        preempted lists every sequence this call preempted, across all
        three reservation sites (running-decode reservation,
        running-prefill reservation, new-prefill admission). All block
        allocation for the step is finalized before this returns.

        Two-phase global reservation:
          Phase 1 - reserve for everything already RUNNING (decode
          cost for every running seq, plus the next chunk for every
          running seq still mid-prefill). Neither is optional or
          skippable this step.
          Phase 2 - spend whatever token budget remains admitting new
          prefill from `waiting`.
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

        chunk_lens: Dict[int, int] = {}
        for seq in in_progress_prefill:
            chunk_len = min(
                len(seq.prompt_token_ids) - seq.computed_len,
                max_chunk,
                max(remaining_budget, 0),
            )
            chunk_lens[id(seq)] = chunk_len
            remaining_budget -= chunk_len

        # --- Phase 1: reserve for everything already running ---
        # Decode first (cheaper, never skippable), then in-progress
        # prefill chunks - both computed as batched totals so the
        # deficit (if any) is known globally before we evict/preempt,
        # rather than resolving seq-by-seq and risking one seq's
        # reservation preempting another seq we're about to reserve for
        # in the very same phase.
        failed_decode = self._reserve_blocks_for_running_decode(decode_seqs)
        if failed_decode:
            failed_ids = {id(s) for s in failed_decode}
            decode_seqs = [s for s in decode_seqs if id(s) not in failed_ids]

        failed_prefill_ids = set()
        for seq in in_progress_prefill:
            chunk_len = chunk_lens.get(id(seq), 0)
            if not self._reserve_blocks_for_prefill_chunk(seq, chunk_len):
                # Sequence already RUNNING and mid-prefill but couldn't
                # get its next chunk's blocks even after eviction and
                # preemption of *other* running sequences - preempt it
                # too, same as a failed decode reservation. There's no
                # "just skip it this step" option once something is
                # RUNNING.
                self._preempt(seq)
                failed_prefill_ids.add(id(seq))

        prefill_seqs: List[Sequence] = [
            seq for seq in in_progress_prefill if id(seq) not in failed_prefill_ids
        ]
        seen_ids = {id(s) for s in prefill_seqs}

        # --- Phase 2: admit new prefill from `waiting` with whatever
        # budget is left over ---
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

    def _admit_prefill(self, budget: int) -> Tuple[List[Sequence], Dict[int, int]]:
        """
        Iterates self.waiting by sequence identity - reservation can
        trigger preemption of a running sequence, which inserts that
        sequence back into self.waiting at index 0.
        """
        max_chunk = self.config.max_chunk_size if self.config.enable_chunked_prefill else 10 ** 9

        admitted: List[Sequence] = []
        chunk_lens: Dict[int, int] = {}
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
        chunk_len tokens of seq before it's included in this step's
        prefill batch:

          1. Cache-only eviction
          2. If still insufficient, preempt already-running sequences
          3. Commit: extend seq.block_table with the newly allocated
             blocks

        Step 3 is the fix for the double-booking gap this used to have:
        previously this only checked/evicted for availability
        (try_allocate) without ever calling allocator.allocate() to
        commit blocks into block_table, so two different waiting seqs
        could each pass the check against the same free pool and only
        one would actually get usable blocks by the time StepExecutor
        read block_table. Now the commit happens here, atomically with
        the check, before this seq is ever returned as admitted.
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

    def _reserve_blocks_for_running_decode(self, decode_seqs: List[Sequence]) -> List[Sequence]:
        """
        Reserves, in a single batched pass, whatever new blocks the
        decode batch needs for this step's token

        Preempts running sequences if short on blocks after eviction.
        Returns the subset of decode_seqs that still couldn't get a
        block; those have already been preempted by this call
        """
        needs: Dict[int, int] = {}
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

        failed: List[Sequence] = []
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
        """
        The single preemption entry point - every one of the three
        reservation sites above (running-decode, running-prefill,
        new-prefill-admission-evicting-a-victim) calls this and only
        this, rather than each duplicating the reset/requeue/fail
        logic. Also the single place step() records that a preemption
        happened, via _preempted_this_step, regardless of which site
        triggered it.

        Runs seq's preemption-count check; on repeated failure past
        max_preemption_retries, hard-fails the sequence instead of
        requeuing it (see _fail_preempted_sequence).
        """
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
        hard-aborted rather than requeued again. This bypasses the
        normal ProcessLoop._finish_sequence teardown (the sequence
        never runs another step to reach it), so this method must do
        the equivalent bookkeeping itself:
          - decref its blocks
          - remove it from running/waiting
          - drop its detokenizer (nothing will read it again)
          - mark the request aborted
          - record it as a dedicated aborted-request metric, NOT as a
            successful record_finished_request - doing the latter
            would pollute TTFT/TPOT averages with a request that never
            actually finished generating.
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