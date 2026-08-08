import math

from engine.memory.allocator import BlockAllocator
from engine.memory.prefix_match import match_prefix_for_new_run
from engine.memory.radix_cache import RadixCache
from engine.scheduling.scheduler import Scheduler
from engine.sequence import Sequence


class BlockAdmission:
    def __init__(
        self,
        allocator: BlockAllocator,
        radix_cache: RadixCache,
        scheduler: Scheduler,
        block_size: int,
        on_preempt=None,
    ):
        self.allocator = allocator
        self.radix_cache = radix_cache
        self.scheduler = scheduler
        self.block_size = block_size
        self._on_preempt = on_preempt

    def allocate_blocks_if_needed(self, seq: Sequence) -> bool:
        logical_blocks = math.ceil(seq.get_len / self.block_size)
        current_blocks = len(seq.block_table)
        needed = logical_blocks - current_blocks

        if needed <= 0:
            return True

        available = self.allocator.get_available_blocks()
        if available < needed:
            to_free = needed - available
            self.radix_cache.evict_lru(to_free)

        if self.allocator.get_available_blocks() < needed:
            return False

        new_blocks = self.allocator.allocate(needed)
        seq.block_table.extend(new_blocks)
        return True

    def preempt_sequence(self, seq: Sequence) -> None:
        if seq.block_table:
            self.allocator.decref(seq.block_table)
            seq.block_table.clear()

        seq.prompt_token_ids.extend(seq.generated_token_ids)
        seq.generated_token_ids.clear()

        matched_tokens, matched_blocks = match_prefix_for_new_run(
            self.radix_cache, self.allocator, seq.prompt_token_ids
        )

        seq.cached_prefix_len = len(matched_tokens)
        seq.computed_len = seq.cached_prefix_len
        seq.block_table = list(matched_blocks)

        seq.request.metrics.record_preemption()

        if seq in self.scheduler.running:
            self.scheduler.running.remove(seq)

        self.scheduler.waiting.insert(0, seq)

        if self._on_preempt is not None:
            self._on_preempt(seq)