
from engine.memory.allocator import BlockAllocator
from engine.sequence import Sequence


def _completion_fraction(seq: Sequence) -> float:
    max_new = seq.request.max_new_tokens
    if max_new <= 0:
        return 1.0
    return len(seq.generated_token_ids) / max_new


def _exclusive_block_count(seq: Sequence, allocator: BlockAllocator) -> int:
    return sum(1 for b in seq.block_table if allocator.ref_counts[b] == 1)


def select_preemption_victims(
    running_seqs: list[Sequence],
    blocks_needed: int,
    allocator: BlockAllocator,
    completion_protection_threshold: float = 0.8,
) -> list[Sequence]:
    """
    Chooses which running sequences to preempt to free at least
    blocks_needed blocks, minimizing wasted work.

    Strategy:
      1. Only blocks with ref_count == 1 (exclusively held by
         this sequence) become free if it's preempted - that's the
         exclusive block count computed per candidate.
      2. Sequences past completion_protection_threshold of their
         max_new_tokens are excluded from candidacy. If every running sequence is
         above the threshold, protection is
         waived rather than deadlocking admission - the least-bad option
         (closest to done, so cheapest to redo) is chosen.
      3. Among eligible candidates, greedily take the highest-yield
         sequence first, repeating until the cumulative freed blocks meet
         blocks_needed.

    Only selects victims, does not preempt them. Assumes blocks_needed already accounts
    for cache-only eviction having been attempted first via
    RadixCache.evict_lru(), which has zero disruption cost and should
    always be exhausted before this function is ever called.
    """
    if blocks_needed <= 0 or not running_seqs:
        return []

    candidates = [
        (seq, _exclusive_block_count(seq, allocator)) for seq in running_seqs
    ]

    eligible = [
        (seq, yield_) for seq, yield_ in candidates
        if _completion_fraction(seq) < completion_protection_threshold
    ]

    if not eligible:
        eligible = sorted(candidates, key=lambda pair: -_completion_fraction(pair[0]))
    else:
        eligible.sort(key=lambda pair: pair[1], reverse=True)

    victims: list[Sequence] = []
    freed = 0
    for seq, yield_ in eligible:
        if freed >= blocks_needed:
            break
        victims.append(seq)
        freed += yield_

    return victims