# Engine Architecture Notes: Memory Management & Scheduling

## Problems
* **Unbounded Preemption Thrashing Risk**: The engine lacks strict rate limits on preemption operations, relying only on a completion protection threshold. Preempting too many running decode sequences in a single step risks system thrashing, where compute budget is wasted continuously recalculating preempted prefill prefixes instead of generating new tokens.
* **Compute Waste on Full Cache Matches**: When a prompt is a 100% match in the cache, `match_prefix_for_new_run` drops the last matched block to force at least one token into the prefill step. This wastes compute by forcing the engine to recalculate up to `block_size` tokens rather than fully utilizing the cache hit.
* **Missing Absolute Context Window Enforcement**: The `sequence.is_finished()` method lacks a safeguard against the model's absolute maximum positional context length. A request can theoretically generate tokens past the hardware embeddings limit, risking a fatal CUDA out-of-bounds crash.

## To Implement

### Correctness
* **Context Window Termination**: Inject a `max_model_len` property into `EngineConfig` and add a hard cutoff condition to `sequence.is_finished()` to safely terminate requests before exceeding hardware constraints.

### Improvements
* **Advanced Eviction Victim Selection**: Use the existing radix tree to create a smarter eviction victim selection algorithm. Need to weigh in on the exact tradeoffs and algorithms (e.g., evaluating shared prefix depths and block overlaps to minimize overall compute cost).
* **Updated Decode path Fused kernels + CUDA Graphs**: Optimize decode path. Benchmark and potentially add fused kernels and CUDA graphs.
* **Direct-to-Decode for Full Cache Matches**: Refactor `AdmissionController` to detect 100% prefix matches. Bypass the prefill budget allocation and immediately push the sequence into the `decode_seqs` batch to eliminate compute waste.
* **Incremental Radix Cache Insertion & Copy-on-Write (CoW)**: Update the memory manager to insert decode blocks into the `RadixCache` continuously during the active generation phase, rather than waiting until the entire sequence finishes in `ProcessLoop._finish_sequence`. 
    * *Partial Blocks*: Introduce a `partial=True` flag to allow sharing of half-filled active decode blocks. 
    * *CoW Requirement*: Must implement Copy-on-Write. If a sequence attempts to append a token to a shared `partial` block (ref_count > 1), the allocator must fork the block to a new memory address to prevent race conditions and silent GPU memory corruption. 