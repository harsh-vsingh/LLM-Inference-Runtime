from typing import List, Tuple
from engine.sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self, max_num_batched_tokens: int = 4096, max_num_seqs: int = 256):
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs

    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)

    def step(self) -> Tuple[List[Sequence], List[Sequence]]:
        finished_seqs = [seq for seq in self.running if seq.is_finished()]
        for seq in finished_seqs:
            seq.status = SequenceStatus.FINISHED
            self.running.remove(seq)

        num_batched_tokens = sum(1 for seq in self.running)
        
        prefill_seqs = []
        
        while self.waiting:
            if len(self.running) >= self.max_num_seqs:
                break
                
            next_seq = self.waiting[0]
            seq_num_tokens = next_seq.get_num_uncomputed_tokens()
            
            if num_batched_tokens + seq_num_tokens > self.max_num_batched_tokens and len(self.running) > 0:
                break
                
            seq = self.waiting.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)
            
            num_batched_tokens += seq_num_tokens

        decode_seqs = [seq for seq in self.running if seq not in prefill_seqs]

        return prefill_seqs, decode_seqs

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running)