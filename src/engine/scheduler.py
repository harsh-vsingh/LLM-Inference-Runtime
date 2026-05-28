from typing import List, Tuple
from engine.sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self, max_batch_size: int = 4):
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.max_batch_size = max_batch_size

    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)

    def step(self) -> Tuple[List[Sequence], List[Sequence]]:
        finished_seqs = [seq for seq in self.running if seq.is_finished()]
        for seq in finished_seqs:
            seq.status = SequenceStatus.FINISHED
            self.running.remove(seq)

        available_slots = self.max_batch_size - len(self.running)
        
        prefill_seqs = []
        while available_slots > 0 and self.waiting:
            seq = self.waiting.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seqs.append(seq)
            available_slots -= 1

        decode_seqs = [seq for seq in self.running if seq not in prefill_seqs]

        return prefill_seqs, decode_seqs

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running)