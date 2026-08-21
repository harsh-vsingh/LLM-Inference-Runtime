from typing import List
import torch
from engine.sequence import Sequence


def sample(logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
    logits = logits.float()

    temps = torch.tensor(
        [getattr(seq.request, "temperature", 0.0) for seq in seqs],
        device=logits.device, dtype=logits.dtype,
    )
    top_ps = [getattr(seq.request, "top_p", 1.0) for seq in seqs]

    greedy_mask = temps <= 1e-6

    greedy_tokens = torch.argmax(logits, dim=-1)

    safe_temps = torch.where(greedy_mask, torch.ones_like(temps), temps)
    scaled = logits / safe_temps.unsqueeze(-1)
    probs = torch.softmax(scaled, dim=-1)

    for i, p in enumerate(top_ps):
        if greedy_mask[i] or p >= 1.0:
            continue
        sorted_probs, sorted_idx = torch.sort(probs[i], descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        probs[i, sorted_idx[remove]] = 0.0

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    final_tokens = torch.where(greedy_mask, greedy_tokens, sampled_tokens)

    return final_tokens.tolist()