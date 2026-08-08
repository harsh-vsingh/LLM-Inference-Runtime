from typing import List

import torch

from engine.sequence import Sequence


def sample(logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
    next_tokens = []

    for i, seq in enumerate(seqs):
        temp = getattr(seq.request, "temperature", 0.0)
        top_p = getattr(seq.request, "top_p", 1.0)

        logit = logits[i].float()

        if temp <= 1e-6:
            next_tokens.append(int(torch.argmax(logit).item()))
            continue

        logit = logit / temp
        probs = torch.softmax(logit, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logit[indices_to_remove] = -float("inf")
            probs = torch.softmax(logit, dim=-1)

        token_id = torch.multinomial(probs, num_samples=1).item()
        next_tokens.append(int(token_id))

    return next_tokens