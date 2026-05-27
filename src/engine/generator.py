import torch
from typing import Generator

@torch.inference_mode()
def generate_tokens(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> Generator[str, None, None]:

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)

    eos_token_id = tokenizer.eos_token_id

    outputs = model(input_ids=input_ids, use_cache=True)

    past_key_values = outputs.past_key_values

    generated_ids = []
    prev_text = ""

    for _ in range(max_new_tokens):

        next_token_id = torch.argmax(outputs.logits[0, -1, :], dim=-1)

        if next_token_id.item() == eos_token_id:
            break

        generated_ids.append(next_token_id.item())

        full_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        new_text = full_text[len(prev_text):]

        yield new_text

        prev_text = full_text

        outputs = model(
            input_ids=next_token_id.unsqueeze(0).unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True
        )

        past_key_values = outputs.past_key_values