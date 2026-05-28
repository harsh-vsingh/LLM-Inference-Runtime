import asyncio
import torch
from typing import Optional
from engine.request import InferenceRequest
from engine.sequence import Sequence
from engine.scheduler import Scheduler

class AsyncInferenceEngine:
    def __init__(self, model, tokenizer, max_batch_size: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.wakeup_event = asyncio.Event()
        self.background_task: Optional[asyncio.Task] = None

    def start(self):
        self.background_task = asyncio.create_task(self._process_loop())

    async def add_request(self, request: InferenceRequest):
        prompt_ids = self.tokenizer(request.prompt).input_ids
        
        seq = Sequence(
            request=request,
            prompt_token_ids=prompt_ids,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        seq.prev_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
        self.scheduler.add_sequence(seq)
        self.wakeup_event.set()

    async def _process_loop(self):
        while True:
            await self.wakeup_event.wait()
            
            while self.scheduler.has_unfinished_sequences():
                step_outputs = await asyncio.to_thread(self._pytorch_step)
                
                for seq, new_text in step_outputs:
                    await seq.request.put_token(new_text)
                    
                    if seq.is_finished():
                        await seq.request.finish()
                        
                await asyncio.sleep(0)
                
            self.wakeup_event.clear()

    @torch.inference_mode()
    def _pytorch_step(self):
        prefill_seqs, decode_seqs = self.scheduler.step()
        outputs = []

        for seq in prefill_seqs:
            input_ids = torch.tensor([seq.prompt_token_ids], device=self.model.device)
            
            out = self.model(input_ids=input_ids, use_cache=True)
            next_token_id = int(torch.argmax(out.logits[0, -1, :], dim=-1).item())
            
            seq.past_key_values = out.past_key_values
            seq.generated_token_ids.append(next_token_id)
            
            full_text = self.tokenizer.decode(
                seq.prompt_token_ids + seq.generated_token_ids, 
                skip_special_tokens=True
            )
            new_text = full_text[len(seq.prev_text):]
            seq.prev_text = full_text
            
            outputs.append((seq, new_text))

        for seq in decode_seqs:
            input_ids = torch.tensor([[seq.generated_token_ids[-1]]], device=self.model.device)
            
            out = self.model(
                input_ids=input_ids, 
                past_key_values=seq.past_key_values, 
                use_cache=True
            )
            next_token_id = int(torch.argmax(out.logits[0, -1, :], dim=-1).item())
            
            seq.past_key_values = out.past_key_values
            seq.generated_token_ids.append(next_token_id)
            
            full_text = self.tokenizer.decode(
                seq.prompt_token_ids + seq.generated_token_ids, 
                skip_special_tokens=True
            )
            new_text = full_text[len(seq.prev_text):]
            seq.prev_text = full_text
            
            outputs.append((seq, new_text))

        return outputs