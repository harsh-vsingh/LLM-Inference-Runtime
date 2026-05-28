from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelLoader:
    def __init__(self, model_name: str, use_quantization: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer

def get_model_loader(model_name: str, use_quantization: bool = True) -> ModelLoader:
    model_loader = ModelLoader(model_name, use_quantization)
    model_loader.load_model()
    return model_loader

loader = get_model_loader("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_quantization=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()
print(model.config._attn_implementation)