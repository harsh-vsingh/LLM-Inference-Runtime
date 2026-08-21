import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self, model_name: str, use_quantization: bool = False, torch_dtype: torch.dtype = torch.float16):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        if self.model is not None:
            raise RuntimeError(
                f"ModelLoader for '{self.model_name}' has already loaded a model. "
                "Create a new ModelLoader instance to load again."
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
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
                    torch_dtype=self.torch_dtype,
                    device_map="auto"
                )
        except Exception:
            logger.exception(f"Failed to load model/tokenizer '{self.model_name}'")
            raise RuntimeError(
                f"Could not load model '{self.model_name}'. "
                "Check the model name is correct and accessible (auth/network)."
            )

        logger.info(f"Loaded model '{self.model_name}' (quantized={self.use_quantization}, dtype={self.torch_dtype})")

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer


def get_model_loader(model_name: str, use_quantization: bool = False, torch_dtype: torch.dtype = torch.float16) -> ModelLoader:
    use_quantization = False
    model_loader = ModelLoader(model_name, use_quantization, torch_dtype)
    model_loader.load_model()
    return model_loader