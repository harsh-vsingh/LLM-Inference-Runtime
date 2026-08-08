from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_quantization: bool = True
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096

    class Config:
        env_prefix = "OPTISERVE_"


settings = Settings()