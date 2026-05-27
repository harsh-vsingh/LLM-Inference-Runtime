from src.models.loader import get_model_loader
from src.engine.generator import generate_tokens

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    loader = get_model_loader(model_name, use_quantization=True)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    prompt = "<|system|>\nYou are a helpful AI.\n<|user|>\nWho is ceo of google.\n<|assistant|>\n"
    
    print("\nStarting generation...\n")
    for token in generate_tokens(model, tokenizer, prompt, max_new_tokens=100):
        print(token, end="", flush=True)
    print("\n\nFinished.")

if __name__ == "__main__":
    main()