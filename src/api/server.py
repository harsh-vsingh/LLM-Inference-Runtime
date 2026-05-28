from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from models.loader import get_model_loader
from engine.async_engine import AsyncInferenceEngine

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes.*")

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    loader = get_model_loader(model_name, use_quantization=True)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    engine = AsyncInferenceEngine(model, tokenizer)
    engine.start()
    
    app.state.engine = engine
    
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)