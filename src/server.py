from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from services.engine_registry import create_engine_registry
from core.config import settings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes.*")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine_registry = create_engine_registry(settings.model_name, use_quantization=settings.use_quantization)
    yield
    await app.state.engine_registry.shutdown(timeout=30.0)

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