from fastapi import FastAPI
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI()

    from api.routes import router as api_router
    app.include_router(api_router)

    return app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)