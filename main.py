"""Main FastAPI application"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.model_downloader import download_bulk_file
from app.db.database import init_db, SessionLocal
from app.api import inference, health
from app.services.queue import InferenceQueue, set_inference_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for application startup and shutdown.
    Handles model loading and queue management.
    """
    logger.info("Starting application...")
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Download model weights on startup
    logger.info("Checking model weights...")
    download_success = download_bulk_file(
        settings.MODEL_WEIGHTS_LIST_FILE,
        settings.GCS_SERVICE_ACCOUNT_FILE,
        settings.GCS_BUCKET_NAME,
        settings.MODEL_WEIGHTS_DIR
    )
    
    if not download_success:
        logger.warning("Some model downloads failed. Proceeding anyway...")

    inference_model = None
    if not settings.INFERENCE_COMMAND:
        logger.info("Loading inference model...")
        try:
            from app.services.inference import DummyInferenceModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "INFERENCE_COMMAND is empty and app.services.inference is not available. "
                "Configure INFERENCE_COMMAND or restore app/services/inference.py."
            ) from exc

        inference_model = DummyInferenceModel(settings.MODEL_WEIGHTS_DIR)
        inference_model.load_model()
    else:
        logger.info("Using external inference command execution")
    
    # Initialize and start inference queue
    logger.info("Starting inference queue...")
    queue = InferenceQueue(model=inference_model)
    set_inference_queue(queue)
    await queue.start_workers(SessionLocal)
    
    logger.info("Application startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await queue.stop_workers()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="FastAPI service for PyTorch model inference with async processing",
    lifespan=lifespan
)

# CORS middleware
allow_origins = settings.CORS_ALLOWED_ORIGINS or (["*"] if settings.DEBUG else [])
allow_credentials = settings.CORS_ALLOW_CREDENTIALS and allow_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    inference.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Inference"]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
