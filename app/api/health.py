"""Health check and monitoring endpoints"""
from fastapi import APIRouter
from app.schemas.inference import HealthResponse
from app.core.config import get_settings
from app.services.queue import get_inference_queue

router = APIRouter()
settings = get_settings()


@router.get(f"{settings.API_V1_PREFIX}/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and model readiness.
    """
    try:
        queue = get_inference_queue()
        model_loaded = queue.model is not None
    except Exception:
        model_loaded = False
    
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        model_loaded=model_loaded
    )


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }
