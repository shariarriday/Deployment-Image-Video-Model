"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class InferenceResponse(BaseModel):
    """Response schema for inference submission"""
    id: str
    status: str
    message: str = "Request received and queued for processing"


class InferenceResult(BaseModel):
    """Schema for inference result"""
    id: str
    file_type: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    model_loaded: bool
