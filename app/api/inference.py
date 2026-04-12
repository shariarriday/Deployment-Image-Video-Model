"""API endpoints for inference requests"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import os
import logging
from typing import Optional

from app.db.database import get_db
from app.models.inference import InferenceRequest
from app.schemas.inference import InferenceResponse, InferenceResult
from app.services.queue import get_inference_queue
from app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


def validate_file_extension(filename: str) -> str:
    """Validate file extension and return file type"""
    ext = Path(filename).suffix.lower()
    
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Determine if it's image or video
    image_extensions = [".jpg", ".jpeg", ".png"]
    video_extensions = [".mp4", ".avi", ".mov"]
    
    if ext in image_extensions:
        return "image"
    elif ext in video_extensions:
        return "video"
    else:
        return "unknown"


@router.post("/inference", response_model=InferenceResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_inference(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Submit an image or video file for inference.
    
    The file will be queued for processing and an ID will be returned.
    Use the /inference/{id} endpoint to check the status and get results.
    
    Args:
        file: Image or video file to process
        
    Returns:
        InferenceResponse with request ID and status
    """
    try:
        # Validate file
        file_type = validate_file_extension(file.filename)
        
        # Generate unique ID and filename
        request_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        unique_filename = f"{request_id}{file_extension}"
        
        # Save file
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / unique_filename
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # Create database record
        request = InferenceRequest(
            id=request_id,
            file_path=str(file_path),
            file_type=file_type,
            status="pending"
        )
        db.add(request)
        db.commit()
        
        # Add to processing queue
        queue = get_inference_queue()
        await queue.add_request(request_id)
        
        return InferenceResponse(
            id=request_id,
            status="pending",
            message="Request received and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting inference request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@router.get("/inference/{request_id}", response_model=InferenceResult)
async def get_inference_result(
    request_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the status and result of an inference request.
    
    Args:
        request_id: ID of the inference request
        
    Returns:
        InferenceResult with status and results (if completed)
    """
    request = db.query(InferenceRequest).filter(
        InferenceRequest.id == request_id
    ).first()
    
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found"
        )
    
    return InferenceResult(
        id=request.id,
        file_type=request.file_type,
        status=request.status,
        result=request.result,
        error=request.error,
        confidence=request.confidence,
        processing_time=request.processing_time,
        created_at=request.created_at,
        updated_at=request.updated_at
    )


@router.get("/inference")
async def list_inference_requests(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all inference requests with optional filtering.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status_filter: Filter by status (pending, processing, completed, failed)
        
    Returns:
        List of inference requests
    """
    query = db.query(InferenceRequest)
    
    if status_filter:
        query = query.filter(InferenceRequest.status == status_filter)
    
    requests = query.offset(skip).limit(limit).all()
    return [req.to_dict() for req in requests]


@router.delete("/inference/{request_id}")
async def delete_inference_request(
    request_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an inference request and its associated file.
    
    Args:
        request_id: ID of the inference request to delete
        
    Returns:
        Success message
    """
    request = db.query(InferenceRequest).filter(
        InferenceRequest.id == request_id
    ).first()
    
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found"
        )
    
    # Delete file if it exists
    try:
        if os.path.exists(request.file_path):
            os.remove(request.file_path)
            logger.info(f"Deleted file: {request.file_path}")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
    
    # Delete database record
    db.delete(request)
    db.commit()
    
    return {"message": f"Request {request_id} deleted successfully"}
