"""Database models for inference requests and results"""
from sqlalchemy import Column, String, DateTime, Text, Float
from sqlalchemy.sql import func
from app.db.database import Base
import uuid


class InferenceRequest(Base):
    """Model for storing inference requests"""
    __tablename__ = "inference_requests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # 'image' or 'video'
    status = Column(String, default="pending")  # pending, processing, completed, failed
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
