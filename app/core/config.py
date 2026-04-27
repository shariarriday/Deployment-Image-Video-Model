"""Application configuration settings"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # App settings
    APP_NAME: str = "Model Inference API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    CORS_ALLOWED_ORIGINS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = False
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./app.db"
    
    # Model settings
    MODEL_WEIGHTS_DIR: str = "./model_weights"
    MODEL_WEIGHTS_LIST_FILE: str = "./model_weights.json"
    GCS_SERVICE_ACCOUNT_FILE: str = "./service-account.json"
    GCS_BUCKET_NAME: str = "inference-exercise"
    
    # Upload settings
    UPLOAD_DIR: str = "./uploads"
    
    # Queue settings
    MAX_QUEUE_SIZE: int = 1000
    WORKER_COUNT: int = 2
    INFERENCE_COMMAND: str = "python /app/model/infer_yolo26.py \
                                --weights $WEIGHTS_PATH_YOLO \
                                --source $INPUT_PATH \
                                --output $OUTPUT_PATH && \
                                python /app/model/Inference_test.py \
                                    --weights $WEIGHTS_PATH_CLASSIFIER \
                                    --source $OUTPUT_PATH/result.jpg \
                                    --output $OUTPUT_PATH/result.json"
    INFERENCE_TEMP_DIR: str = "/tmp_inference"
    INFERENCE_RESULT_FILE: str = "result.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
