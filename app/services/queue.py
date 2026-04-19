"""Queue management and background processing"""
import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Optional
from app.models.inference import InferenceRequest
from app.core.config import get_settings
from app.core.file_uploader import upload_file_to_google_cloud
import time

logger = logging.getLogger(__name__)
settings = get_settings()


class InferenceQueue:
    """Manages the inference request queue and background workers"""
    
    def __init__(self, model: Optional[Any] = None):
        """
        Initialize the inference queue.
        
        Args:
            model: Optional in-process inference model for fallback execution
        """
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=settings.MAX_QUEUE_SIZE)
        self.model = model
        self.workers = []
        self.is_running = False
        self.command_template = getattr(settings, "INFERENCE_COMMAND", "").strip()
        self.temp_root_dir = getattr(settings, "INFERENCE_TEMP_DIR", "./tmp_inference")
        self.weight_path = getattr(settings, "MODEL_WEIGHTS_DIR", "./model_weights")

    def _run_external_command(self, source_file_path: str, request_id: str) -> dict[str, Any]:
        """Copy input file to a temp directory, run command, then read result JSON."""
        if not self.command_template:
            raise ValueError("INFERENCE_COMMAND is not configured")

        Path(self.temp_root_dir).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"{request_id}_", dir=self.temp_root_dir) as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_input_path = temp_dir_path / Path(source_file_path).name
            print(f"Copying {source_file_path} to {temp_input_path}")
            print(temp_dir_path, temp_input_path)
            shutil.copy2(source_file_path, temp_input_path)

            temp_weight_path = Path(self.weight_path) / "best_yolo26m.pt"

            command = self.command_template
            command = command.replace("$WEIGHTS_PATH", str(temp_weight_path))
            command = command.replace("$INPUT_PATH", str(temp_input_path))
            command = command.replace("$OUTPUT_PATH", str(self.temp_root_dir))

            completed = subprocess.run(
                command,
                cwd=str(temp_dir_path),
                shell=True,
                text=True,
                capture_output=True,
            )

            result_path = Path(self.temp_root_dir) / "results" / f"{request_id}.jpg"

            if completed.returncode != 0:
                raise RuntimeError(
                    f"Inference command failed with code {completed.returncode}. "
                    f"stderr: {completed.stderr.strip()}"
                )

            if not result_path.exists():
                raise FileNotFoundError(
                    f"Expected result file not found: {result_path}"
                )

            result_payload = {"result_file": str(result_path)}

            if not isinstance(result_payload, dict):
                raise ValueError("result.json must contain a JSON object")

            return result_payload

    def _run_model_inference(self, request: InferenceRequest) -> dict[str, Any]:
        """Fallback inference path using in-process model implementation."""
        if not self.model:
            raise ValueError("No model instance provided for fallback inference")

        if request.file_type == "image":
            return self.model.process_image(request.file_path)
        if request.file_type == "video":
            return self.model.process_video(request.file_path)

        raise ValueError(f"Unknown file type: {request.file_type}")
        
    async def start_workers(self, db_session_factory):
        """Start background worker tasks"""
        self.is_running = True
        self.db_session_factory = db_session_factory
        
        logger.info(f"Starting {settings.WORKER_COUNT} workers...")
        for i in range(settings.WORKER_COUNT):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        logger.info("Workers started successfully")
    
    async def stop_workers(self):
        """Stop all background workers"""
        logger.info("Stopping workers...")
        self.is_running = False
        
        # Wait for all workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Workers stopped")
    
    async def add_request(self, request_id: str) -> None:
        """
        Add a request to the processing queue.
        
        Args:
            request_id: ID of the inference request to process
        """
        try:
            await self.queue.put(request_id)
            logger.info(f"Request {request_id} added to queue. Queue size: {self.queue.qsize()}")
        except asyncio.QueueFull:
            logger.error(f"Queue is full. Cannot add request {request_id}")
            raise Exception("Queue is full. Please try again later.")
    
    async def _worker(self, worker_id: int):
        """
        Background worker that processes requests from the queue.
        
        Args:
            worker_id: Unique identifier for this worker
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get request from queue with timeout
                request_id = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=1.0
                )
                
                logger.info(f"Worker {worker_id} processing request {request_id}")
                await self._process_request(request_id)
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                # No items in queue, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_request(self, request_id: str):
        """
        Process a single inference request.
        
        Args:
            request_id: ID of the request to process
        """
        db = self.db_session_factory()
        request: Optional[InferenceRequest] = None
        
        try:
            # Get request from database
            request = db.query(InferenceRequest).filter(
                InferenceRequest.id == request_id
            ).first()
            
            if not request:
                logger.error(f"Request {request_id} not found in database")
                return
            
            # Update status to processing
            request.status = "processing"
            db.commit()
            
            # Run inference
            start_time = time.time()

            if self.command_template:
                result = await asyncio.to_thread(
                    self._run_external_command,
                    request.file_path,
                    request_id,
                )
            else:
                result = await asyncio.to_thread(self._run_model_inference, request)

            uploaded_file_url: Optional[str] = None
            if settings.GCS_BUCKET_NAME and os.path.exists(settings.GCS_SERVICE_ACCOUNT_FILE):
                uploaded_file_url = upload_file_to_google_cloud(
                    service_account_file=settings.GCS_SERVICE_ACCOUNT_FILE,
                    bucket_name=settings.GCS_BUCKET_NAME,
                    source_file_path=request.file_path,
                    destination_blob_name=f"processed/{Path(request.file_path).name}",
                )
                if not uploaded_file_url:
                    raise RuntimeError("Processed file upload failed")
            else:
                logger.info(
                    "Skipping GCS upload for request %s because bucket or service account is not configured",
                    request_id,
                )

            try:
                if os.path.exists(request.file_path):
                    os.remove(request.file_path)
            except Exception as cleanup_exc:
                logger.warning(
                    "Uploaded file but failed to remove local file %s: %s",
                    request.file_path,
                    cleanup_exc,
                )

            if uploaded_file_url:
                result["uploaded_file_url"] = uploaded_file_url
            
            processing_time = time.time() - start_time
            
            # Update request with results
            request.status = "completed"
            request.result = result.get("result_file")

            confidence_value = result.get("confidence")
            if isinstance(confidence_value, (int, float)):
                request.confidence = float(confidence_value)

            request.processing_time = processing_time
            db.commit()
            
            logger.info(f"Request {request_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}", exc_info=True)
            
            # Update request with error
            if request is not None:
                request.status = "failed"
                request.error = str(e)
                db.commit()
            
        finally:
            db.close()


# Global queue instance
inference_queue: Optional[InferenceQueue] = None


def get_inference_queue() -> InferenceQueue:
    """Get the global inference queue instance"""
    global inference_queue
    if inference_queue is None:
        raise Exception("Inference queue not initialized")
    return inference_queue


def set_inference_queue(queue: InferenceQueue):
    """Set the global inference queue instance"""
    global inference_queue
    inference_queue = queue
