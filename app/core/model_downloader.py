"""Utility functions for downloading model weights from Google Cloud Storage."""
import json
import logging
import os
from pathlib import Path
from typing import Any

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def download_from_google_cloud(
    service_account_file: str,
    bucket_name: str,
    file_name: str,
    destination_path: str,
) -> bool:
    """Download a single file from GCS using a gs:// bucket URL."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file
        )
        client = storage.Client(credentials=credentials, project=credentials.project_id)

        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

        if file_name.startswith("gs://"):
            gsutil_url = file_name
        else:
            if not bucket_name:
                raise ValueError("bucket_name is required when file_name is not a gs:// URL")
            gsutil_url = f"gs://{bucket_name}/{file_name.lstrip('/')}"

        blob = storage.Blob.from_string(gsutil_url, client=client)
        blob.download_to_filename(destination_path)

        logger.info("Downloaded %s to %s", gsutil_url, destination_path)
        return True
    except Exception as exc:
        logger.error("Failed to download %s: %s", file_name, exc)
        return False


def download_bulk_file(
    weights_json_path: str,
    service_account_file: str,
    bucket_name: str,
    destination_dir: str = "./model_weights",
) -> bool:
    """
    Download model weights listed in a JSON file into the model_weights directory.

    Supported JSON formats:
    - ["path/in/bucket/file1.pt", "path/in/bucket/file2.pt"]
    - {"weights": ["path/in/bucket/file1.pt"]}
    - [{"file_name": "path/in/bucket/file1.pt", "local_name": "file1.pt"}]
    """
    if not os.path.exists(weights_json_path):
        logger.warning("Weights JSON file not found: %s", weights_json_path)
        return True

    try:
        with open(weights_json_path, "r", encoding="utf-8") as file:
            payload: Any = json.load(file)
    except Exception as exc:
        logger.error("Failed to read weights JSON %s: %s", weights_json_path, exc)
        return False

    weights_list: list[Any]
    if isinstance(payload, list):
        weights_list = payload
    elif isinstance(payload, dict) and isinstance(payload.get("weights"), list):
        weights_list = payload["weights"]
    else:
        logger.error("Invalid weights JSON format in %s", weights_json_path)
        return False

    Path(destination_dir).mkdir(parents=True, exist_ok=True)

    all_success = True
    for item in weights_list:
        remote_name: str | None = None
        local_name: str | None = None

        if isinstance(item, str):
            remote_name = item
            local_name = os.path.basename(item)
        elif isinstance(item, dict):
            remote_name = item.get("file_name") or item.get("blob_name") or item.get("gcs_path")
            local_name = item.get("local_name") or (os.path.basename(remote_name) if remote_name else None)

        if not remote_name or not local_name:
            logger.warning("Skipping invalid weight entry: %s", item)
            all_success = False
            continue

        if not bucket_name and not remote_name.startswith("gs://"):
            logger.warning(
                "Skipping weight entry without bucket configuration: %s",
                remote_name,
            )
            all_success = False
            continue

        destination_path = os.path.join(destination_dir, local_name)
        if os.path.exists(destination_path):
            logger.info("Weight file already exists, skipping: %s", destination_path)
            continue

        if not download_from_google_cloud(
            service_account_file=service_account_file,
            bucket_name=bucket_name,
            file_name=remote_name,
            destination_path=destination_path,
        ):
            all_success = False

    return all_success
