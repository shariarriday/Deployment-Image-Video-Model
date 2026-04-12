"""Utility functions for uploading files to Google Cloud Storage."""
import logging
import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def upload_file_to_google_cloud(
	service_account_file: str,
	bucket_name: str,
	source_file_path: str,
	destination_blob_name: str | None = None,
) -> str | None:
	"""
	Upload a local file to Google Cloud Storage.

	Args:
		service_account_file: Path to service account JSON key.
		bucket_name: Target GCS bucket.
		source_file_path: Local file path to upload.
		destination_blob_name: Optional destination object name in GCS.
			If omitted, the source filename is used.

	Returns:
		The uploaded gs:// URL if successful, otherwise None.
	"""
	try:
		if not os.path.exists(source_file_path):
			logger.error("Upload source file does not exist: %s", source_file_path)
			return None

		credentials = service_account.Credentials.from_service_account_file(
			service_account_file
		)
		client = storage.Client(credentials=credentials, project=credentials.project_id)

		blob_name = destination_blob_name or Path(source_file_path).name

		bucket = client.bucket(bucket_name)
		blob = bucket.blob(blob_name)
		blob.upload_from_filename(source_file_path)

		gs_url = f"gs://{bucket_name}/{blob_name}"
		logger.info("Uploaded %s to %s", source_file_path, gs_url)
		return gs_url
	except Exception as exc:
		logger.error(
			"Failed to upload %s to bucket %s: %s",
			source_file_path,
			bucket_name,
			exc,
		)
		return None
