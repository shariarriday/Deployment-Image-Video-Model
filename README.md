# Model Inference API

Production-ready FastAPI service for image and video inference with asynchronous queue-based processing.

## Features

- Async processing queue for concurrent inference requests
- Flexible model integration path
- REST API for submit, fetch, list, and delete operations
- SQLite request tracking
- Docker and docker-compose support
- Automatic model weight download from Google Cloud Storage
- Command-based external inference execution support

## Project Structure

```text
Model-Deployment/
|-- app/
|   |-- api/
|   |   |-- inference.py
|   |   `-- health.py
|   |-- core/
|   |   |-- config.py
|   |   |-- file_uploader.py
|   |   `-- model_downloader.py
|   |-- db/
|   |   `-- database.py
|   |-- models/
|   |   `-- inference.py
|   |-- schemas/
|   |   `-- inference.py
|   `-- services/
|       `-- queue.py
|-- model_weights/
|-- uploads/
|-- model_weights.json
|-- docker-compose.yml
|-- Dockerfile
|-- main.py
|-- requirements.txt
`-- .env.example
```

## Quick Start

### Docker

1. Clone and configure:

```bash
cd Model-Deployment
cp .env.example .env
```

1. Update `.env` and `model_weights.json`, then place your local service account key at the path set in `GCS_SERVICE_ACCOUNT_FILE`.

1. Build and run:

```bash
docker-compose up --build
```

1. Access:

- API docs: [http://localhost:9876/docs](http://localhost:9876/docs)
- Health: [http://localhost:9876/api/v1/health](http://localhost:9876/api/v1/health)

### Local Development

1. Install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate
pip install -r requirements.txt
```

1. Configure environment:

```bash
cp .env.example .env
```

1. Run:

```bash
python main.py
```

## API Endpoints

- `POST /api/v1/inference`
- `GET /api/v1/inference/{request_id}`
- `GET /api/v1/inference?status_filter=completed&limit=10`
- `DELETE /api/v1/inference/{request_id}`
- `GET /api/v1/health`

## Google Cloud Configuration

Set these values in `.env`:

```bash
MODEL_WEIGHTS_DIR=./model_weights
MODEL_WEIGHTS_LIST_FILE=./model_weights.json
GCS_SERVICE_ACCOUNT_FILE=./service-account.json
GCS_BUCKET_NAME=your-real-bucket-name
INFERENCE_COMMAND=your-processing-command
INFERENCE_RESULT_FILE=result.json
```

### model_weights.json format

Supported examples:

```json
[
  "weights/best_model.pth",
  "weights/label_maps.json"
]
```

```json
[
  "gs://your-real-bucket-name/weights/best_model.pth",
  "gs://your-real-bucket-name/weights/label_maps.json"
]
```

```json
[
  {
    "file_name": "weights/best_model.pth",
    "local_name": "best_model.pth"
  }
]
```

## Deployment Steps

1. Prepare GCS bucket(s):

- Create bucket(s) for model weights and optional processed outputs.
- Upload real model artifacts.
- Verify object paths used in `model_weights.json`.

1. Prepare service account:

- Create a dedicated service account for this application.
- Grant least-privilege permissions.
- Download key JSON and place it at `GCS_SERVICE_ACCOUNT_FILE`.
- Create a local `service-account.json` (or another untracked path) and set `GCS_SERVICE_ACCOUNT_FILE` accordingly.

1. Configure runtime values:

- Set `GCS_BUCKET_NAME` to your real bucket.
- Set `INFERENCE_COMMAND` to your processing command.
- Ensure `INFERENCE_RESULT_FILE` matches command output.

1. Deploy:

```bash
docker-compose up --build -d
```

1. Validate:

```bash
curl http://localhost:9876/api/v1/health
docker-compose logs -f model-api
```

## Security Implications

- Never commit `service-account.json` to source control.
- Prefer secret managers or runtime secret injection over static key files.
- Use least-privilege IAM roles only.
- Restrict bucket access to required principals.
- Rotate service-account keys regularly and revoke unused keys.
- Keep GCS access and audit logs enabled.
- Separate buckets by purpose when possible.
- Validate uploaded files and command outputs before use.

## Troubleshooting

### Health endpoint returns 404

- Ensure the app is running latest code where health route is `/api/v1/health`.
- Rebuild and restart containers:

```bash
docker-compose up --build
```

### Model download fails with 404

- Check object paths in `model_weights.json`.
- If using `gs://...` entries, ensure bucket/object exists.
- If using relative entries, ensure `GCS_BUCKET_NAME` is correct.

### Inference command fails

- Verify `INFERENCE_COMMAND` writes valid JSON to `INFERENCE_RESULT_FILE`.
- Confirm command placeholders are correct: `{input_file}`, `{work_dir}`, `{result_json}`, `{request_id}`.

## Docker Commands

```bash
docker-compose build
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## License

MIT
