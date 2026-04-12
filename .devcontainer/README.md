# DevContainer Configuration

This directory contains the VS Code DevContainer configuration for the Model Inference API project.

## What is a DevContainer?

A DevContainer allows you to develop inside a Docker container with all dependencies pre-installed and configured. This ensures:

- **Consistent environment** across all developers
- **No local setup required** - everything runs in Docker
- **Isolated development** - no conflicts with local Python installations
- **Pre-configured tools** - linters, formatters, and extensions ready to use

## Getting Started

### Prerequisites

1. **Install Docker Desktop**
   - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - Linux: [Docker Engine](https://docs.docker.com/engine/install/)

2. **Install VS Code**
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)

3. **Install Dev Containers Extension**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Dev Containers"
   - Install the extension by Microsoft

### Opening the Project in DevContainer

1. **Open the project folder in VS Code**

  ```powershell
   code d:\Codes\Model-Deployment
   ```

1. **Reopen in Container**
   - Press `F1` or `Ctrl+Shift+P`
   - Type: "Dev Containers: Reopen in Container"
   - Select it and wait for the container to build

1. **Wait for setup to complete**
   - First time will take several minutes (downloading images, installing dependencies)
   - Subsequent opens will be much faster

## What's Included

### Extensions

The devcontainer automatically installs:

- **Python Development**
  - Python extension
  - Pylance (IntelliSense)
  - Black formatter
  - isort (import sorting)
  - Debugger

- **API Development**
  - REST Client
  - OpenAPI support

- **Docker & Git**
  - Docker extension
  - GitLens

- **Productivity**
  - Error Lens
  - TODO Tree
  - Material Icon Theme
  - Code Spell Checker

### Development Tools

Pre-installed in the container:

```bash
# Testing
pytest
pytest-asyncio
pytest-cov

# Code quality
black
isort
flake8
mypy

# Debugging
ipython
ipdb

# API testing
httpx
```

## Usage

### Running the Application

Once inside the devcontainer:

```bash
# Run the FastAPI server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be accessible at <http://localhost:8000>

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_inference.py
```

### Code Formatting

```bash
# Format with Black
black .

# Sort imports
isort .

# Lint with flake8
flake8 app/

# Type checking
mypy app/
```

### Debugging

1. Set breakpoints in your code (click left of line numbers)
2. Press `F5` or go to Run and Debug panel
3. Select "Python: FastAPI" configuration
4. The debugger will start and stop at breakpoints

## Configuration

The devcontainer uses the project's main `Dockerfile` to ensure development and production environments are identical. Additional development tools (pytest, black, mypy, etc.) are installed via the `postCreateCommand` hook without modifying the base image.

## Environment Variables

The devcontainer sets `DEBUG=True` automatically. To modify environment variables:

1. Edit `.env` file in the project root
2. Or modify `containerEnv` in `devcontainer.json`
3. Rebuild container: `F1` → "Dev Containers: Rebuild Container"

For model weight download from Google Cloud Storage, ensure `.env` includes:

```bash
MODEL_WEIGHTS_DIR=./model_weights
MODEL_WEIGHTS_LIST_FILE=./model_weights.json
GCS_SERVICE_ACCOUNT_FILE=./service-account.json
GCS_BUCKET_NAME=your-gcs-bucket-name
```

Also ensure these files exist in the project root:

- `model_weights.json` (list of GCS blob paths to download)
- `service-account.json` (Google service account key with storage read access)

## Deployment Notes

Before deploying from a devcontainer workflow, verify these items:

1. Set the correct bucket name in `GCS_BUCKET_NAME`.
2. Upload real model weights to that bucket and ensure `model_weights.json` points to valid objects.
3. Replace placeholder values in `service-account.json` with a real Google Cloud service account key.
4. Confirm the service account has least-privilege permissions required by the app.

## Security Notes

- Do not commit `service-account.json` to source control.
- Store credentials with a secret manager or CI/CD secret store where possible.
- Rotate keys on a schedule and remove unused keys immediately.
- Limit bucket access to only the service account used by this service.
- Avoid broad IAM roles and use object-level access scopes where possible.

## Troubleshooting

### Container won't start

```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs

# Rebuild container
# In VS Code: F1 → "Dev Containers: Rebuild Container"
```

### Port already in use

If port 8000 is already in use:

1. Edit `devcontainer.json` and change `forwardPorts`
2. Update the port in your application
3. Rebuild container

### Python packages not found

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or rebuild container
# F1 → "Dev Containers: Rebuild Container"
```

### Changes not reflected

Make sure you're editing files in `/app` inside the container. The workspace is mounted, so changes should sync automatically.

## Tips

1. **Terminal in Container**: Use VS Code's integrated terminal (Ctrl+`) - it runs inside the container

2. **Install Extensions**: Any extensions installed while in devcontainer are container-specific

3. **Git**: Git commands work normally - your local git config is used

4. **Performance**: On Windows, use WSL 2 backend for Docker for better performance

5. **Ports**: The devcontainer forwards port 8000 automatically. Access APIs at <http://localhost:8000>

## Advanced Customization

### Add more VS Code extensions

Edit `devcontainer.json`:

```json
"customizations": {
  "vscode": {
    "extensions": [
      "existing.extensions",
      "your.new-extension"
    ]
  }
}
```

### Add system packages

Edit `Dockerfile.dev`:

```dockerfile
RUN apt-get update && apt-get install -y \
    your-package \
    && rm -rf /var/lib/apt/lists/*
```

### Modify Python settings

Edit the `settings` section in `devcontainer.json`.

## More Information

- [VS Code DevContainers Docs](https://code.visualstudio.com/docs/devcontainers/containers)
- [DevContainer Spec](https://containers.dev/)
- [DevContainer Features](https://containers.dev/features)
