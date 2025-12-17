FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps (kept minimal; helps with scientific/python wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps (robust: use requirements.txt if present, else install minimal runtime deps)
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
 && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; \
    else pip install --no-cache-dir fastapi uvicorn joblib numpy pandas scikit-learn lightgbm prophet requests prefect; fi

# Copy the ENTIRE mlops-project folder (app/, training/, prefect/, models/, data/, docs, etc.)
COPY . .

# Build-time sanity check: fail the build if the full project layout isn't present
RUN test -d /app/app \
 && test -d /app/training \
 && test -d /app/prefect \
 && test -d /app/models \
 && echo "OK: full mlops-project present in image"

EXPOSE 8000

# Default to serving the API; you can still run training/prefect by overriding the command.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
