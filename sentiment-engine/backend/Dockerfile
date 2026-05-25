FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for hdbscan and umap native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first so this layer is cached
# unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose the FastAPI port
EXPOSE 8050

# Start uvicorn. Models are downloaded from HuggingFace on first startup
# and cached inside the container layer.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8050"]
