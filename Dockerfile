FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY downloader.py processor.py app.py npc_export.py physchem_upload.py ./
COPY assets/ ./assets/
COPY .env.example .

VOLUME ["/app/data"]

EXPOSE 8090

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8090/ || exit 1

CMD ["python", "app.py"]
