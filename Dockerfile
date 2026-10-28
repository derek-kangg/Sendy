# Sendy — one container, one process. Works as-is on Fly.io, Cloud Run,
# Railway or any host that runs a Docker image and sets $PORT.
FROM python:3.12-slim

# opencv-python-headless needs only glib at runtime; everything else is wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Routes live here; mount a volume at /data in production so they survive
# redeploys (SENDY_DATA_DIR is read by app.py).
ENV SENDY_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8080

# exec so Streamlit receives the host's stop signal directly
CMD ["sh", "-c", "exec streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"]
