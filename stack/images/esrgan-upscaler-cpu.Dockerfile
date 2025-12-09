FROM python:3.11-slim

# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# ---------------------------------------------------------
# Install system libraries required by Pillow
# ---------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libopenjp2-7 \
        libtiff5 \
        libwebp7 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
# If you want exact versions, pin them:
# RUN pip install --no-cache-dir pillow==10.3.0 redis==5.0.1
RUN pip install --no-cache-dir \
      pillow \
      redis

# ---------------------------------------------------------
# App directory
# ---------------------------------------------------------
WORKDIR /app
