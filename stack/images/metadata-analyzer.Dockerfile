FROM python:3.11-slim AS runtime

# Ensure predictable output + no buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# ---------------------------------------------------------
# Install native libs required for image processing
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
# Install Python deps
# ---------------------------------------------------------
# If you want exact versions, uncomment + pin:
# RUN pip install --no-cache-dir \
#       pillow==10.3.0 \
#       redis==5.0.1 \
#       requests==2.32.3

RUN pip install --no-cache-dir \
      pillow \
      redis \
      requests

# ---------------------------------------------------------
# Create working dir
# ---------------------------------------------------------
WORKDIR /opt

# Optional: copy the analyzer script here in your service
# COPY metadata_analyzer.py /opt/metadata_analyzer.py

# Default command set in docker-compose
