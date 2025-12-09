FROM ghcr.io/xinntao/real-esrgan:latest

# ---------------------------------------------------------
# Runtime environment improvements
# ---------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# ---------------------------------------------------------
# Install Python runtime dependencies
# ---------------------------------------------------------
# We only need redis for your pubsub pipeline.
# Requests is already inside base Real-ESRGAN image,
# but we can ensure it's present and updated.
RUN pip install --no-cache-dir \
      redis \
      requests

# ---------------------------------------------------------
# Working directory (ESRGAN scripts already here)
# ---------------------------------------------------------
WORKDIR /app
