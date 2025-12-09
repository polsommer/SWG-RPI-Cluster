FROM ghcr.io/xinntao/real-esrgan:latest

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir redis
