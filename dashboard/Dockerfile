FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY static ./static

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8081
ENV OVERLAY_NETWORK=cluster_net
ENV PYTHONUNBUFFERED=1

EXPOSE 8081
CMD ["flask", "run", "--host=0.0.0.0", "--port=8081"]
