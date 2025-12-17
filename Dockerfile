FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN mkdir -p /app/models/production

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN python -m src.pipelines.aqi_flow

EXPOSE 7860
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port 7860"]
