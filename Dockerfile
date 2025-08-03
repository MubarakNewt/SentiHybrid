# Use Python slim image
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# ✅ Use 1 worker for free Fly.io plan
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "backend.app:app"]
