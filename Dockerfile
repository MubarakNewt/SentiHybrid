# ✅ Use official Python slim image
FROM python:3.11-slim

# ✅ Set working directory inside the container
WORKDIR /app

# ✅ Copy only requirements first for better layer caching
COPY requirements.txt .

# ✅ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy your backend code and config
COPY backend/ ./backend/
COPY fly.toml ./fly.toml

# ✅ Explicitly copy your models & preprocess folders if needed
# (
