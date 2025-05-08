FROM python:3.9-slim

WORKDIR /app

# Install dependensi sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements.txt
COPY requirements.txt .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn websockets

# Salin kode aplikasi (kecuali yang diignore di .dockerignore)
COPY . .

# Pastikan direktori model ada
RUN mkdir -p /app/model

# Salin model dari exp5 ke lokasi yang tetap dalam container
COPY ./runs/train/exp5/weights/best.pt /app/model/qris_model.pt

# Set environment variable MODEL_PATH
ENV MODEL_PATH=/app/model/qris_model.pt

# Expose port untuk FastAPI
EXPOSE 8080

# Command untuk menjalankan aplikasi
CMD ["python", "app.py"]