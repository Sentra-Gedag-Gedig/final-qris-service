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

# Buat struktur direktori untuk model
RUN mkdir -p runs/train/exp10/weights

# Salin kode aplikasi (kecuali yang diignore di .dockerignore)
COPY . .

# Expose port untuk FastAPI
EXPOSE 8080

# Command untuk menjalankan aplikasi
CMD ["python", "app.py"]