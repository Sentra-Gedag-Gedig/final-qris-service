# Use a simpler Dockerfile based on the Ultralytics image
FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn websockets

COPY . .

EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port $PORT