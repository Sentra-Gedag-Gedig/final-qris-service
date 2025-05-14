import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, APIRouter
from pathlib import Path
import torch
import os
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import traceback

app = FastAPI()

api_router = APIRouter(prefix="/api/v1")
qris_router = APIRouter(prefix="/qris")

weights = os.environ.get('MODEL_PATH', 'runs/train/exp5/weights/best.pt')
print(f"Using model from: {weights}")
device = select_device('')
try:
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride = model.stride
    imgsz = (640, 640)
    model.eval()
    print(f"Model loaded successfully from {weights}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Check if the model file exists at: {weights}")
    model_path = Path(weights)
    if model_path.exists():
        print(f"File exists, size: {model_path.stat().st_size} bytes")
    else:
        print(f"File does not exist!")
    import sys
    sys.exit(1)

def process_image(img):
    img_resized = letterbox(img, imgsz, stride=stride, auto=True)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).float().to(device)
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

@qris_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected to QRIS service!")

    while True:
        try:
            try:
                data = await websocket.receive_text()
                img_data = base64.b64decode(data)
            except Exception as e:
                print(f"Receiving as text failed: {e}, trying binary mode")
                img_data = await websocket.receive_bytes()

            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_json({"message": "Invalid image data"})
                continue

            img_tensor = process_image(img)
            pred = model(img_tensor)
            pred = non_max_suppression(pred, 0.3, 0.45)[0]

            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img.shape).round()
                x1, y1, x2, y2, conf, cls = pred[0].cpu().numpy()

                w, h = img.shape[1], img.shape[0]
                box_w, box_h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                center_left = w * 0.35
                center_right = w * 0.65
                center_top = h * 0.35
                center_bottom = h * 0.65
                min_width_ratio = 0.3
                max_width_ratio = 0.95
                ideal_width_ratio = 0.6

                too_small = box_w < w * min_width_ratio
                too_large = box_w > w * max_width_ratio
                too_left = cx < center_left
                too_right = cx > center_right
                too_high = cy < center_top
                too_low = cy > center_bottom

                if too_small:
                    msg = "QRIS kurang dekat"
                elif too_large:
                    msg = "QRIS terlalu dekat"
                elif too_left and too_high:
                    msg = "QRIS geser ke kanan-bawah"
                elif too_left and too_low:
                    msg = "QRIS geser ke kanan-atas"
                elif too_right and too_high:
                    msg = "QRIS geser ke kiri-bawah"
                elif too_right and too_low:
                    msg = "QRIS geser ke kiri-atas"
                elif too_left:
                    msg = "QRIS geser ke kanan"
                elif too_right:
                    msg = "QRIS geser ke kiri"
                elif too_high:
                    msg = "QRIS geser ke bawah"
                elif too_low:
                    msg = "QRIS geser ke atas"
                else:
                    width_diff = abs(box_w/w - ideal_width_ratio)
                    if width_diff > 0.25:
                        if box_w/w < ideal_width_ratio:
                            msg = "QRIS sedikit kurang dekat"
                        else:
                            msg = "QRIS sedikit terlalu dekat"
                    else:
                        msg = "OK"

                horizontal_score = 100 - min(100, abs(cx - w/2) / (w/4) * 100)
                vertical_score = 100 - min(100, abs(cy - h/2) / (h/4) * 100)
                size_diff = abs(box_w/w - ideal_width_ratio)
                size_score = 100 - min(100, size_diff / (ideal_width_ratio/2) * 100)
                position_score = int((horizontal_score + vertical_score + size_score) / 3)

                result = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "center": [float(cx), float(cy)],
                    "box_size": [float(box_w), float(box_h)],
                    "message": msg,
                    "position_metrics": {
                        "horizontal_score": int(horizontal_score),
                        "vertical_score": int(vertical_score),
                        "size_score": int(size_score),
                        "overall_score": position_score
                    }
                }
            else:
                result = {
                    "message": "QRIS tidak terdeteksi",
                    "position_metrics": {
                        "horizontal_score": 0,
                        "vertical_score": 0,
                        "size_score": 0,
                        "overall_score": 0
                    }
                }

            await websocket.send_json(result)

        except Exception as e:
            error_detail = traceback.format_exc()
            print(f"Error: {e}\n{error_detail}")
            await websocket.send_json({"error": str(e), "detail": error_detail})
            continue

api_router.include_router(qris_router)
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "QRIS Detection Service is running"}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))

    print(f"Model loaded: {weights}")
    print(f"Device: {device}")
    print(f"Model stride: {stride}")
    print(f"Input image size: {imgsz}")
    print(f"Starting QRIS detection service on http://0.0.0.0:{port}")
    print(f"WebSocket available at ws://0.0.0.0:{port}/api/v1/qris/ws")

    uvicorn.run(app, host="0.0.0.0", port=port)