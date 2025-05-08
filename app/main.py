from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import os
import uuid
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import io

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='app/model/best.pt')
model.conf = 0.3  # Confidence threshold

LABEL_MAP = {0: "evse", 1: "panel", 2: "gfi"}

os.makedirs("app/static", exist_ok=True)

def convert_pdf_to_image(file_bytes):
    return convert_from_bytes(file_bytes, dpi=200)[0]

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y, w, h = det['bbox']
        draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
        draw.text((x, y - 10), f"{det['label']} {det['confidence']:.2f}", fill='red')
    return image

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()
    image = convert_pdf_to_image(content) if ext == ".pdf" else Image.open(io.BytesIO(content)).convert("RGB")

    results = model(image)
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf < 0.3: continue
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append({
            "label": LABEL_MAP[int(cls)],
            "confidence": round(float(conf), 2),
            "bbox": [x1, y1, x2 - x1, y2 - y1]
        })

    annotated = draw_boxes(image.copy(), detections)
    filename = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join("app/static", filename)
    annotated.save(out_path)

    return {
        "detections": detections,
        "image_url": f"/static/{filename}"
    }
