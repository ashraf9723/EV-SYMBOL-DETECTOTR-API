import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv5 model from local file
model = torch.hub.load('ultralytics/yolov5', 'custom', path='app/model/best.pt')
model.conf = 0.3  # Confidence threshold

# Load a sample image (update path as needed)
image_path = 'data/sample_page.png'
img = Image.open(image_path).convert("RGB")

# Run inference
results = model(img)

# Print results
results.print()  # Prints detections: label, confidence, box

# Show image with bounding boxes
results.show()

# Optionally, save annotated image
results.save(save_dir='results')
