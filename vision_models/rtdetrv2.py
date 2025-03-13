import torch
import requests
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

import numpy as np
import cv2
import os

# --- Configuration ---
IMAGE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
OUTPUT_DIR = "output"  # Directory to save the output images
DETECTION_THRESHOLD = 0.5
MASK_ALPHA = 0.5  # Transparency of the mask overlay

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper Functions ---
def visualize_boxes(image, boxes, labels):
    """Visualizes bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")  # Display the label
    return image


# --- Load Image ---
try:
    image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()


# --- RT-DETRv2 Object Detection ---
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([(image.height, image.width)]),
    threshold=DETECTION_THRESHOLD,
)

# Prepare bounding boxes and labels
bounding_boxes = []
labels = []
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        bounding_boxes.append(box)
        labels.append(model.config.id2label[label])
        print(f"{model.config.id2label[label]}: {score:.2f} ")

print("\nBounding boxes (xyxy):", bounding_boxes)
print("Labels:", labels)


# --- Visualize Detected Objects (Optional) ---
visualization_image = image.copy()
visualized_image = visualize_boxes(visualization_image, bounding_boxes, labels)
visualized_image.save(os.path.join(OUTPUT_DIR, "detected_objects_detrv2.png"))
print(f"Saved detected objects image to: {os.path.join(OUTPUT_DIR, 'detected_objects_detrv2.png')}")