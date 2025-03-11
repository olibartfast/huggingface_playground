import torch
import requests
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor #pip install sam2

import numpy as np
import cv2

# Load RT-DETRv2 model and processor
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

# Prepare bounding boxes for SAM
bounding_boxes = []
labels = []
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        bounding_boxes.append(box)
        labels.append(model.config.id2label[label])
        print(f"{model.config.id2label[label]}: {score:.2f} ")

# Convert bounding boxes to the format expected by SAM (xyxy -> xywh if needed, and normalized coordinates)
sam_boxes = []
for box in bounding_boxes:
    x1, y1, x2, y2 = box
    # SAM usually expects normalized coordinates (0-1)
    x1_norm = x1 / image.width
    y1_norm = y1 / image.height
    x2_norm = x2 / image.width
    y2_norm = y2 / image.height
    sam_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

print("\nBounding boxes for SAM (normalized xyxy):", sam_boxes)
print("Labels:", labels)


# --- Visualization (Optional, but highly recommended for debugging) ---
def visualize_boxes(image, boxes, labels):
    """Visualizes bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        # Denormalize for drawing
        x1 *= image.width
        y1 *= image.height
        x2 *= image.width
        y2 *= image.height

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")  # Display the label
    return image

# Visualize the normalized boxes (for debugging)
visualization_image = image.copy()  # Create a copy to draw on
visualized_image = visualize_boxes(visualization_image, sam_boxes, labels)
visualized_image.show()
# --- End of Visualization ---


# --- Integration with SAM2 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large").to(device)

    # Convert PIL Image to numpy array (REQUIRED for SAM) and to RGB
    image_np = np.array(image.convert("RGB"))

    predictor.set_image(image_np)

    # Prepare prompts:  sam_boxes are already normalized xyxy
    prompts = {
        "boxes": torch.tensor(sam_boxes, device=device),
        "box_labels": torch.tensor([0] * len(sam_boxes), device=device),  #  0 for foreground, 1 for background.  All foreground here.
    }

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32): # Use bfloat16 if on CUDA
        masks, _, _ = predictor.predict(prompts)

    masks = masks.cpu().numpy()

    # --- Visualize Masks ---
    def visualize_masks(image, masks, boxes, labels, alpha=0.5):
        """Visualizes masks and bounding boxes on the image."""
        image_np = np.array(image)
        overlay = image_np.copy()
        for mask, box, label in zip(masks, boxes, labels):
            # Draw the mask
            color = np.array([30, 144, 255], dtype=np.uint8)  # Example color (blue)
            overlay[mask == True] = color

            # Draw the bounding box (denormalize first)
            x1, y1, x2, y2 = box
            x1 *= image.width
            y1 *= image.height
            x2 *= image.width
            y2 *= image.height
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Blend the overlay with the original image
        output = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)
        return Image.fromarray(output)

    visualized_image_with_masks = visualize_masks(image, masks, sam_boxes, labels)
    visualized_image_with_masks.show()

except ImportError as e:
    print(f"Error: {e}.  Make sure you have the 'sam2' library installed (`pip install sam2`).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
