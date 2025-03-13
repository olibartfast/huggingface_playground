# https://huggingface.co/facebook/sam2-hiera-large
# https://huggingface.co/docs/transformers/en/model_doc/rt_detr_v2
# rtdetrv2 is used to feed  bounding boxes to samv2
# for other promptless input methods check https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
import torch
import requests
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor  # pip install sam2

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
        # Denormalize for drawing
        x1 *= image.width
        y1 *= image.height
        x2 *= image.width
        y2 *= image.height

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")  # Display the label
    return image


def visualize_masks(image, masks, boxes, labels, alpha=0.5):
    """Visualizes masks and bounding boxes on the image."""
    image_np = np.array(image)
    overlay = image_np.copy()
    for mask, box, label in zip(masks, boxes, labels):
        # Draw the mask
        color = np.array([30, 144, 255], dtype=np.uint8)  # Example color (blue)
        overlay[mask.squeeze() == True] = color  # Added .squeeze()

        # Draw the bounding box (denormalize first)
        x1, y1, x2, y2 = box
        x1 *= image.width
        y1 *= image.height
        x2 *= image.width
        y2 *= image.height
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(
            overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # Blend the overlay with the original image
    output = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)
    return Image.fromarray(output)


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

# Convert bounding boxes to normalized xyxy format
sam_boxes = []
for box in bounding_boxes:
    x1, y1, x2, y2 = box
    x1_norm = x1 / image.width
    y1_norm = y1 / image.height
    x2_norm = x2 / image.width
    y2_norm = y2 / image.height
    sam_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

print("\nBounding boxes for SAM (normalized xyxy):", sam_boxes)
print("Labels:", labels)


# --- Visualize Detected Objects (Optional) ---
visualization_image = image.copy()
visualized_image = visualize_boxes(visualization_image, sam_boxes, labels)
visualized_image.save(os.path.join(OUTPUT_DIR, "detected_objects.png"))
print(f"Saved detected objects image to: {os.path.join(OUTPUT_DIR, 'detected_objects.png')}")


# --- SAM2 Segmentation ---
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    image_np = np.array(image.convert("RGB"))
    predictor.set_image(image_np)
    input_boxes = np.array(bounding_boxes)  # Already in xyxy format
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )

    # --- Save Image with Masks and Bounding Boxes to Disk ---
    def show_mask(mask, image_np, random_color=False, alpha=0.5):
        """Helper function to overlay mask on image."""
        if random_color:
            color = np.concatenate([np.random.random(3) * 255, np.array([alpha * 255])], axis=0)
        else:
            color = np.array([30, 144, 255, alpha * 255], dtype=np.uint8)  # Blue with alpha
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image.astype(np.uint8)

    def save_masks_to_image(image, masks, boxes, labels, output_path="output_with_masks.jpg", alpha=0.5):
        """Saves the image with masks and bounding boxes to disk."""
        image_np = np.array(image)
        overlay = image_np.copy()
        for mask, box, label in zip(masks, boxes, labels):
            mask_image = show_mask(mask, image_np)
            overlay = cv2.addWeighted(overlay, 1.0, mask_image[:, :, :3], alpha, 0)

            # Draw bounding box (already in pixel coordinates)
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_image = Image.fromarray(overlay)
        output_image.save(output_path)
        print(f"Image with masks and bounding boxes saved to {output_path}")

    save_masks_to_image(image, masks, bounding_boxes, labels, "output_with_masks.jpg")


except ImportError as e:
    print(
        f"Error: {e}.  Make sure you have the 'sam2' library installed (`pip install sam2`)."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")