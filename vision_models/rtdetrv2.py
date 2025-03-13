import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import numpy as np
import argparse  # Import argparse
import requests
import os
from PIL import Image, ImageDraw  # Import PIL components


class RTDetrDetector:
    def __init__(self, model_name="PekingU/rtdetr_v2_r18vd", threshold=0.5, device=None):
        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        self.threshold = threshold
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()  # Put the model in evaluation mode

    def detect(self, image: Image.Image):
        """Detects objects in the image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )
        bounding_boxes, labels, scores = [], [], []
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]
                bounding_boxes.append(box)
                labels.append(self.model.config.id2label[label])
                scores.append(score)
        return bounding_boxes, labels, scores


def visualize_boxes(image: Image.Image, boxes, labels):
    """Visualizes bounding boxes on the image (local to rtdetrv2.py)."""
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")
    return image


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="RT-DETRv2 Object Detector")
    parser.add_argument("--image_url", type=str, default='http://images.cocodataset.org/val2017/000000039769.jpg',
                        help="URL of the image to process.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the output image.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold.")
    parser.add_argument("--model_name", type=str, default="PekingU/rtdetr_v2_r18vd",
                        help="Name of the RT-DETRv2 model to use.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu).  Defaults to CUDA if available.")
    args = parser.parse_args()

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Image ---
    try:
        image = Image.open(requests.get(args.image_url, stream=True).raw)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        exit()
    except Exception as e:
        print(f"Error opening image: {e}")
        exit()

    # --- Object Detection ---
    detector = RTDetrDetector(model_name=args.model_name, threshold=args.threshold, device=args.device)
    bounding_boxes, labels, scores = detector.detect(image)

    print("\nBounding boxes (in xyxy):", bounding_boxes)
    print("Labels:", labels)
    print("Scores:", scores)

    # --- Visualize and Save ---
    visualization_image = image.copy()
    visualized_image = visualize_boxes(visualization_image, bounding_boxes, labels)
    output_path = os.path.join(args.output_dir, "detected_objects_standalone.png")
    visualized_image.save(output_path)
    print(f"Saved detected objects image to: {output_path}")