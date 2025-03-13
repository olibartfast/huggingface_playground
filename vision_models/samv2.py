# https://huggingface.co/facebook/sam2-hiera-large
# https://huggingface.co/docs/transformers/en/model_doc/rt_detr_v2
# rtdetrv2 is used to feed  bounding boxes to samv2
# for other promptless input methods check https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
import torch
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw
import cv2
import os
import requests
import argparse

# Import RTDetrDetector from the rtdetrv2 module
from rtdetrv2 import RTDetrDetector


class SAM2Segmentor:
    def __init__(self, model_name="facebook/sam2-hiera-tiny", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        except ImportError as e:
            raise ImportError(
                "Error: Could not import SAM2ImagePredictor.  Install 'sam2' (`pip install sam2`)."
            ) from e
        except Exception as e:
            raise Exception(f"Error loading SAM2: {e}") from e

        # Crucially, *NO* self.predictor.to(self.device) here.
        self.image_set = False

    def set_image(self, image: Image.Image):
        image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(image_np)  # Device handling is done *within* set_image
        self.image_set = True

    def segment(self, boxes: np.ndarray):
        if not self.image_set:
            raise ValueError("Image must be set using set_image() before segmenting.")

        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=None, point_labels=None, box=boxes, multimask_output=False
            )
        return masks, scores


def show_mask(mask, image_np, random_color=False, alpha=0.5):
    if random_color:
        color = np.concatenate([np.random.random(3) * 255, np.array([alpha * 255])], axis=0)
    else:
        color = np.array([30, 144, 255, alpha * 255], dtype=np.uint8)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image.astype(np.uint8)

def save_masks_to_image(image, masks, boxes, labels, output_path, alpha=0.5):
    image_np = np.array(image)
    overlay = image_np.copy()
    for mask, box, label in zip(masks, boxes, labels):
        mask_image = show_mask(mask, image_np)
        overlay = cv2.addWeighted(overlay, 1.0, mask_image[:, :, :3], alpha, 0)
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    Image.fromarray(overlay).save(output_path)
    print(f"Image with masks and bounding boxes saved to {output_path}")

def visualize_boxes(image: Image.Image, boxes, labels):
        draw = ImageDraw.Draw(image)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            draw.text((x1, y1 - 10), label, fill="red")
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM2 Segmentation with RT-DETRv2")
    parser.add_argument("--image_url", type=str, default='http://images.cocodataset.org/val2017/000000039769.jpg', help="URL of the image.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--detection_threshold", type=float, default=0.5, help="Detection threshold.")
    parser.add_argument("--mask_alpha", type=float, default=0.5, help="Mask transparency.")
    parser.add_argument("--rtdetr_model_name", type=str, default="PekingU/rtdetr_v2_r18vd", help="RT-DETRv2 model name.")
    parser.add_argument("--sam2_model_name", type=str, default="facebook/sam2-hiera-tiny", help="SAM2 model name.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        image = Image.open(requests.get(args.image_url, stream=True).raw)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}"); exit()
    except Exception as e:
        print(f"Error opening image: {e}"); exit()

    detector = RTDetrDetector(model_name=args.rtdetr_model_name, threshold=args.detection_threshold, device=args.device)
    bounding_boxes, labels, scores = detector.detect(image)

    visualization_image = image.copy()
    visualized_image = visualize_boxes(visualization_image, bounding_boxes, labels)
    visualized_image.save(os.path.join(args.output_dir, "detected_objects.png"))

    segmentor = SAM2Segmentor(model_name=args.sam2_model_name, device=args.device)
    segmentor.set_image(image)
    masks, _ = segmentor.segment(np.array(bounding_boxes))

    save_masks_to_image(image, masks, bounding_boxes, labels,
                        os.path.join(args.output_dir, "output_with_masks.jpg"), alpha=args.mask_alpha)

    print("Processing complete.")