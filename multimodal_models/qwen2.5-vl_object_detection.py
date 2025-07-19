import requests
from PIL import Image, ImageDraw, ImageFont
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import json
import re
import io
import pprint
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectDetectionConfig:
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "auto"
    device: str = "auto"
    max_new_tokens: int = 1000
    box_color: str = "red"
    box_width: int = 3
    font_size: int = 32
    text_color: str = "white"
    text_bg: str = "red"


class Qwen2_5_VLObjectDetector:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info(f"Model loaded on: {self.model.device}")
    
    def _repair_newlines_inside_strings(self, txt: str) -> str:
        """
        Replace raw newlines that occur *inside* JSON string literals with a space.
        Very lightweight: it simply looks for a quote, then any run of characters
        that is NOT a quote or backslash, then a newline, then continues…
        """
        pattern = re.compile(r'("([^"\\]|\\.)*)\n([^"]*")')
        while pattern.search(txt):
            txt = pattern.sub(lambda m: m.group(1) + r'\n' + m.group(3), txt)
        return txt

    def extract_json(self, code_block: str, parse: bool = True):
        """
        Remove Markdown code-block markers (``` or ```json) and return:
          • the raw JSON string   (parse=False, default)
          • the parsed Python obj (parse=True)
        """
        # Look for triple-backtick blocks, optionally tagged with a language (e.g. ```json)
        block_re = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.DOTALL)
        m = block_re.search(code_block)
        payload = (m.group(1) if m else code_block).strip()
        if parse:
            try:
                return json.loads(payload)
            except json.JSONDecodeError as e:
                # attempt a mild repair and retry once
                payload_fixed = self._repair_newlines_inside_strings(payload)
                return json.loads(payload_fixed)
        else:
            return payload
    
    def _text_wh(self, draw, text, font):
        """
        Return (width, height) of *text* under the given *font*, coping with
        Pillow ≥10.0 (textbbox) and older versions (textsize).
        """
        # Check if the draw object has the 'textbbox' method (Pillow >= 8.0)
        if hasattr(draw, "textbbox"):  # Pillow ≥8.0, preferred
            # Get the bounding box of the text
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            # Calculate and return the width and height
            return right - left, bottom - top
        # Check if the draw object has the 'textsize' method (Pillow < 10.0)
        elif hasattr(draw, "textsize"):  # Pillow <10.0
            # Get the size of the text
            return draw.textsize(text, font=font)
        # Fallback for other or older versions of Pillow
        else:  # Fallback
            # Get the bounding box from the font itself
            left, top, right, bottom = font.getbbox(text)
            # Calculate and return the width and height
            return right - left, bottom - top

    def draw_bboxes(self, img, detections, box_color="red", box_width=3, 
                   font_size=32, text_color="white", text_bg="red"):
        """Draw bounding boxes on image"""
        # Create a drawing object for the image
        draw = ImageDraw.Draw(img)
        try:
            # Try to load a TrueType font
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            # If TrueType font is not found, load the default font
            font = ImageFont.load_default()

        # Iterate through each detected object
        for det in detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = det["bbox_2d"]
            # Get the label of the detected object, default to empty string if not present
            label = str(det.get("label", ""))

            # Draw the rectangle (bounding box) on the image
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

            # If a label exists, draw the label text
            if label:
                # Get the width and height of the text label
                tw, th = self._text_wh(draw, label, font)
                # Set padding around the text
                pad = 2
                # Calculate the top-left x-coordinate for the text background
                tx1 = x1
                # Calculate the top-left y-coordinate for the text background, ensuring it stays within the top edge of the image
                ty1 = max(0, y1 - th - 2 * pad)  # keep inside top edge
                # Calculate the bottom-right x-coordinate for the text background
                tx2 = x1 + tw + 2 * pad
                # Calculate the bottom-right y-coordinate for the text background
                ty2 = ty1 + th + 2 * pad

                # If a text background color is specified, draw the background rectangle
                if text_bg:
                    draw.rectangle([tx1, ty1, tx2, ty2],
                                 fill=text_bg, outline=box_color)
                # Draw the text label on the image
                draw.text((tx1 + pad, ty1 + pad), label,
                          fill=text_color, font=font)

        # Return the modified image with bounding boxes and labels
        return img
    
    def create_detection_message(self, image: Image.Image, detection_prompt: str = "Detect all objects in this image") -> List[Dict]:
        """Create a message format for object detection with Qwen2.5-VL"""
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an object detector. The format of your output should be a valid JSON array "
                            "with objects of the form {'bbox_2d': [x1, y1, x2, y2], 'label': 'class'} where "
                            "class is the name of the detected object and [x1, y1, x2, y2] are the bounding box coordinates."
                        )
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": detection_prompt
                    }
                ],
            }
        ]
    
    def run_inference(self, msgs: List[Dict], max_new_tokens: int = 1000) -> List[Dict]:
        """Run object detection inference"""
        logger.info("Running object detection inference...")
        
        # Build the full textual prompt that Qwen-VL expects
        text_prompt = self.processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Extract vision-modalities from msgs and convert them to model-ready tensors
        image_inputs, video_inputs = process_vision_info(msgs)

        # Pack text + vision into model-ready tensors
        inputs = self.processor(
            text=[text_prompt],      # 1-element batch containing the chat prompt string
            images=image_inputs,     # list of raw PIL images (pre-processed inside processor)
            videos=video_inputs,     # list of raw video clips (if any)
            padding=True,            # pad sequences so text/vision tokens line up in a batch
            return_tensors="pt",     # return a dict of PyTorch tensors (input_ids, pixel_values, …)
        ).to(self.model.device)      # move every tensor—text & vision—to the model's GPU/CPU

        # Run inference (no gradients, pure generation)
        with torch.no_grad():                     # disable autograd to save memory
            generated_ids = self.model.generate(  # autoregressive decoding
                **inputs,                         # unpack dict into generate(...)
                max_new_tokens=max_new_tokens     # cap the response to max_new_tokens
            )
        
        # Extract the newly generated tokens (skip the prompt length)
        output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=False
        )[0]
        
        logger.info(f"RAW output:\n{output}\n")
        
        # Extract JSON from the output
        try:
            bounding_boxes = self.extract_json(output)
            logger.info("JSON output:")
            pprint.pprint(bounding_boxes, indent=4)
            return bounding_boxes
        except Exception as e:
            logger.error(f"Failed to parse JSON output: {e}")
            return []
    
    def detect_objects(self, image: Image.Image, detection_prompt: str = "Detect all objects in this image", 
                      max_new_tokens: int = 1000) -> List[Dict]:
        """Main object detection method"""
        msgs = self.create_detection_message(image, detection_prompt)
        return self.run_inference(msgs, max_new_tokens)
    
    def detect_and_visualize(self, image: Image.Image, detection_prompt: str = "Detect all objects in this image",
                           max_new_tokens: int = 1000, box_color: str = "red", box_width: int = 3,
                           font_size: int = 32, text_color: str = "white", text_bg: str = "red") -> Tuple[List[Dict], Image.Image]:
        """Detect objects and return both detections and annotated image"""
        detections = self.detect_objects(image, detection_prompt, max_new_tokens)
        
        if detections:
            annotated_image = self.draw_bboxes(
                image.copy(), detections, box_color, box_width, 
                font_size, text_color, text_bg
            )
        else:
            annotated_image = image.copy()
        
        return detections, annotated_image


def load_image(source: Union[str, Image.Image]) -> Image.Image:
    """Load image from URL, file path, or PIL Image"""
    if isinstance(source, Image.Image):
        return source
    elif source.startswith(('http://', 'https://')):
        response = requests.get(source, stream=True, timeout=15)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(source).convert("RGB")


def display_image(img: Image.Image, title: str = "Image"):
    """Display image using matplotlib"""
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def save_image(image: Image.Image, output_path: str):
    """Save image to file"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")


def save_detections(detections: List[Dict], output_path: str):
    """Save detection results to JSON file"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, indent=2)
    logger.info(f"Detections saved to {output_path}")


def display_results(detection_prompt: str, detections: List[Dict]):
    """Display the detection prompt and results"""
    print("\n" + "="*60)
    print("QWEN2.5-VL OBJECT DETECTION RESULTS")
    print("="*60)
    print(f"Detection Prompt: {detection_prompt}")
    print(f"\nDetected Objects ({len(detections)} total):")
    for i, detection in enumerate(detections, 1):
        bbox = detection.get("bbox_2d", [])
        label = detection.get("label", "unknown")
        print(f"{i:2d}. {label:<20} - BBox: {bbox}")
    print("="*60)


def cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    """Main function for object detection"""
    try:
        print("Model selected:", args.model)
        print("Detection prompt:", args.detection_prompt)
        print("Max new tokens:", args.max_new_tokens)
        
        # Load image
        print("Loading image...")
        print("Image source:", args.image_source)
        image = load_image(args.image_source)
        print("Image loaded successfully!")
        
        # Display input image if requested
        if args.visualize_input:
            display_image(image, "Input Image")
        
        # Save input image if requested
        if args.save_input:
            save_image(image, args.save_input)
        
        # Initialize object detector
        detector = Qwen2_5_VLObjectDetector(args.model)
        
        # Run object detection
        detections, annotated_image = detector.detect_and_visualize(
            image, 
            args.detection_prompt, 
            args.max_new_tokens,
            args.box_color,
            args.box_width,
            args.font_size,
            args.text_color,
            args.text_bg
        )
        
        # Display results
        display_results(args.detection_prompt, detections)
        
        # Display annotated image if requested
        if args.visualize_output:
            display_image(annotated_image, "Object Detection Results")
        
        # Save annotated image if requested
        if args.output_image:
            save_image(annotated_image, args.output_image)
        
        # Save detection results if requested
        if args.output_detections:
            save_detections(detections, args.output_detections)
        
        return detections, annotated_image
        
    except Exception as e:
        logger.error(f"Error during object detection: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Object detection using Qwen2.5-VL")
    
    parser.add_argument("--model", type=str, 
                       choices=["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"],
                       default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Qwen2.5-VL model name")
    
    parser.add_argument("--image_source", type=str, 
                       default="https://learnopencv.com/wp-content/uploads/2025/06/elephants.jpg",
                       help="URL or path to image")
    
    parser.add_argument("--detection_prompt", type=str, 
                       default="Detect all objects in this image",
                       help="Text prompt for object detection")
    
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                       help="Maximum number of new tokens to generate")
    
    # Visualization parameters
    parser.add_argument("--box_color", type=str, default="red",
                       help="Color for bounding boxes")
    
    parser.add_argument("--box_width", type=int, default=3,
                       help="Width of bounding box lines")
    
    parser.add_argument("--font_size", type=int, default=32,
                       help="Font size for labels")
    
    parser.add_argument("--text_color", type=str, default="white",
                       help="Color for label text")
    
    parser.add_argument("--text_bg", type=str, default="red",
                       help="Background color for label text")
    
    # Display and save options
    parser.add_argument("--visualize_input", action="store_true",
                       help="Display the input image")
    
    parser.add_argument("--visualize_output", action="store_true",
                       help="Display the annotated output image")
    
    parser.add_argument("--save_input", type=str,
                       help="Path to save input image")
    
    parser.add_argument("--output_image", type=str,
                       help="Path to save annotated image")
    
    parser.add_argument("--output_detections", type=str,
                       help="Path to save detection results JSON")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)