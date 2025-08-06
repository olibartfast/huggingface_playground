# https://huggingface.co/docs/transformers/model_doc/grounding-dino

import requests
from PIL import Image
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["grounding-dino-tiny", "grounding-dino-base"],
        default="grounding-dino-tiny",
        help="Model to use for object detection (grounding-dino-tiny or grounding-dino-base)"
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to process"
    )
    parser.add_argument(
        "--text_labels",
        type=str,
        nargs='+',
        default=["cat", "dog"],
        help="Text labels to detect. Example: --text_labels 'cat' 'dog'"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for object detection"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="output_image.png",
        help="File path to save the output image with bounding boxes"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable additional visualization behavior (e.g., display image)"
    )
    args = parser.parse_args()
    return args

def setup_model(model_name):
    """Sets up the processor and model based on the model name."""
    model_id = f"IDEA-Research/{model_name}"
    print(f"Loading {model_name} model...")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Processor type: {type(processor)}")  # Debugging line: Print processor type
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    print(f"{model_name} model loaded successfully!")
    return processor, model

def normalize_label(label, text_labels_from_input):
    """Normalize detected label to match input text labels."""
    label = label.lower().strip()
    for input_label in text_labels_from_input:
        if input_label.lower() in label or any(word in label for word in input_label.lower().split()):
            return input_label
    return f"Unknown ({label})"

def draw_detections(image, boxes, scores, labels, text_labels_from_input, output_path, visualize=False):
    """Draw detections and save the output image to a file."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        try:
            if isinstance(label, (int, torch.Tensor)):
                category_name = text_labels_from_input[int(label)]
            else:
                category_name = normalize_label(label, text_labels_from_input)
        except (ValueError, IndexError):
            category_name = f"Invalid label ({label})"
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f"{category_name}: {round(score.item(), 3)}", bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Output image saved to {output_path}")
    if visualize:
        plt.show()
    plt.close(fig)

def main(args):
    """Main function to perform object detection."""
    print("Model selected:", args.model)
    print("Image URL:", args.image_url)
    print("Text labels:", args.text_labels)
    print("Confidence threshold:", args.threshold)
    print("Output image path:", args.output_image)
    if args.visualize:
        print("Visualization enabled (image will be displayed)")

    try:
        processor, model = setup_model(args.model)
        if processor is None or model is None:
            raise Exception("Model setup failed.")

        print("Downloading and opening image...")
        image = Image.open(requests.get(args.image_url, stream=True).raw)
        print("Image loaded successfully!")

        # Prepare text labels (Grounding DINO expects a single string with classes separated by periods)
        text_labels = ". ".join(args.text_labels) + "."
        inputs = processor(images=image, text=[text_labels], return_tensors="pt")
        print("Processing image with model...")
        with torch.no_grad():
            outputs = model(**inputs)
        print("Object detection completed!")

        # Target image sizes (width, height) for rescaling
        target_sizes = torch.tensor([image.size[::-1]])  # Convert (height, width) to (width, height)
        print("Post-processing results...")
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
            threshold=args.threshold,
        )
        print("Post-processing completed!")

        # Retrieve predictions for the first image
        result = results[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        text_labels_from_input = args.text_labels  # Use original text labels for output

        # Debugging: Print raw labels to inspect their format
        print("Raw labels from post-processing:", labels)

        # Check if text_labels is available in result (for transformers>=4.51.0)
        if "text_labels" in result:
            detected_labels = result["text_labels"]
        else:
            detected_labels = labels  # Fallback to labels for older versions

        print("\nDetections:")
        for box, score, label in zip(boxes, scores, detected_labels):
            box = [round(i, 2) for i in box.tolist()]
            try:
                if isinstance(label, (int, torch.Tensor)):
                    category_name = text_labels_from_input[int(label)]
                else:
                    category_name = normalize_label(label, text_labels_from_input)
            except (ValueError, IndexError):
                category_name = f"Invalid label ({label})"
            print(
                f"Detected {category_name} with confidence {round(score.item(), 3)} at location {box}"
            )

        # Always save the output image, display if --visualize is enabled
        draw_detections(image, boxes, scores, detected_labels, text_labels_from_input, args.output_image, args.visualize)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)