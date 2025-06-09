# https://huggingface.co/docs/transformers/model_doc/depth_anything_v2

import requests
from PIL import Image
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "Depth-Anything-V2-Small-hf",
            "Depth-Anything-V2-Base-hf",
            "Depth-Anything-V2-Large-hf",
        ],
        default="Depth-Anything-V2-Small-hf",
        help="Model to use for depth estimation (e.g., Depth-Anything-V2-Small-hf)"
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to process"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for depth map clipping (0.0 to 1.0, 0.0 means no clipping)"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="depth_map.png",
        help="File path to save the output depth map image"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable additional visualization behavior (e.g., display depth map)"
    )
    args = parser.parse_args()
    return args

def setup_model(model_name):
    """Sets up the processor and model based on the model name."""
    model_id = f"depth-anything/{model_name}"
    print(f"Loading {model_name} model...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    print(f"Processor type: {type(processor)}")  # Debugging line: Print processor type
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    print(f"{model_name} model loaded successfully!")
    return processor, model

def draw_detections(image, depth, output_path, visualize=False):
    """Draw and save the depth map as an image."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Normalize depth to 0-255 for visualization
    depth = depth.detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    depth_image = Image.fromarray(depth.astype("uint8"))

    # Save the depth map
    depth_image.save(output_path)
    print(f"Depth map saved to {output_path}")

    # Optionally display the depth map
    if visualize:
        plt.figure()
        plt.imshow(depth_image, cmap='viridis')
        plt.axis('off')
        plt.show()
        plt.close()

def main(args):
    """Main function to perform depth estimation."""
    print("Model selected:", args.model)
    print("Image URL:", args.image_url)
    print("Confidence threshold:", args.threshold)
    print("Output image path:", args.output_image)
    if args.visualize:
        print("Visualization enabled (depth map will be displayed)")

    try:
        processor, model = setup_model(args.model)
        if processor is None or model is None:
            raise Exception("Model setup failed.")

        print("Downloading and opening image...")
        image = Image.open(requests.get(args.image_url, stream=True).raw)
        print("Image loaded successfully!")

        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")
        print("Processing image with model...")
        with torch.no_grad():
            outputs = model(**inputs)
        print("Depth estimation completed!")

        # Post-process to interpolate to original size
        print("Post-processing results...")
        post_processed_output = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        print("Post-processing completed!")

        # Retrieve the predicted depth map
        predicted_depth = post_processed_output[0]["predicted_depth"]

        # Apply threshold if specified (clip depth values)
        if args.threshold > 0.0:
            predicted_depth = torch.clamp(predicted_depth, min=args.threshold * predicted_depth.max())

        # Save and optionally display the depth map
        draw_detections(image, predicted_depth, args.output_image, args.visualize)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)