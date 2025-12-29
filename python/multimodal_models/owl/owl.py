# https://huggingface.co/docs/transformers/en/model_doc/owlvit
# https://huggingface.co/docs/transformers/en/model_doc/owlv2
import requests
from PIL import Image
import torch
import argparse

# required transformers version: 4.49.0
from transformers import Owlv2Processor, Owlv2ForObjectDetection, OwlViTProcessor, OwlViTForObjectDetection

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["owlvit", "owl2"], default="owl2", help="Model to use for object detection (owlvit or owl2)")
    parser.add_argument("--image_url", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg", help="URL of the image to process")
    parser.add_argument("--text_labels", type=str, nargs='+', default=["a photo of a cat", "a photo of a dog"], help="Text labels to detect. Example: --text_labels 'a photo of a cat' 'a photo of a dog'")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold for object detection")
    args = parser.parse_args()
    return args

def setup_model(model_name):
    """Sets up the processor and model based on the model name."""
    processor = None
    model = None
    if model_name == "owl2":
        print("Loading OwlV2 model...")
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        print(f"Processor type: {type(processor)}") # Debugging line: Print processor type
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        print("OwlV2 model loaded successfully!")
    elif model_name == "owlvit":
        print("Loading OwlViT model...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        print(f"Processor type: {type(processor)}") # Debugging line: Print processor type
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        print("OwlViT model loaded successfully!")
    else:
        raise ValueError(f"Invalid model name: {model_name}. Choose from 'owlvit' or 'owl2'.")
    return processor, model

def main(args):
    """Main function to perform object detection."""
    print("Model selected:", args.model)
    print("Image URL:", args.image_url)
    print("Text labels:", args.text_labels)
    print("Confidence threshold:", args.threshold)

    try:
        processor, model = setup_model(args.model)
        if processor is None or model is None:
            raise Exception("Model setup failed.")

        print("Downloading and opening image...")
        image = Image.open(requests.get(args.image_url, stream=True).raw)
        print("Image loaded successfully!")

        text_labels = [args.text_labels] # Wrap text_labels in a list to match expected input format
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        print("Processing image with model...")
        outputs = model(**inputs)
        print("Object detection completed!")

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([(image.height, image.width)])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        print("Post-processing results...")
        results = processor.post_process_grounded_object_detection( # Reverted to post_process_grounded_object_detection for OwlV2
            outputs=outputs, target_sizes=target_sizes, threshold=args.threshold, text_labels=text_labels
        )

        print("Post-processing completed!")

        # Retrieve predictions for the first image for the corresponding text queries
        result = results[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"] # Use labels instead of text_labels from result
        text_labels_from_input = args.text_labels # Use original text labels for output

        print("\nDetections:")
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            category_name = text_labels_from_input[label] # Get category name from input text labels using predicted label index
            print(
                f"Detected {category_name} with confidence {round(score.item(), 3)} at location {box}"
            )

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except ValueError as ve:
        print(ve) # Handle ValueError from setup_model for invalid model name
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

# onnx export
# pip install optimum[exporters]
# optimum-cli export onnx --model google/owlvit-base-patch32 owlvit_onnx
# optimum-cli export onnx --model google/owlv2-base-patch16 owl2_onnx  
