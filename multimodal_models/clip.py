import requests
from PIL import Image
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from transformers import pipeline, AutoProcessor, AutoModel, CLIPProcessor, CLIPModel

class CLIPZeroShotClassifier:
    """Zero-shot image classifier using CLIP model."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def preprocess_inputs(self, image, text_labels):
        """Preprocess image and text labels for CLIP model."""
        return self.processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    
    def classify_image(self, image, text_labels):
        """Classify an image using zero-shot classification."""
        inputs = self.preprocess_inputs(image, text_labels)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probabilities = logits_per_image.softmax(dim=-1)
        
        return {
            'text_labels': text_labels,
            'probabilities': probabilities[0].tolist(),
            'predicted_label': text_labels[probabilities.argmax()],
            'predicted_probability': probabilities.max().item()
        }

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--image_url", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg")
    parser.add_argument("--labels", type=str, nargs='+', default=["a photo of a cat", "a photo of a dog", "a photo of a car"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--use_pipeline", action="store_true")
    parser.add_argument("--use_clip_class", action="store_true")
    parser.add_argument("--torch_dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output_image", type=str, default="clip_output_image.png")
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--save_plot", type=str, default="")
    return parser.parse_args()

def get_torch_dtype(dtype_str):
    """Convert string to torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.bfloat16)

def get_device(device_str):
    """Get the appropriate device."""
    return 0 if device_str == "auto" and torch.cuda.is_available() else device_str

def setup_pipeline(model_name, torch_dtype, device):
    """Sets up the pipeline for zero-shot image classification."""
    return pipeline(task="zero-shot-image-classification", model=model_name, torch_dtype=torch_dtype, device=device)

def setup_automodel(model_name, torch_dtype):
    """Sets up the processor and model using AutoModel approach."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, attn_implementation="sdpa")
    return processor, model

def process_results(outputs, labels, method="pipeline"):
    """Process inference results for all methods."""
    if method == "pipeline":
        # Pipeline returns results sorted by score, need to reorder to match input labels
        scores = []
        for label in labels:
            for result in outputs:
                if result['label'] == label:
                    scores.append(result['score'])
                    break
            else:
                scores.append(0.0)  # If label not found
        top_label = outputs[0]['label']
        top_score = outputs[0]['score']
    else:  # AutoModel or CLIPZeroShotClassifier
        probs = outputs['probabilities'] if method == "clip_class" else outputs.softmax(dim=1)[0].tolist()
        scores = probs
        top_label = labels[np.argmax(probs)]
        top_score = max(probs)
    return top_label, top_score, scores

def pipeline_inference(clip_pipeline, image, labels):
    """Perform inference using pipeline approach."""
    return clip_pipeline(image, candidate_labels=labels)

def automodel_inference(processor, model, image, labels):
    """Perform inference using AutoModel approach."""
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image

def create_results_plot(labels, probabilities):
    """Create a bar plot of the classification probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, probabilities, color=['skyblue' if i != np.argmax(probabilities) else 'orange' for i in range(len(labels))])
    ax.set_title('Zero-Shot Classification Probabilities', fontsize=16, fontweight='bold')
    ax.set_xlabel('Class Labels', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1)
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

def display_figure(figure_or_image, output_path=None, visualize=False, figure_type="image"):
    """Display and optionally save a matplotlib figure or PIL image."""
    fig = figure_or_image if figure_type == "plot" else plt.figure()
    if figure_type != "plot":
        ax = fig.add_subplot(111)
        ax.imshow(figure_or_image)
        ax.axis('off')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    if visualize:
        plt.show()
    plt.close(fig)

def display_results(labels, scores, predicted_label, predicted_score, threshold=0.0):
    """Display classification results in a formatted way."""
    print("\nClassification Results:")
    for label, score in zip(labels, scores):
        if score >= threshold:
            print(f"  {label}: {score:.4f}")
    print(f"\nPredicted Label: {predicted_label} ({predicted_score*100:.2f}%)")

def main(args):
    """Main function to perform zero-shot image classification."""
    try:
        torch_dtype = get_torch_dtype(args.torch_dtype)
        device = get_device(args.device)
        
        image = Image.open(requests.get(args.image_url, stream=True).raw)
        
        if args.visualize or args.output_image:
            display_figure(image, args.output_image if args.output_image != "clip_output_image.png" or args.visualize else None, args.visualize, "image")

        if args.use_pipeline:
            clip_pipeline = setup_pipeline(args.model, torch_dtype, device)
            outputs = pipeline_inference(clip_pipeline, image, args.labels)
            top_label, top_score, scores = process_results(outputs, args.labels, method="pipeline")
        elif args.use_clip_class:
            classifier = CLIPZeroShotClassifier(args.model)
            results = classifier.classify_image(image, args.labels)
            top_label, top_score, scores = process_results(results, args.labels, method="clip_class")
        else:
            processor, model = setup_automodel(args.model, torch_dtype)
            outputs = automodel_inference(processor, model, image, args.labels)
            top_label, top_score, scores = process_results(outputs, args.labels, method="automodel")

        if args.plot_results or args.save_plot:
            plot_fig = create_results_plot(args.labels, scores)
            display_figure(plot_fig, args.save_plot if args.save_plot else None, args.plot_results, "plot")

        display_results(args.labels, scores, top_label, top_score, args.threshold)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)