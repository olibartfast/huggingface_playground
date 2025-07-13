import requests
from PIL import Image
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

from transformers import pipeline, AutoProcessor, AutoModel, CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CLIPConfig:
    model: str = "openai/clip-vit-base-patch32"
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    threshold: float = 0.1


class CLIPZeroShotClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def preprocess_inputs(self, image: Image.Image, text_labels: List[str]):
        return self.processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    
    def classify_image(self, image: Image.Image, text_labels: List[str]) -> Dict:
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


def load_image(source: Union[str, Image.Image]) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    elif source.startswith(('http://', 'https://')):
        response = requests.get(source, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    else:
        return Image.open(source)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def get_device(device_str: str) -> Union[int, str]:
    if device_str == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device_str


def run_inference(args, image: Image.Image, labels: List[str]) -> Tuple:
    torch_dtype = get_torch_dtype(args.torch_dtype)
    device = get_device(args.device)
    
    if args.use_pipeline:
        logger.info(f"Setting up pipeline for {args.model}")
        clip_pipeline = pipeline(
            task="zero-shot-image-classification",
            model=args.model,
            torch_dtype=torch_dtype,
            device=device
        )
        outputs = clip_pipeline(image, candidate_labels=labels)
        method = "pipeline"
        
    elif args.use_clip_class:
        logger.info(f"Setting up CLIPZeroShotClassifier for {args.model}")
        classifier = CLIPZeroShotClassifier(args.model)
        outputs = classifier.classify_image(image, labels)
        method = "clip_class"
        
    else:
        logger.info(f"Setting up AutoModel for {args.model}")
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa"
        )
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs).logits_per_image
        method = "automodel"
    
    return process_results(outputs, labels, method)


def process_results(outputs, labels: List[str], method: str = "pipeline") -> Tuple[str, float, List[float]]:
    if method == "pipeline":
        scores = []
        for label in labels:
            for result in outputs:
                if result['label'] == label:
                    scores.append(result['score'])
                    break
            else:
                scores.append(0.0)
        top_label = outputs[0]['label']
        top_score = outputs[0]['score']
    else:
        probs = outputs['probabilities'] if method == "clip_class" else outputs.softmax(dim=1)[0].tolist()
        scores = probs
        top_label = labels[np.argmax(probs)]
        top_score = max(probs)
    
    return top_label, top_score, scores


def create_results_plot(labels: List[str], probabilities: List[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['skyblue' if i != np.argmax(probabilities) else 'orange' for i in range(len(labels))]
    bars = ax.bar(labels, probabilities, color=colors)
    
    ax.set_title('Zero-Shot Classification Probabilities', fontsize=16, fontweight='bold')
    ax.set_xlabel('Class Labels', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(figure: plt.Figure, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    logger.info(f"Figure saved to {output_path}")


def save_image(image: Image.Image, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")


def display_results(labels: List[str], scores: List[float], predicted_label: str, 
                   predicted_score: float, threshold: float = 0.0):
    print("\nClassification Results:")
    for label, score in zip(labels, scores):
        if score >= threshold:
            print(f"  {label}: {score:.4f}")
    print(f"\nPredicted Label: {predicted_label} ({predicted_score*100:.2f}%)")


def validate_args(args):
    if not args.labels:
        raise ValueError("At least one label must be provided")
    
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    methods = [args.use_pipeline, args.use_clip_class]
    if sum(methods) > 1:
        raise ValueError("Only one inference method can be selected")


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    try:
        validate_args(args)
        
        image = load_image(args.image_url)
        
        if args.visualize:
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        
        if args.output_image:
            save_image(image, args.output_image)
        
        top_label, top_score, scores = run_inference(args, image, args.labels)
        
        if args.plot_results or args.save_plot:
            plot_fig = create_results_plot(args.labels, scores)
            if args.save_plot:
                save_figure(plot_fig, args.save_plot)
            if args.plot_results:
                plt.show()
            plt.close(plot_fig)
        
        display_results(args.labels, scores, top_label, top_score, args.threshold)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image: {e}")
        raise
    except (RuntimeError, ValueError) as e:
        logger.error(f"Model inference error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot image classification using CLIP")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                       help="CLIP model name")
    parser.add_argument("--image_url", type=str, 
                       default="http://images.cocodataset.org/val2017/000000039769.jpg",
                       help="URL or path to image")
    parser.add_argument("--labels", type=str, nargs='+', 
                       default=["a photo of a cat", "a photo of a dog", "a photo of a car"],
                       help="List of candidate labels")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Minimum probability threshold for display")
    parser.add_argument("--use_pipeline", action="store_true",
                       help="Use transformers pipeline")
    parser.add_argument("--use_clip_class", action="store_true",
                       help="Use CLIPZeroShotClassifier")
    parser.add_argument("--torch_dtype", type=str, 
                       choices=["float32", "float16", "bfloat16"], default="bfloat16",
                       help="Torch data type")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--visualize", action="store_true",
                       help="Display the input image")
    parser.add_argument("--output_image", type=str,
                       help="Path to save input image")
    parser.add_argument("--plot_results", action="store_true",
                       help="Display results plot")
    parser.add_argument("--save_plot", type=str,
                       help="Path to save results plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
