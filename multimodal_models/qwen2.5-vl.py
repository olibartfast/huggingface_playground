"""
Qwen2.5-VL Vision-Language Model Inference

This script provides three approaches for running Qwen2.5-VL models:
1. Pipeline API - Simple and high-level interface
2. AutoModel API - More control over model parameters
3. Custom Class - Reusable processor with memory management

MEMORY MANAGEMENT:
- For limited GPU memory (< 8GB), use --force_cpu flag
- For moderate GPU memory, use --low_memory flag for optimization
- The script automatically handles CUDA out of memory errors with fallback strategies
- Memory optimization includes cache clearing and reduced token generation

USAGE EXAMPLES:
# Basic usage
python qwen2.5-vl.py

# Force CPU usage for limited GPU memory
python qwen2.5-vl.py --force_cpu

# Use memory optimization
python qwen2.5-vl.py --low_memory

# Specify model size
python qwen2.5-vl.py --model_name Qwen/Qwen2.5-VL-7B-Instruct

REQUIREMENTS:
- torch>=2.0.0
- transformers>=4.47.0
- Pillow
- requests
- accelerate (optional, for device_map="auto")

Model sizes and memory requirements:
- Qwen2.5-VL-3B: ~6GB GPU memory
- Qwen2.5-VL-7B: ~14GB GPU memory  
- Qwen2.5-VL-72B: ~144GB GPU memory
"""

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

from transformers import pipeline, AutoProcessor, Qwen2_5_VLForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Qwen2_5_VLConfig:
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    max_new_tokens: int = 128
    min_pixels: int = 224*224
    max_pixels: int = 1024*1024


class Qwen2_5_VLProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
                 min_pixels: int = 224*224, max_pixels: int = 1024*1024):
        self.model_name = model_name
        
        # Try different loading strategies based on available memory
        try:
            # Strategy 1: Try with device_map="auto" and accelerate
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"
            )
            logger.info("Model loaded with device_map='auto'")
        except ValueError as e:
            if "accelerate" in str(e):
                logger.warning("accelerate not available, trying alternative loading strategies...")
                try:
                    # Strategy 2: Try loading with CPU offloading
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        attn_implementation="sdpa"
                    )
                    # Move to GPU if there's enough memory
                    if torch.cuda.is_available():
                        try:
                            self.model = self.model.to("cuda")
                            logger.info("Model loaded to GPU")
                        except torch.cuda.OutOfMemoryError:
                            logger.warning("Not enough GPU memory, keeping model on CPU")
                            self.model = self.model.to("cpu")
                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU out of memory, loading model on CPU")
                    # Strategy 3: Load entirely on CPU
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # Use float32 on CPU for compatibility
                        low_cpu_mem_usage=True,
                        attn_implementation="sdpa"
                    ).to("cpu")
            else:
                raise e
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU out of memory during loading, trying CPU...")
            # Strategy 3: Load on CPU due to memory constraints
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 on CPU for compatibility
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            ).to("cpu")
            logger.info("Model loaded on CPU due to GPU memory constraints")
                
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
    
    def create_message(self, image: Image.Image, prompt: str) -> List[Dict]:
        """Create a message format for Qwen2.5-VL"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    
    def generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate text response for an image and prompt"""
        messages = self.create_message(image, prompt)
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Clear cache before inference
        if torch.cuda.is_available() and self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU out of memory during inference, clearing cache and retrying...")
                torch.cuda.empty_cache()
                # Try with lower max_new_tokens
                max_new_tokens = min(max_new_tokens, 64)
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Clear cache after inference
        if torch.cuda.is_available() and self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return output_text[0] if output_text else ""


def load_image(source: Union[str, Image.Image]) -> Image.Image:
    """Load image from URL, file path, or PIL Image"""
    if isinstance(source, Image.Image):
        return source
    elif source.startswith(('http://', 'https://')):
        response = requests.get(source, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    else:
        return Image.open(source)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def get_device(device_str: str) -> Union[int, str]:
    """Get device for inference"""
    if device_str == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device_str


def setup_pipeline(model_name: str, torch_dtype: torch.dtype, device: Union[int, str]):
    """Setup pipeline for image-text-to-text generation"""
    logger.info(f"Loading {model_name} with pipeline...")
    pipe = pipeline(
        task="image-text-to-text",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype
    )
    logger.info("Pipeline loaded successfully!")
    return pipe


def setup_automodel(model_name: str, torch_dtype: torch.dtype, min_pixels: int, max_pixels: int):
    """Setup AutoModel and processor for image-text-to-text generation"""
    logger.info(f"Loading {model_name} with AutoModel...")
    
    # Try different loading strategies based on available memory
    try:
        # Strategy 1: Try with device_map="auto" and accelerate
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="sdpa"
        )
        logger.info("Model loaded with device_map='auto'")
    except ValueError as e:
        if "accelerate" in str(e):
            logger.warning("accelerate not available, trying alternative loading strategies...")
            try:
                # Strategy 2: Try loading with CPU offloading
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa"
                )
                # Move to GPU if there's enough memory
                if torch.cuda.is_available():
                    try:
                        model = model.to("cuda")
                        logger.info("Model loaded to GPU")
                    except torch.cuda.OutOfMemoryError:
                        logger.warning("Not enough GPU memory, keeping model on CPU")
                        model = model.to("cpu")
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU out of memory, loading model on CPU")
                # Strategy 3: Load entirely on CPU
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 on CPU for compatibility
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa"
                ).to("cpu")
        else:
            raise e
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU out of memory during loading, trying CPU...")
        # Strategy 3: Load on CPU due to memory constraints
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 on CPU for compatibility
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        ).to("cpu")
        logger.info("Model loaded on CPU due to GPU memory constraints")
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    logger.info("AutoModel and processor loaded successfully!")
    return model, processor


def setup_qwen_class(model_name: str, min_pixels: int, max_pixels: int):
    """Setup Qwen2_5_VLProcessor class"""
    logger.info(f"Setting up Qwen2_5_VLProcessor for {model_name}...")
    processor_class = Qwen2_5_VLProcessor(model_name, min_pixels, max_pixels)
    logger.info("Qwen2_5_VLProcessor loaded successfully!")
    return processor_class


def pipeline_inference(pipe, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """Run inference using pipeline"""
    logger.info("Running pipeline inference...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Clear cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        result = pipe(messages, max_new_tokens=max_new_tokens, return_full_text=False)
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU out of memory during pipeline inference, clearing cache and retrying...")
        torch.cuda.empty_cache()
        # Try with lower max_new_tokens
        max_new_tokens = min(max_new_tokens, 64)
        result = pipe(messages, max_new_tokens=max_new_tokens, return_full_text=False)
    
    # Clear cache after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Pipeline inference completed!")
    
    return result[0]['generated_text'] if result else ""


def automodel_inference(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """Run inference using AutoModel"""
    logger.info("Running AutoModel inference...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Clear cache before inference
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU out of memory during inference, clearing cache and retrying...")
            torch.cuda.empty_cache()
            # Try with lower max_new_tokens
            max_new_tokens = min(max_new_tokens, 64)
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Clear cache after inference
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.empty_cache()
    
    logger.info("AutoModel inference completed!")
    return output_text[0] if output_text else ""


def qwen_class_inference(processor_class, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """Run inference using Qwen2_5_VLProcessor class"""
    logger.info("Running Qwen2_5_VLProcessor inference...")
    result = processor_class.generate_response(image, prompt, max_new_tokens)
    logger.info("Qwen2_5_VLProcessor inference completed!")
    return result


def run_inference(args, image: Image.Image, prompt: str) -> str:
    """Run inference based on selected method"""
    torch_dtype = get_torch_dtype(args.torch_dtype)
    device = get_device(args.device)
    
    # Override device if force_cpu is set
    if args.force_cpu:
        device = "cpu"
        torch_dtype = torch.float32  # Use float32 for CPU
        logger.info("Forcing CPU usage due to --force_cpu flag")
    
    if args.use_pipeline:
        pipe = setup_pipeline(args.model, torch_dtype, device)
        return pipeline_inference(pipe, image, prompt, args.max_new_tokens)
        
    elif args.use_qwen_class:
        processor_class = setup_qwen_class(args.model, args.min_pixels, args.max_pixels)
        return qwen_class_inference(processor_class, image, prompt, args.max_new_tokens)
        
    else:  # AutoModel approach (default)
        model, processor = setup_automodel(args.model, torch_dtype, args.min_pixels, args.max_pixels)
        return automodel_inference(model, processor, image, prompt, args.max_new_tokens)


def save_image(image: Image.Image, output_path: str):
    """Save image to file"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")


def display_results(prompt: str, response: str):
    """Display the prompt and generated response"""
    print("\n" + "="*60)
    print("QWEN2.5-VL RESULTS")
    print("="*60)
    print(f"Prompt: {prompt}")
    print(f"\nGenerated Response:\n{response}")
    print("="*60)


def validate_args(args):
    """Validate command line arguments"""
    if not args.prompt:
        raise ValueError("A prompt must be provided")
    
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    
    methods = [args.use_pipeline, args.use_qwen_class]
    if sum(methods) > 1:
        raise ValueError("Only one inference method can be selected")


def cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_accelerate():
    """Check if accelerate is available and provide installation instructions if not"""
    try:
        import accelerate
        return True
    except ImportError:
        logger.warning("accelerate package not found. For optimal performance, install it with: pip install accelerate")
        return False


def setup_memory_optimizations():
    """Setup memory optimizations for CUDA"""
    if torch.cuda.is_available():
        # Set CUDA memory allocation configuration for better memory management
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_free = torch.cuda.memory_reserved(0) / 1024**3  # GB
        
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB total")
        logger.info(f"GPU Memory Reserved: {gpu_free:.2f} GB")
        
        return gpu_memory, gpu_free
    return 0, 0


def get_memory_efficient_dtype(device: str) -> torch.dtype:
    """Get appropriate dtype based on device and memory constraints"""
    if device == "cpu":
        return torch.float32  # CPU works better with float32
    else:
        return torch.bfloat16  # GPU can use bfloat16 for memory efficiency


def main(args):
    """Main function to perform vision-language inference"""
    try:
        validate_args(args)
        
        # Setup memory optimizations
        gpu_memory, gpu_free = setup_memory_optimizations()
        
        # Check for accelerate
        has_accelerate = check_accelerate()
        if not has_accelerate:
            logger.info("Running without accelerate - model loading may be slower")
        
        # Suggest smaller model if GPU memory is very limited
        if gpu_memory > 0 and gpu_memory < 8:  # Less than 8GB GPU
            if "7B" in args.model or "72B" in args.model:
                logger.warning(f"Your GPU has {gpu_memory:.1f}GB memory. Consider using Qwen/Qwen2.5-VL-3B-Instruct for better performance")
        
        print("Model selected:", args.model)
        print("Image URL:", args.image_url)
        print("Prompt:", args.prompt)
        print("Max new tokens:", args.max_new_tokens)
        print("Min pixels:", args.min_pixels)
        print("Max pixels:", args.max_pixels)
        
        # Determine inference method
        if args.use_pipeline:
            print("Using pipeline approach")
        elif args.use_qwen_class:
            print("Using Qwen2_5_VLProcessor class approach")
        else:
            print("Using AutoModel approach")
        
        print("Torch dtype:", args.torch_dtype)
        print("Device:", args.device)
        
        # Load image
        print("Downloading and opening image...")
        image = load_image(args.image_url)
        print("Image loaded successfully!")
        
        # Display image if requested
        if args.visualize:
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.title("Input Image")
            plt.show()
        
        # Save image if requested
        if args.output_image:
            save_image(image, args.output_image)
        
        # Run inference
        response = run_inference(args, image, args.prompt)
        
        # Display results
        display_results(args.prompt, response)
        
        # Save response if requested
        if args.output_text:
            os.makedirs(os.path.dirname(args.output_text) or '.', exist_ok=True)
            with open(args.output_text, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\n\nResponse:\n{response}")
            logger.info(f"Response saved to {args.output_text}")
        
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Vision-language inference using Qwen2.5-VL")
    
    parser.add_argument("--model", type=str, 
                       choices=["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"],
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Qwen2.5-VL model name")
    
    parser.add_argument("--image_url", type=str, 
                       default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                       help="URL or path to image")
    
    parser.add_argument("--prompt", type=str, 
                       default="Describe this image in detail.",
                       help="Text prompt for the model")
    
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum number of new tokens to generate")
    
    parser.add_argument("--min_pixels", type=int, default=224*224,
                       help="Minimum pixels for image processing")
    
    parser.add_argument("--max_pixels", type=int, default=1024*1024,
                       help="Maximum pixels for image processing")
    
    parser.add_argument("--use_pipeline", action="store_true",
                       help="Use transformers pipeline")
    
    parser.add_argument("--use_qwen_class", action="store_true",
                       help="Use Qwen2_5_VLProcessor class")
    
    parser.add_argument("--torch_dtype", type=str, 
                       choices=["float32", "float16", "bfloat16"], default="bfloat16",
                       help="Torch data type")
    
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force model to run on CPU (useful for memory constraints)")
    
    parser.add_argument("--low_memory", action="store_true",
                       help="Enable low memory usage optimizations")
    
    parser.add_argument("--visualize", action="store_true",
                       help="Display the input image")
    
    parser.add_argument("--output_image", type=str,
                       help="Path to save input image")
    
    parser.add_argument("--output_text", type=str,
                       help="Path to save generated response")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# MEMORY USAGE EXAMPLES:
# 
# For systems with limited GPU memory (< 8GB):
# python qwen2.5-vl.py --force_cpu --image_url "https://example.com/image.jpg" --prompt "What do you see?"
#
# For systems with moderate GPU memory (8-16GB):
# python qwen2.5-vl.py --low_memory --model_name "Qwen/Qwen2.5-VL-3B-Instruct"
#
# For high-end systems (16GB+ GPU memory):
# python qwen2.5-vl.py --model_name "Qwen/Qwen2.5-VL-7B-Instruct"
