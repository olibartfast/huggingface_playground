# https://huggingface.co/docs/transformers/model_doc/dinov3
import requests
from PIL import Image
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.spatial.distance import cdist

from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
from typing import Dict, Optional, Tuple, Any
import gc

def get_device(model) -> torch.device:
    """Get the device of the model."""
    return next(model.parameters()).device

def validate_model_exists(model_name: str) -> bool:
    """Check if model exists on HuggingFace."""
    try:
        from huggingface_hub import model_info
        model_info(f"facebook/{model_name}")
        return True
    except Exception as e:
        # Don't fail completely, just warn
        print(f"Warning: Could not validate model existence: {e}")
        return True  # Assume it exists and let the actual loading handle the error

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "feature_extraction",
            "instance_segmentation",
        ],
        default="feature_extraction",
        help="Task to perform with DINOv3 (more tasks coming soon)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "dinov3-vits16-pretrain-lvd1689m",
            "dinov3-vitb16-pretrain-lvd1689m", 
            "dinov3-vitl16-pretrain-lvd1689m",
        ],
        default="dinov3-vits16-pretrain-lvd1689m",
        help="DINOv3 model to use (requires access approval from Meta/Facebook)"
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the first image to process (for image-based tasks)"
    )
    parser.add_argument(
        "--second_image_url",
        type=str,
        default=None,
        help="URL of the second image (for sparse_matching, dense_matching, instance_retrieval)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to the video file (for video_classification)"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="dinov3_output.png",
        help="File path to save the output image"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (display output image or feature map)"
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use quantization to reduce memory usage"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed error messages"
    )
    parser.add_argument(
        "--local_image",
        type=str,
        default=None,
        help="Path to local image file (alternative to image_url)"
    )
    args = parser.parse_args()
    return args

def setup_model(model_name, task, use_quantization=False):
    """Sets up the processor and model based on the model name and task."""
    model_id = f"facebook/{model_name}"
    print(f"Loading {model_name} model for {task}...")
    
    # Validate model exists
    if not validate_model_exists(model_name):
        print(f"Warning: Model '{model_id}' may not exist or is not accessible.")
        print("Proceeding anyway, but this might fail...")
    
    try:
        print(f"Attempting to load processor from: {model_id}")
        processor = AutoImageProcessor.from_pretrained(model_id)
        print(f"✓ Processor loaded successfully! Type: {type(processor)}")
    except OSError as e:
        error_msg = str(e)
        print(f"✗ OSError loading processor: {e}")
        
        if "gated repo" in error_msg or "restricted" in error_msg or "authorized list" in error_msg:
            print("\n🔒 ACCESS REQUIRED: This is a gated repository!")
            print("DINOv3 models require explicit access approval from Meta/Facebook.")
            print("\nTo get access:")
            print("1. Visit: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
            print("2. Click 'Request access' and fill out the form")
            print("3. Wait for approval (usually takes a few hours to days)")
            print("4. Make sure you're logged in: huggingface-cli login")
        else:
            print("This usually means the model doesn't exist or you don't have access.")
            print("Available DINOv3 models (all require access):")
            print("  - dinov3-vits16-pretrain-lvd1689m")
            print("  - dinov3-vitb16-pretrain-lvd1689m")
            print("  - dinov3-vitl16-pretrain-lvd1689m")
        return None, None
    except Exception as e:
        print(f"✗ Unexpected error loading processor: {e}")
        return None, None
    
    # Set up model with optional quantization
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "attn_implementation": "sdpa" if torch.cuda.is_available() else None,
    }
    
    if use_quantization and torch.cuda.is_available():
        try:
            from transformers import TorchAoConfig
            from torchao.quantization import Int4WeightOnlyConfig
            quant_type = Int4WeightOnlyConfig(group_size=128)
            quantization_config = TorchAoConfig(quant_type=quant_type)
            model_kwargs["quantization_config"] = quantization_config
            print("Using Int4 quantization")
        except ImportError:
            print("TorchAO not available, proceeding without quantization")
    
    try:
        print(f"Attempting to load model from: {model_id}")
        model = AutoModel.from_pretrained(model_id, **model_kwargs)
        print("✓ Model loaded successfully!")
    except torch.cuda.OutOfMemoryError:
        print("✗ CUDA out of memory. Try using --use_quantization flag or a smaller model.")
        return None, None
    except OSError as e:
        print(f"✗ OSError loading model: {e}")
        print("This usually means the model doesn't exist or you don't have access.")
        return None, None
    except Exception as e:
        print(f"✗ Unexpected error loading model: {e}")
        return None, None
    
    if torch.cuda.is_available() and not use_quantization:
        try:
            model = model.to("cuda")
            print("✓ Model moved to CUDA")
        except torch.cuda.OutOfMemoryError:
            print("✗ CUDA out of memory when moving model. Try --use_quantization flag.")
            return None, None
    
    print(f"✓ {model_name} setup completed!")
    print(f"Model config - Hidden size: {model.config.hidden_size}, Patch size: {model.config.patch_size}")
    if hasattr(model.config, 'num_register_tokens'):
        print(f"Register tokens: {model.config.num_register_tokens}")
    
    return processor, model

class InstanceSegmentationHead(nn.Module):
    """Enhanced instance segmentation head for DINOv3."""
    def __init__(self, in_channels=384, num_instances=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 4, num_instances, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return self.sigmoid(x)

def extract_features_with_registers(image: Image.Image, processor, model) -> Dict[str, torch.Tensor]:
    """Extract features including register tokens analysis."""
    inputs = processor(images=image, return_tensors="pt")
    
    # Better device handling
    device = get_device(model)
    inputs = inputs.to(device)

    batch_size, _, img_height, img_width = inputs.pixel_values.shape
    patch_size = model.config.patch_size
    num_patches_height = img_height // patch_size
    num_patches_width = img_width // patch_size
    num_patches_flat = num_patches_height * num_patches_width
    
    with torch.inference_mode():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # Extract different token types
    cls_token = last_hidden_states[:, 0, :]  # CLS token
    
    # Handle register tokens if they exist
    num_register_tokens = getattr(model.config, 'num_register_tokens', 0)
    if num_register_tokens > 0:
        register_tokens = last_hidden_states[:, 1:1+num_register_tokens, :]
        patch_features_flat = last_hidden_states[:, 1+num_register_tokens:, :]
    else:
        register_tokens = None
        patch_features_flat = last_hidden_states[:, 1:, :]
    
    # Reshape patch features to spatial dimensions
    patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
    
    return {
        'cls_token': cls_token,
        'register_tokens': register_tokens,
        'patch_features': patch_features,
        'patch_features_flat': patch_features_flat,
        'spatial_dims': (num_patches_height, num_patches_width)
    }

def instance_segmentation(image, processor, model, output_path, visualize=False):
    """Perform instance segmentation with DINOv3."""
    print("WARNING: Instance segmentation with DINOv3 requires fine-tuning for accurate results.")
    print("This is a demonstration using an UNTRAINED segmentation head.")
    print("The results will be RANDOM and not meaningful for actual object detection!")
    
    response = input("Continue with demo anyway? (y/n): ")
    if response.lower() != 'y':
        print("Skipping instance segmentation.")
        return None
    
    features = extract_features_with_registers(image, processor, model)
    
    in_channels = model.config.hidden_size
    seg_head = InstanceSegmentationHead(in_channels=in_channels, num_instances=10)
    
    # Move to same device as model
    device = get_device(model)
    seg_head = seg_head.to(device)

    # Use patch features for segmentation
    patch_features = features['patch_features']
    batch_size, h, w, channels = patch_features.shape
    patch_features = patch_features.permute(0, 3, 1, 2)  # [B, C, H, W]

    print(f"Patch features shape: {patch_features.shape}")
    
    with torch.no_grad():
        masks = seg_head(patch_features)

    print(f"Generated masks shape: {masks.shape}")
    
    # Interpolate to original image size
    masks = torch.nn.functional.interpolate(
        masks, size=(image.size[1], image.size[0]), 
        mode='bilinear', align_corners=False
    )
    masks = masks[0].cpu().numpy()  # [num_instances, height, width]

    print(f"Final masks shape: {masks.shape}")
    
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Show top 5 instance masks
    for i in range(min(5, masks.shape[0])):
        plt.subplot(2, 3, i + 2)
        mask = masks[i]
        plt.imshow(mask, cmap='hot')
        plt.title(f"Random Mask {i+1}\n(Not Meaningful)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Segmentation visualization saved to {output_path}")

    if visualize:
        plt.show()
    plt.close()

    return masks

def feature_extraction(image, processor, model, output_path, visualize=False):
    """Perform feature extraction with DINOv3 including register token analysis."""
    print("Extracting features with DINOv3...")
    features = extract_features_with_registers(image, processor, model)
    
    cls_token = features['cls_token']
    register_tokens = features['register_tokens']
    patch_features = features['patch_features']
    
    print("CLS token shape:", cls_token.shape)
    print("Patch features shape:", patch_features.shape)
    if register_tokens is not None:
        print("Register tokens shape:", register_tokens.shape)

    # Save embeddings
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    embedding_path = os.path.splitext(output_path)[0] + "_cls_embedding.txt"
    np.savetxt(embedding_path, cls_token.cpu().numpy())
    print(f"CLS embedding saved to {embedding_path}")
    
    if register_tokens is not None:
        register_path = os.path.splitext(output_path)[0] + "_register_tokens.txt"
        np.savetxt(register_path, register_tokens.cpu().numpy().reshape(-1, register_tokens.shape[-1]))
        print(f"Register tokens saved to {register_path}")

    # Visualize feature maps
    if visualize or output_path:
        feature_map = patch_features.mean(dim=-1).cpu().numpy()[0]  # Average across channels
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8) * 255
        
        plt.figure(figsize=(12, 8))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Feature map
        plt.subplot(2, 2, 2)
        plt.imshow(feature_map, cmap='viridis')
        plt.title("DINOv3 Feature Map (Patch Tokens)")
        plt.axis('off')
        
        # Register token visualization (if available)
        if register_tokens is not None:
            plt.subplot(2, 2, 3)
            register_norms = torch.norm(register_tokens, dim=-1).cpu().numpy()[0]
            plt.bar(range(len(register_norms)), register_norms)
            plt.title("Register Token Magnitudes")
            plt.xlabel("Register Token Index")
            plt.ylabel("L2 Norm")
        
        # CLS token visualization
        plt.subplot(2, 2, 4)
        cls_norm = torch.norm(cls_token, dim=-1).cpu().numpy()[0]
        plt.bar([0], [cls_norm])
        plt.title("CLS Token Magnitude")
        plt.ylabel("L2 Norm")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to {output_path}")
        
        if visualize:
            plt.show()
        plt.close()

    return features

def cleanup_resources():
    """Clean up GPU memory and resources."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

def load_image_from_source(image_url=None, local_image=None):
    """Load image from URL or local file."""
    if local_image:
        if not os.path.exists(local_image):
            raise FileNotFoundError(f"Local image file not found: {local_image}")
        print(f"Loading local image: {local_image}")
        return Image.open(local_image)
    elif image_url:
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw)
    else:
        raise ValueError("Either image_url or local_image must be provided")

def get_task_handler(task: str):
    """Get the appropriate handler function for the task."""
    handlers = {
        "feature_extraction": feature_extraction,
        "instance_segmentation": instance_segmentation,
    }
    return handlers.get(task)

def print_available_tasks():
    """Print available tasks."""
    print("Available tasks:")
    print("  - feature_extraction: Extract DINOv3 features and visualize")
    print("  - instance_segmentation: Demo segmentation (untrained head)")

def main(args):
    """Main function to perform the specified task with DINOv3."""
    print("=" * 50)
    print("DINOv3 Vision Foundation Model")
    print("=" * 50)
    print("Task selected:", args.task)
    print("Model selected:", args.model)
    print("Image URL:", args.image_url if not args.local_image else "N/A")
    print("Local image:", args.local_image if args.local_image else "N/A")
    print("Output image path:", args.output_image)
    print("Quantization:", args.use_quantization)
    if args.visualize:
        print("Visualization enabled (output will be displayed)")

    try:
        # Setup model
        processor, model = setup_model(args.model, args.task, args.use_quantization)
        if processor is None or model is None:
            print("Model setup failed. Exiting.")
            return

        # Load image
        try:
            image = load_image_from_source(args.image_url, args.local_image)
            print(f"Image loaded successfully! Size: {image.size}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Execute task
        task_handler = get_task_handler(args.task)
        if task_handler:
            try:
                result = task_handler(
                    image, processor, model, 
                    args.output_image, args.visualize
                )
                
                if result is not None:
                    print(f"\n{args.task.replace('_', ' ').title()} completed successfully!")
                    print(f"Results saved to: {args.output_image}")
                else:
                    print(f"\n{args.task.replace('_', ' ').title()} was skipped or failed.")
                    
            except Exception as e:
                print(f"Error during {args.task}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        else:
            print(f"Task '{args.task}' not implemented.")
            print_available_tasks()

        print("\n" + "=" * 50)
        print("Process completed!")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        # Always cleanup resources
        cleanup_resources()

if __name__ == "__main__":
    args = parse_args()
    main(args)