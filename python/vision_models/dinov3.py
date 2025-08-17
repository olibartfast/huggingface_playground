# https://huggingface.co/docs/transformers/model_doc/dinov3
# pip install --upgrade git+https://github.com/huggingface/transformers.git}
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

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. Custom processor will not work.")

# Updated imports based on official documentation
from transformers import (
    AutoImageProcessor, 
    AutoModel, 
    pipeline,
    DINOv3ViTImageProcessorFast
)
from transformers.image_utils import load_image
from typing import Dict, Optional, Tuple, Any
import gc

def create_custom_dinov3_processor():
    """Create a custom processor for DINOv3 using torchvision transforms."""
    if not TORCHVISION_AVAILABLE:
        print("✗ torchvision not available for custom processor")
        return None
        
    class CustomDINOv3Processor:
        def __init__(self):
            # DINOv3 typically uses similar preprocessing to DINOv2
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Standard ViT input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225]    # ImageNet std
                )
            ])
        
        def __call__(self, images, return_tensors="pt"):
            if not isinstance(images, list):
                images = [images]
            
            # Convert PIL images to tensor
            pixel_values = []
            for img in images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pixel_values.append(self.transform(img))
            
            pixel_values = torch.stack(pixel_values)
            
            if return_tensors == "pt":
                return {"pixel_values": pixel_values}
            else:
                return {"pixel_values": pixel_values.numpy()}
    
    return CustomDINOv3Processor()

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
    
    # First try the pipeline approach (recommended by documentation)
    if task == "feature_extraction":
        try:
            print("Trying pipeline approach (recommended)...")
            pipe = pipeline(
                task="image-feature-extraction",
                model=model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Pipeline loaded successfully!")
            return pipe, None  # Return pipeline instead of processor/model
        except Exception as e:
            print(f"Pipeline approach failed: {e}")
            print("Falling back to manual model loading...")
    
    # Validate model exists
    if not validate_model_exists(model_name):
        print(f"Warning: Model '{model_id}' may not exist or is not accessible.")
        print("Proceeding anyway, but this might fail...")
    
    # Try to load processor - now with multiple fallback strategies
    processor = None
    try:
        print(f"Attempting to load processor from: {model_id}")
        # Try the fast processor first (from documentation)
        try:
            processor = DINOv3ViTImageProcessorFast.from_pretrained(model_id)
            print(f"✓ Fast processor loaded successfully! Type: {type(processor)}")
        except:
            # Fallback to regular AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(model_id)
            print(f"✓ Processor loaded successfully! Type: {type(processor)}")
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error loading processor: {e}")
        
        # Check for different error types
        if "gated repo" in error_msg or "restricted" in error_msg or "authorized list" in error_msg:
            print("\n🔒 ACCESS REQUIRED: This is a gated repository!")
            print("DINOv3 models require explicit access approval from Meta/Facebook.")
            print("\nTo get access:")
            print("1. Visit: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
            print("2. Click 'Request access' and fill out the form")
            print("3. Wait for approval (usually takes a few hours to days)")
            print("4. Make sure you're logged in: huggingface-cli login")
            return None, None
        elif "Unrecognized image processor" in error_msg or "image_processor_type" in error_msg:
            print("⚠️ DINOv3 processor not recognized. Trying DINOv2 processor as fallback...")
            try:
                # Use DINOv2 processor as fallback (compatible with DINOv3)
                fallback_processor_id = "facebook/dinov2-base"
                processor = AutoImageProcessor.from_pretrained(fallback_processor_id)
                print(f"✓ Using DINOv2 processor as fallback: {type(processor)}")
            except Exception as fallback_error:
                print(f"✗ Fallback processor also failed: {fallback_error}")
                print("Creating custom processor...")
                processor = create_custom_dinov3_processor()
                if processor is None:
                    return None, None
                print("✓ Custom processor created successfully!")
        else:
            print("This usually means the model doesn't exist or you don't have access.")
            print("Available DINOv3 models (all require access):")
            print("  - dinov3-vits16-pretrain-lvd1689m")
            print("  - dinov3-vitb16-pretrain-lvd1689m")
            print("  - dinov3-vitl16-pretrain-lvd1689m")
            return None, None
    
    # Set up model with optional quantization (following documentation)
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    if use_quantization and torch.cuda.is_available():
        try:
            from transformers import TorchAoConfig
            from torchao.quantization import Int4WeightOnlyConfig
            # Following the documentation example
            quant_type = Int4WeightOnlyConfig(group_size=128)
            quantization_config = TorchAoConfig(quant_type=quant_type)
            model_kwargs["quantization_config"] = quantization_config
            print("Using Int4 quantization (following documentation)")
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
        error_msg = str(e)
        print(f"✗ OSError loading model: {e}")
        if "gated repo" in error_msg or "restricted" in error_msg or "authorized list" in error_msg:
            print("\n🔒 ACCESS REQUIRED: This is a gated repository!")
            print("You need access to both the processor AND the model.")
            print("Make sure you're logged in: huggingface-cli login")
        else:
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
    """Extract features including register tokens analysis following official documentation."""
    
    # Handle pipeline case
    if hasattr(processor, 'model'):  # This is a pipeline
        print("Using pipeline for feature extraction...")
        # For pipeline, we need to get the underlying model and processor
        actual_model = processor.model
        actual_processor = processor.tokenizer if hasattr(processor, 'tokenizer') else processor.feature_extractor
        
        # Get features using pipeline
        features = processor(image)
        
        # For detailed analysis, we need to process manually
        if hasattr(actual_processor, '__call__'):
            inputs = actual_processor(images=image, return_tensors="pt")
        else:
            # Use load_image from documentation
            from transformers.image_utils import load_image
            if isinstance(image, str):
                image = load_image(image)
            inputs = {"pixel_values": torch.tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0}
        
        device = next(actual_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = actual_model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        model_config = actual_model.config
    else:
        # Manual processing (fallback)
        inputs = processor(images=image, return_tensors="pt")
        
        # Better device handling
        device = get_device(model)
        inputs = inputs.to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        model_config = model.config

    # Following the documentation pattern
    batch_size, _, img_height, img_width = inputs['pixel_values'].shape
    patch_size = model_config.patch_size
    num_patches_height = img_height // patch_size
    num_patches_width = img_width // patch_size
    num_patches_flat = num_patches_height * num_patches_width
    
    print(f"Image size: {img_height}x{img_width}")
    print(f"Patch size: {patch_size}")
    print(f"Num patches: {num_patches_height}x{num_patches_width} = {num_patches_flat}")

    # Extract different token types following documentation
    cls_token = last_hidden_states[:, 0, :]  # CLS token
    
    # Handle register tokens if they exist (following documentation)
    num_register_tokens = getattr(model_config, 'num_register_tokens', 0)
    print(f"Num register tokens: {num_register_tokens}")
    
    if num_register_tokens > 0:
        register_tokens = last_hidden_states[:, 1:1+num_register_tokens, :]
        patch_features_flat = last_hidden_states[:, 1+num_register_tokens:, :]
    else:
        register_tokens = None
        patch_features_flat = last_hidden_states[:, 1:, :]
    
    # Reshape patch features to spatial dimensions (following documentation)
    patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
    
    # Verify shapes match documentation expectations
    expected_sequence_length = 1 + num_register_tokens + num_patches_flat
    actual_sequence_length = last_hidden_states.shape[1]
    print(f"Expected sequence length: {expected_sequence_length}")
    print(f"Actual sequence length: {actual_sequence_length}")
    
    if expected_sequence_length != actual_sequence_length:
        print("⚠️ Warning: Sequence length mismatch with documentation expectations")
    
    return {
        'cls_token': cls_token,
        'register_tokens': register_tokens,
        'patch_features': patch_features,
        'patch_features_flat': patch_features_flat,
        'spatial_dims': (num_patches_height, num_patches_width),
        'last_hidden_states': last_hidden_states,
        'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
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

def feature_extraction(image, processor_or_pipeline, model, output_path, visualize=False):
    """Perform feature extraction with DINOv3 including register token analysis."""
    print("Extracting features with DINOv3...")
    
    # Handle pipeline case
    if hasattr(processor_or_pipeline, 'model'):  # This is a pipeline
        print("Using pipeline for simple feature extraction...")
        # Simple pipeline usage
        try:
            pipeline_features = processor_or_pipeline(image)
            print(f"Pipeline features shape: {np.array(pipeline_features).shape}")
        except Exception as e:
            print(f"Pipeline extraction failed: {e}")
            pipeline_features = None
        
        # For detailed analysis, extract manually
        features = extract_features_with_registers(image, processor_or_pipeline, None)
    else:
        # Manual approach
        features = extract_features_with_registers(image, processor_or_pipeline, model)
    
    cls_token = features['cls_token']
    register_tokens = features['register_tokens']
    patch_features = features['patch_features']
    pooler_output = features.get('pooler_output')
    
    print("CLS token shape:", cls_token.shape)
    print("Patch features shape:", patch_features.shape)
    if register_tokens is not None:
        print("Register tokens shape:", register_tokens.shape)
    if pooler_output is not None:
        print("Pooler output shape:", pooler_output.shape)

    # Save embeddings
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    embedding_path = os.path.splitext(output_path)[0] + "_cls_embedding.txt"
    np.savetxt(embedding_path, cls_token.cpu().numpy())
    print(f"CLS embedding saved to {embedding_path}")
    
    if pooler_output is not None:
        pooler_path = os.path.splitext(output_path)[0] + "_pooler_output.txt"
        np.savetxt(pooler_path, pooler_output.cpu().numpy())
        print(f"Pooler output saved to {pooler_path}")
    
    if register_tokens is not None:
        register_path = os.path.splitext(output_path)[0] + "_register_tokens.txt"
        np.savetxt(register_path, register_tokens.cpu().numpy().reshape(-1, register_tokens.shape[-1]))
        print(f"Register tokens saved to {register_path}")

    # Visualize feature maps
    if visualize or output_path:
        feature_map = patch_features.mean(dim=-1).cpu().numpy()[0]  # Average across channels
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8) * 255
        
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Feature map
        plt.subplot(2, 3, 2)
        plt.imshow(feature_map, cmap='viridis')
        plt.title("DINOv3 Feature Map (Patch Tokens)")
        plt.axis('off')
        
        # CLS token visualization
        plt.subplot(2, 3, 3)
        cls_norm = torch.norm(cls_token, dim=-1).cpu().numpy()[0]
        plt.bar([0], [cls_norm])
        plt.title("CLS Token Magnitude")
        plt.ylabel("L2 Norm")
        
        # Register token visualization (if available)
        if register_tokens is not None:
            plt.subplot(2, 3, 4)
            register_norms = torch.norm(register_tokens, dim=-1).cpu().numpy()[0]
            plt.bar(range(len(register_norms)), register_norms)
            plt.title("Register Token Magnitudes")
            plt.xlabel("Register Token Index")
            plt.ylabel("L2 Norm")
        
        # Pooler output (if available)
        if pooler_output is not None:
            plt.subplot(2, 3, 5)
            pooler_norm = torch.norm(pooler_output, dim=-1).cpu().numpy()[0]
            plt.bar([0], [pooler_norm])
            plt.title("Pooler Output Magnitude")
            plt.ylabel("L2 Norm")
        
        # Feature distribution
        plt.subplot(2, 3, 6)
        all_features = features['last_hidden_states'].cpu().numpy().flatten()
        plt.hist(all_features, bins=50, alpha=0.7)
        plt.title("Feature Distribution")
        plt.xlabel("Feature Value")
        plt.ylabel("Count")
        
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
        processor_or_pipeline, model = setup_model(args.model, args.task, args.use_quantization)
        if processor_or_pipeline is None:
            print("Model setup failed. Exiting.")
            return

        # Load image (using transformers load_image utility when possible)
        try:
            if args.local_image:
                image = load_image_from_source(None, args.local_image)
            else:
                # Try using transformers load_image utility
                try:
                    from transformers.image_utils import load_image
                    image = load_image(args.image_url)
                    print(f"Image loaded with transformers utility! Size: {image.size}")
                except:
                    # Fallback to custom loader
                    image = load_image_from_source(args.image_url, None)
                    print(f"Image loaded with custom loader! Size: {image.size}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Execute task
        task_handler = get_task_handler(args.task)
        if task_handler:
            try:
                # Handle both pipeline and manual approaches
                if hasattr(processor_or_pipeline, 'model'):  # This is a pipeline
                    print("Using pipeline-based approach...")
                    result = task_handler(
                        image, processor_or_pipeline, None,  # model is None for pipeline
                        args.output_image, args.visualize
                    )
                else:
                    print("Using manual model approach...")
                    result = task_handler(
                        image, processor_or_pipeline, model, 
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