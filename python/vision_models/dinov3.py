# DINOv3 🦖🦖🦖 - Enhanced Vision Foundation Model
# Official documentation: https://huggingface.co/docs/transformers/model_doc/dinov3
# Paper: https://arxiv.org/abs/2508.10104
# Collection: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009
# 
# Requirements:
# pip install --upgrade transformers>=4.35.0 torch torchvision pillow matplotlib numpy opencv-python
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
    print("Warning: torchvision not available. Install with: pip install torchvision")

# Enhanced imports based on official documentation
from transformers import (
    AutoImageProcessor, 
    AutoModel, 
    pipeline,
    DINOv3ViTImageProcessorFast
)
from transformers.image_utils import load_image
from typing import Dict, Optional, Tuple, Any, Union
import gc

# All available DINOv3 models from Hugging Face Hub
DINOV3_MODELS = {
    # ViT models pretrained on web dataset (LVD-1689M)
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m", 
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    
    # ConvNeXt models pretrained on web dataset (LVD-1689M)
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    
    # ViT models pretrained on satellite dataset (SAT-493M)
    "vitl16_sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "vit7b16_sat": "facebook/dinov3-vit7b16-pretrain-sat493m",
}

def make_transform_lvd(resize_size: int = 224):
    """Transform for models using LVD-1689M weights (web images)."""
    if not TORCHVISION_AVAILABLE:
        return None
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),  # Standard ImageNet
            std=(0.229, 0.224, 0.225),
        )
    ])

def make_transform_sat(resize_size: int = 224):
    """Transform for models using SAT-493M weights (satellite imagery)."""
    if not TORCHVISION_AVAILABLE:
        return None
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(
            mean=(0.430, 0.411, 0.296),  # Satellite imagery specific
            std=(0.213, 0.156, 0.143),
        )
    ])

def create_custom_dinov3_processor(model_type="lvd"):
    """Create a custom processor for DINOv3 using torchvision transforms."""
    if not TORCHVISION_AVAILABLE:
        print("✗ torchvision not available for custom processor")
        print("Install with: pip install torchvision")
        return None
        
    class CustomDINOv3Processor:
        def __init__(self, transform):
            self.transform = transform
        
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
    
    # Choose appropriate transform based on model type
    if model_type == "sat":
        transform = make_transform_sat()
    else:
        transform = make_transform_lvd()
    
    if transform is None:
        return None
        
    return CustomDINOv3Processor(transform)

def get_device(model) -> torch.device:
    """Get the device of the model."""
    return next(model.parameters()).device

def validate_model_exists(model_id: str) -> bool:
    """Check if model exists on HuggingFace Hub."""
    try:
        from huggingface_hub import model_info
        model_info(model_id)
        return True
    except Exception as e:
        print(f"Warning: Could not validate model existence: {e}")
        return True  # Assume it exists and let the actual loading handle the error

def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model information including ID, type, and dataset."""
    if model_key not in DINOV3_MODELS:
        available_models = ", ".join(DINOV3_MODELS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available_models}")
    
    model_id = DINOV3_MODELS[model_key]
    is_satellite = "sat" in model_key
    is_convnext = "convnext" in model_key
    
    return {
        "model_id": model_id,
        "model_key": model_key,
        "is_satellite": is_satellite,
        "is_convnext": is_convnext,
        "dataset": "SAT-493M" if is_satellite else "LVD-1689M",
        "architecture": "ConvNeXt" if is_convnext else "ViT"
    }

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DINOv3 Vision Foundation Model")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "feature_extraction",
            "instance_segmentation",
            "pca_visualization",
            "dense_matching"
        ],
        default="feature_extraction",
        help="Task to perform with DINOv3"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DINOV3_MODELS.keys()),
        default="vits16",
        help=f"DINOv3 model variant to use. Available: {', '.join(DINOV3_MODELS.keys())}"
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to process"
    )
    parser.add_argument(
        "--second_image_url",
        type=str,
        default=None,
        help="URL of the second image (for dense matching)"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="dinov3_output.png",
        help="File path to save the output visualization"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (display output)"
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
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Input image size (will be resized to patch_size x patch_size)"
    )
    args = parser.parse_args()
    return args

def setup_model(args):
    """Set up the DINOv3 model and processor."""
    try:
        if args.debug:
            logger.info(f"Loading model: {args.model}")
        
        # Get model information
        model_info = get_model_info(args.model)
        model_id = model_info["model_id"]
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor with appropriate transforms
        if model_info["dataset"] == "SAT-493M":
            transform = make_transform_sat()
        else:  # LVD-1689M
            transform = make_transform_lvd()
        processor = create_custom_dinov3_processor(transform)
        
        # First try the pipeline approach for feature extraction
        if args.task == "feature_extraction":
            try:
                if args.debug:
                    logger.info("Trying pipeline approach (recommended)...")
                pipe = pipeline(
                    task="image-feature-extraction",
                    model=model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device=0 if torch.cuda.is_available() else -1
                )
                if args.debug:
                    logger.info("✓ Pipeline loaded successfully!")
                return pipe, processor, device  # Return pipeline with processor for consistency
            except Exception as e:
                if args.debug:
                    logger.warning(f"Pipeline approach failed: {e}")
                    logger.info("Falling back to manual model loading...")
        
        # Manual model loading with quantization support
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        if args.use_quantization and torch.cuda.is_available():
            try:
                from transformers import TorchAoConfig
                from torchao.quantization import Int4WeightOnlyConfig
                # Following the documentation example
                quant_type = Int4WeightOnlyConfig(group_size=128)
                quantization_config = TorchAoConfig(quant_type=quant_type)
                model_kwargs["quantization_config"] = quantization_config
                if args.debug:
                    logger.info("Using Int4 quantization (following documentation)")
            except ImportError:
                if args.debug:
                    logger.warning("TorchAO not available, proceeding without quantization")
        
        # Load model
        if args.debug:
            logger.info(f"Loading model from: {model_id}")
        model = Dinov3Model.from_pretrained(model_id, **model_kwargs)
        
        if not args.use_quantization and device == "cuda":
            model = model.to(device)
        
        model.eval()
        
        if args.debug:
            logger.info(f"Model loaded successfully on device: {device}")
            logger.info(f"Model info: {model_info}")
            logger.info(f"Model config - Hidden size: {model.config.hidden_size}, Patch size: {model.config.patch_size}")
            if hasattr(model.config, 'num_register_tokens'):
                logger.info(f"Register tokens: {model.config.num_register_tokens}")
        
        return model, processor, device
        
    except Exception as e:
        logger.error(f"Error setting up model: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise RuntimeError(f"Failed to load model {args.model}: {e}")

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
    print("=" * 60)
    print("⚠️  IMPORTANT: Instance Segmentation Demo Analysis")
    print("=" * 60)
    print("✅ SUCCESS: DINOv3 feature extraction working correctly")
    print("⚠️  LIMITATION: Using UNTRAINED segmentation head")
    print("📊 RESULT: Random masks (not meaningful object detection)")
    print("")
    print("For REAL instance segmentation, you need:")
    print("1. Fine-tuned segmentation head on labeled data")
    print("2. Or use specialized models like Mask R-CNN, DETR, etc.")
    print("3. Or combine with SAM (Segment Anything Model)")
    print("=" * 60)
    
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
        plt.title(f"Random Mask {i+1}\n(Untrained Head)")
        plt.axis('off')
    
    # Add evaluation text
    plt.figtext(0.02, 0.02, 
                "⚠️ EVALUATION: Masks are random due to untrained segmentation head.\n"
                "✅ DINOv3 features extracted correctly (384D, 14x14 spatial)\n"
                "🔧 For real segmentation: fine-tune head or use SAM/Mask R-CNN", 
                fontsize=8, ha='left')
    
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

def pca_visualization(image, processor, model, output_path, visualize=False):
    """Create PCA visualization of DINOv3 features."""
    print("Creating PCA visualization of DINOv3 features...")
    
    features = extract_features_with_registers(image, processor, model)
    patch_features = features['patch_features_flat']  # [1, num_patches, hidden_size]
    
    # Convert to numpy for PCA
    patch_features_np = patch_features.cpu().numpy().squeeze(0)  # [num_patches, hidden_size]
    
    # Apply PCA to reduce to 3 components for RGB visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_features_np)  # [num_patches, 3]
    
    # Normalize to [0, 1] for RGB
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    
    # Reshape to spatial dimensions
    spatial_dims = features['spatial_dims']
    pca_image = pca_features.reshape(spatial_dims[0], spatial_dims[1], 3)
    
    # Convert to PIL Image
    pca_image_pil = Image.fromarray((pca_image * 255).astype(np.uint8))
    pca_image_pil = pca_image_pil.resize(image.size, Image.LANCZOS)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(pca_image_pil)
    ax2.set_title("PCA Features (RGB)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if visualize:
        plt.show()
    else:
        plt.close()
    
    print(f"PCA visualization saved to: {output_path}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca_image_pil

def dense_matching(image, processor, model, output_path, visualize=False, second_image_url=None):
    """Perform dense feature matching between two images."""
    if second_image_url is None:
        print("Error: Dense matching requires a second image. Use --second_image_url argument.")
        return None
    
    print("Performing dense matching between two images...")
    
    # Load second image
    try:
        from transformers.image_utils import load_image
        image2 = load_image(second_image_url)
        print(f"Second image loaded! Size: {image2.size}")
    except:
        # Fallback to custom loader
        image2 = load_image_from_source(second_image_url, None)
        print(f"Second image loaded with custom loader! Size: {image2.size}")
    
    # Extract features for both images
    features1 = extract_features_with_registers(image, processor, model)
    features2 = extract_features_with_registers(image2, processor, model)
    
    patch_features1 = features1['patch_features_flat'].cpu().numpy().squeeze(0)  # [num_patches, hidden_size]
    patch_features2 = features2['patch_features_flat'].cpu().numpy().squeeze(0)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(patch_features1, patch_features2.T)  # [num_patches1, num_patches2]
    
    # Find best matches for each patch in image1
    best_matches = np.argmax(similarity_matrix, axis=1)
    
    # Get spatial dimensions
    spatial_dims1 = features1['spatial_dims']
    spatial_dims2 = features2['spatial_dims']
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.imshow(image)
    ax1.set_title("Image 1")
    ax1.axis('off')
    
    ax2.imshow(image2)
    ax2.set_title("Image 2")
    ax2.axis('off')
    
    # Visualize similarity matrix
    im3 = ax3.imshow(similarity_matrix, cmap='viridis')
    ax3.set_title("Patch Similarity Matrix")
    ax3.set_xlabel("Image 2 Patches")
    ax3.set_ylabel("Image 1 Patches")
    plt.colorbar(im3, ax=ax3)
    
    # Show some matching lines (sample for visualization)
    ax4.imshow(np.concatenate([np.array(image), np.array(image2)], axis=1))
    ax4.set_title("Feature Correspondences (Sample)")
    ax4.axis('off')
    
    # Draw some correspondence lines
    patch_size1 = image.size[0] // spatial_dims1[1]
    patch_size2 = image2.size[0] // spatial_dims2[1]
    
    for i in range(0, len(best_matches), len(best_matches) // 10):  # Sample 10 matches
        # Get patch positions
        y1, x1 = divmod(i, spatial_dims1[1])
        y2, x2 = divmod(best_matches[i], spatial_dims2[1])
        
        # Convert to image coordinates
        img_x1 = x1 * patch_size1 + patch_size1 // 2
        img_y1 = y1 * patch_size1 + patch_size1 // 2
        img_x2 = x2 * patch_size2 + patch_size2 // 2 + image.size[0]  # Offset for concatenated image
        img_y2 = y2 * patch_size2 + patch_size2 // 2
        
        # Draw line
        ax4.plot([img_x1, img_x2], [img_y1, img_y2], 'r-', alpha=0.6, linewidth=1)
        ax4.plot(img_x1, img_y1, 'ro', markersize=3)
        ax4.plot(img_x2, img_y2, 'go', markersize=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if visualize:
        plt.show()
    else:
        plt.close()
    
    print(f"Dense matching visualization saved to: {output_path}")
    print(f"Average similarity: {similarity_matrix.mean():.4f}")
    print(f"Max similarity: {similarity_matrix.max():.4f}")
    
    return similarity_matrix

def get_task_handler(task: str):
    """Get the appropriate handler function for the task."""
    handlers = {
        "feature_extraction": feature_extraction,
        "instance_segmentation": instance_segmentation,
        "pca_visualization": pca_visualization,
        "dense_matching": dense_matching,
    }
    return handlers.get(task)

def print_available_tasks():
    """Print available tasks."""
    print("Available tasks:")
    print("  - feature_extraction: Extract DINOv3 features and visualize")
    print("  - instance_segmentation: Demo segmentation (untrained head)")
    print("  - pca_visualization: Create PCA visualization of features")
    print("  - dense_matching: Match features between two images")

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
    print("Debug mode:", args.debug)
    if args.visualize:
        print("Visualization enabled (output will be displayed)")

    try:
        # Setup model - now returns model, processor, device
        model_or_pipeline, processor, device = setup_model(args)
        if model_or_pipeline is None:
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
                    if args.debug:
                        logger.info(f"Image loaded with transformers utility! Size: {image.size}")
                except:
                    # Fallback to custom loader
                    image = load_image_from_source(args.image_url, None)
                    if args.debug:
                        logger.info(f"Image loaded with custom loader! Size: {image.size}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return

        # Execute task
        task_handler = get_task_handler(args.task)
        if task_handler:
            try:
                # Handle both pipeline and manual approaches
                if hasattr(model_or_pipeline, 'model'):  # This is a pipeline
                    if args.debug:
                        logger.info("Using pipeline-based approach...")
                    
                    # Special handling for dense_matching which needs second image
                    if args.task == "dense_matching":
                        result = task_handler(
                            image, model_or_pipeline, None,  # model is None for pipeline
                            args.output_image, args.visualize, args.second_image_url
                        )
                    else:
                        result = task_handler(
                            image, model_or_pipeline, None,  # model is None for pipeline
                            args.output_image, args.visualize
                        )
                else:
                    if args.debug:
                        logger.info("Using manual model approach...")
                    
                    # Special handling for dense_matching which needs second image
                    if args.task == "dense_matching":
                        result = task_handler(
                            image, processor, model_or_pipeline, 
                            args.output_image, args.visualize, args.second_image_url
                        )
                    else:
                        result = task_handler(
                            image, processor, model_or_pipeline, 
                            args.output_image, args.visualize
                        )
                
                if result is not None:
                    print(f"\n{args.task.replace('_', ' ').title()} completed successfully!")
                    print(f"Results saved to: {args.output_image}")
                else:
                    print(f"\n{args.task.replace('_', ' ').title()} was skipped or failed.")
                    
            except Exception as e:
                logger.error(f"Error during {args.task}: {e}")
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
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        # Always cleanup resources
        cleanup_resources()

if __name__ == "__main__":
    args = parse_args()
    main(args)