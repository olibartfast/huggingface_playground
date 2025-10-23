# VitPose implementation for pose estimation
# https://huggingface.co/docs/transformers/model_doc/vitpose
import requests
from PIL import Image, ImageDraw
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import json

from transformers import VitPoseImageProcessor, VitPoseForPoseEstimation
from typing import Dict, List, Tuple, Optional, Any
import gc

def get_device(model) -> torch.device:
    """Get the device of the model."""
    return next(model.parameters()).device

def validate_model_exists(model_name: str) -> bool:
    """Check if model exists on HuggingFace."""
    try:
        from huggingface_hub import model_info
        model_info(model_name)
        return True
    except Exception as e:
        print(f"Warning: Could not validate model existence: {e}")
        return True

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="VitPose - Vision Transformer for Pose Estimation")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "nielsr/vitpose-base-simple",
            "nielsr/vitpose-large-simple",
            "nielsr/vitpose-huge-simple",
        ],
        default="nielsr/vitpose-base-simple",
        help="VitPose model to use for pose estimation"
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://images.unsplash.com/photo-1518611012118-696072aa579a?w=500",
        help="URL of the image to process for pose estimation"
    )
    parser.add_argument(
        "--local_image",
        type=str,
        default=None,
        help="Path to local image file (alternative to image_url)"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="vitpose_output.png",
        help="File path to save the output image with pose visualization"
    )
    parser.add_argument(
        "--output_keypoints",
        type=str,
        default="vitpose_keypoints.json",
        help="File path to save the detected keypoints in JSON format"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (display output image with pose overlay)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for keypoint visibility (0.0-1.0)"
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
        "--draw_skeleton",
        action="store_true",
        help="Draw skeleton connections between keypoints"
    )
    parser.add_argument(
        "--keypoint_size",
        type=int,
        default=3,
        help="Size of keypoint circles in visualization"
    )
    return parser.parse_args()

def setup_model(model_name: str, use_quantization: bool = False):
    """Sets up the processor and model for VitPose."""
    print(f"Loading {model_name} model for pose estimation...")
    
    # Validate model exists
    if not validate_model_exists(model_name):
        print(f"Warning: Model '{model_name}' may not exist or is not accessible.")
        print("Proceeding anyway, but this might fail...")
    
    try:
        print(f"Attempting to load processor from: {model_name}")
        processor = VitPoseImageProcessor.from_pretrained(model_name)
        print(f"✓ Processor loaded successfully! Type: {type(processor)}")
    except Exception as e:
        print(f"✗ Error loading processor: {e}")
        print("Available VitPose models:")
        print("  - nielsr/vitpose-base-simple")
        print("  - nielsr/vitpose-large-simple") 
        print("  - nielsr/vitpose-huge-simple")
        return None, None
    
    # Set up model with optional quantization
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    if use_quantization and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            print("Using 4-bit quantization")
        except ImportError:
            print("BitsAndBytesConfig not available, proceeding without quantization")
    
    try:
        print(f"Attempting to load model from: {model_name}")
        model = VitPoseForPoseEstimation.from_pretrained(model_name, **model_kwargs)
        print("✓ Model loaded successfully!")
    except torch.cuda.OutOfMemoryError:
        print("✗ CUDA out of memory. Try using --use_quantization flag or a smaller model.")
        return None, None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None
    
    if torch.cuda.is_available() and not use_quantization:
        try:
            model = model.to("cuda")
            print("✓ Model moved to CUDA")
        except torch.cuda.OutOfMemoryError:
            print("✗ CUDA out of memory when moving model. Try --use_quantization flag.")
            return None, None
    
    print(f"✓ VitPose setup completed!")
    print(f"Model config - Hidden size: {model.config.hidden_size}")
    print(f"Number of keypoints: {model.config.num_keypoints}")
    
    return processor, model

# COCO keypoint connections for skeleton drawing
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # head
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # torso
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],         # arms
    [2, 4], [3, 5], [4, 6], [5, 7]                     # more connections
]

# COCO keypoint names
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def load_image_from_source(image_url: Optional[str] = None, local_image: Optional[str] = None) -> Image.Image:
    """Load image from URL or local file."""
    if local_image:
        if not os.path.exists(local_image):
            raise FileNotFoundError(f"Local image file not found: {local_image}")
        print(f"Loading local image: {local_image}")
        return Image.open(local_image).convert('RGB')
    elif image_url:
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert('RGB')
    else:
        raise ValueError("Either image_url or local_image must be provided")

def detect_poses(image: Image.Image, processor, model, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """Detect poses in the image using VitPose."""
    print("Running pose detection...")
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Move to same device as model
    device = get_device(model)
    inputs = inputs.to(device)
    
    print(f"Input image tensor shape: {inputs.pixel_values.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract predictions
    predictions = outputs.poses[0]  # Get first (and likely only) image predictions
    print(f"Raw predictions shape: {predictions.shape}")
    
    # Convert to numpy and process
    keypoints = predictions.cpu().numpy()
    
    # Apply confidence threshold
    visible_keypoints = keypoints[..., 2] > confidence_threshold  # visibility scores
    
    results = {
        'keypoints': keypoints,
        'visible_keypoints': visible_keypoints,
        'num_detected_poses': len(keypoints),
        'confidence_threshold': confidence_threshold,
        'keypoint_names': COCO_KEYPOINT_NAMES[:keypoints.shape[1]] if keypoints.ndim > 1 else []
    }
    
    print(f"Detected {results['num_detected_poses']} pose(s)")
    print(f"Keypoints per pose: {keypoints.shape[1] if keypoints.ndim > 1 else 0}")
    
    return results

def visualize_poses(image: Image.Image, pose_results: Dict[str, Any], 
                   draw_skeleton: bool = True, keypoint_size: int = 3) -> Image.Image:
    """Visualize detected poses on the image."""
    # Create a copy for drawing
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    keypoints = pose_results['keypoints']
    visible_keypoints = pose_results['visible_keypoints']
    
    # Colors for different poses (if multiple people detected)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    # Draw each detected pose
    if keypoints.ndim == 3:  # Multiple poses: [num_poses, num_keypoints, 3]
        for pose_idx, (pose_kpts, pose_visible) in enumerate(zip(keypoints, visible_keypoints)):
            color = colors[pose_idx % len(colors)]
            draw_single_pose(draw, pose_kpts, pose_visible, color, draw_skeleton, keypoint_size)
    elif keypoints.ndim == 2:  # Single pose: [num_keypoints, 3]
        draw_single_pose(draw, keypoints, visible_keypoints, 'red', draw_skeleton, keypoint_size)
    
    return vis_image

def draw_single_pose(draw: ImageDraw.Draw, keypoints: np.ndarray, visible: np.ndarray, 
                    color: str, draw_skeleton: bool, keypoint_size: int):
    """Draw a single pose on the image."""
    # Draw skeleton connections first (so they appear behind keypoints)
    if draw_skeleton:
        for connection in COCO_SKELETON:
            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-based indexing
            
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and 
                visible[kpt1_idx] and visible[kpt2_idx]):
                
                x1, y1 = keypoints[kpt1_idx][:2]
                x2, y2 = keypoints[kpt2_idx][:2]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if visible[i]:
            # Draw keypoint as circle
            radius = keypoint_size
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='white')
            
            # Optionally draw keypoint index
            if keypoint_size >= 3:
                draw.text((x+radius+2, y-radius), str(i), fill='white')

def save_keypoints_json(pose_results: Dict[str, Any], output_path: str):
    """Save detected keypoints to JSON file."""
    # Prepare data for JSON serialization
    json_data = {
        'num_poses': pose_results['num_detected_poses'],
        'confidence_threshold': pose_results['confidence_threshold'],
        'keypoint_names': pose_results['keypoint_names'],
        'poses': []
    }
    
    keypoints = pose_results['keypoints']
    visible_keypoints = pose_results['visible_keypoints']
    
    if keypoints.ndim == 3:  # Multiple poses
        for pose_idx, (pose_kpts, pose_visible) in enumerate(zip(keypoints, visible_keypoints)):
            pose_data = {
                'pose_id': pose_idx,
                'keypoints': []
            }
            
            for kpt_idx, (x, y, conf) in enumerate(pose_kpts):
                kpt_data = {
                    'id': kpt_idx,
                    'name': pose_results['keypoint_names'][kpt_idx] if kpt_idx < len(pose_results['keypoint_names']) else f'keypoint_{kpt_idx}',
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf),
                    'visible': bool(pose_visible[kpt_idx])
                }
                pose_data['keypoints'].append(kpt_data)
            
            json_data['poses'].append(pose_data)
    
    elif keypoints.ndim == 2:  # Single pose
        pose_data = {
            'pose_id': 0,
            'keypoints': []
        }
        
        for kpt_idx, (x, y, conf) in enumerate(keypoints):
            kpt_data = {
                'id': kpt_idx,
                'name': pose_results['keypoint_names'][kpt_idx] if kpt_idx < len(pose_results['keypoint_names']) else f'keypoint_{kpt_idx}',
                'x': float(x),
                'y': float(y),
                'confidence': float(conf),
                'visible': bool(visible_keypoints[kpt_idx])
            }
            pose_data['keypoints'].append(kpt_data)
        
        json_data['poses'].append(pose_data)
    
    # Save to JSON file
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Keypoints saved to {output_path}")

def cleanup_resources():
    """Clean up GPU memory and resources."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

def main(args):
    """Main function to perform pose estimation with VitPose."""
    print("=" * 60)
    print("VitPose - Vision Transformer for Pose Estimation")
    print("=" * 60)
    print("Model selected:", args.model)
    print("Image URL:", args.image_url if not args.local_image else "N/A")
    print("Local image:", args.local_image if args.local_image else "N/A")
    print("Output image path:", args.output_image)
    print("Output keypoints path:", args.output_keypoints)
    print("Confidence threshold:", args.confidence_threshold)
    print("Draw skeleton:", args.draw_skeleton)
    print("Quantization:", args.use_quantization)
    if args.visualize:
        print("Visualization enabled")
    
    try:
        # Setup model
        processor, model = setup_model(args.model, args.use_quantization)
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
        
        # Detect poses
        try:
            pose_results = detect_poses(image, processor, model, args.confidence_threshold)
            
            if pose_results['num_detected_poses'] == 0:
                print("No poses detected with the given confidence threshold.")
                print(f"Try lowering --confidence_threshold (current: {args.confidence_threshold})")
                return
            
            # Visualize poses
            vis_image = visualize_poses(image, pose_results, args.draw_skeleton, args.keypoint_size)
            
            # Save visualization
            output_dir = os.path.dirname(args.output_image) or '.'
            os.makedirs(output_dir, exist_ok=True)
            vis_image.save(args.output_image)
            print(f"Pose visualization saved to {args.output_image}")
            
            # Save keypoints
            save_keypoints_json(pose_results, args.output_keypoints)
            
            # Display if requested
            if args.visualize:
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(vis_image)
                plt.title(f"Detected Poses ({pose_results['num_detected_poses']} poses)")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # Print summary
            print(f"\n✓ Pose estimation completed successfully!")
            print(f"✓ Detected {pose_results['num_detected_poses']} pose(s)")
            print(f"✓ Results saved to: {args.output_image}")
            print(f"✓ Keypoints saved to: {args.output_keypoints}")
            
        except Exception as e:
            print(f"Error during pose estimation: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Process completed!")
        print("=" * 60)
    
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
