# DINOv3 + SAM: Combining DINOv3 features with Segment Anything Model
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
import requests
from transformers import AutoModel, AutoImageProcessor, pipeline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

try:
    # Try to import SAM from transformers (newer versions)
    from transformers import SamModel, SamProcessor
    SAM_AVAILABLE = True
    SAM_SOURCE = "transformers"
except ImportError:
    try:
        # Fallback to segment-anything package
        from segment_anything import SamPredictor, sam_model_registry
        SAM_AVAILABLE = True
        SAM_SOURCE = "segment_anything"
    except ImportError:
        SAM_AVAILABLE = False
        SAM_SOURCE = None

class DINOv3SAMSegmentation:
    """Combine DINOv3 features with SAM for intelligent segmentation."""
    
    def __init__(self, 
                 dinov3_model="facebook/dinov3-vits16-pretrain-lvd1689m",
                 sam_model="facebook/sam-vit-base",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        print(f"Initializing DINOv3 + SAM pipeline on {device}")
        
        # Initialize DINOv3
        print("Loading DINOv3...")
        self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model)
        self.dinov3_model = AutoModel.from_pretrained(dinov3_model).to(device)
        self.dinov3_model.eval()
        
        # Initialize SAM
        self.sam_predictor = None
        if SAM_AVAILABLE:
            print(f"Loading SAM using {SAM_SOURCE}...")
            if SAM_SOURCE == "transformers":
                self.sam_processor = SamProcessor.from_pretrained(sam_model)
                self.sam_model = SamModel.from_pretrained(sam_model).to(device)
                self.sam_model.eval()
            elif SAM_SOURCE == "segment_anything":
                # For segment-anything package (requires manual installation)
                sam_checkpoint = "sam_vit_b_01ec64.pth"  # You need to download this
                model_type = "vit_b"
                if torch.cuda.is_available():
                    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    sam.to(device=device)
                    self.sam_predictor = SamPredictor(sam)
        else:
            print("⚠️ SAM not available. Install transformers>=4.32 or segment-anything package")
            print("Falling back to DINOv3-only clustering approach")
    
    def extract_dinov3_features(self, image):
        """Extract spatial features from DINOv3."""
        # Preprocess image
        inputs = self.dinov3_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.dinov3_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
            # Extract patch features (skip CLS and register tokens)
            num_register_tokens = getattr(self.dinov3_model.config, 'num_register_tokens', 0)
            patch_features_flat = last_hidden_states[:, 1+num_register_tokens:, :]
            
            # Reshape to spatial dimensions
            batch_size = patch_features_flat.shape[0]
            feature_dim = patch_features_flat.shape[-1]
            spatial_size = int(np.sqrt(patch_features_flat.shape[1]))
            
            patch_features = patch_features_flat.view(
                batch_size, spatial_size, spatial_size, feature_dim
            )
            
            return patch_features.squeeze(0)  # Remove batch dimension
    
    def generate_sam_prompts_from_dinov3(self, image, patch_features, method="clustering", num_clusters=5):
        """Generate SAM prompts based on DINOv3 features."""
        h, w, c = patch_features.shape
        
        if method == "clustering":
            # Use K-means clustering on DINOv3 features
            features_flat = patch_features.view(-1, c).cpu().numpy()
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_flat)
            cluster_map = cluster_labels.reshape(h, w)
            
            # Generate point prompts from cluster centers
            prompts = []
            for cluster_id in range(num_clusters):
                mask = (cluster_map == cluster_id)
                if np.any(mask):
                    # Find centroid of cluster
                    y_coords, x_coords = np.where(mask)
                    center_y = int(np.mean(y_coords) * image.size[1] / h)
                    center_x = int(np.mean(x_coords) * image.size[0] / w)
                    prompts.append([center_x, center_y])
            
            return prompts, cluster_map
        
        elif method == "attention":
            # Use attention-like mechanism to find interesting regions
            # Compute feature magnitude
            feature_magnitude = torch.norm(patch_features, dim=-1).cpu().numpy()
            
            # Find local maxima as prompt points
            from scipy import ndimage
            local_maxima = ndimage.maximum_filter(feature_magnitude, size=3) == feature_magnitude
            maxima_coords = np.where(local_maxima)
            
            # Convert to image coordinates and take top-k
            prompts = []
            magnitudes = feature_magnitude[maxima_coords]
            top_indices = np.argsort(magnitudes)[-num_clusters:]
            
            for idx in top_indices:
                y, x = maxima_coords[0][idx], maxima_coords[1][idx]
                center_y = int(y * image.size[1] / h)
                center_x = int(x * image.size[0] / w)
                prompts.append([center_x, center_y])
            
            return prompts, feature_magnitude
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def segment_with_sam(self, image, prompts):
        """Run SAM segmentation with the generated prompts."""
        if not SAM_AVAILABLE:
            print("SAM not available, skipping segmentation")
            return None
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        if SAM_SOURCE == "transformers":
            # Use Hugging Face transformers SAM
            masks = []
            for prompt in prompts:
                inputs = self.sam_processor(
                    image, 
                    input_points=[[prompt]], 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.sam_model(**inputs)
                
                # Get the best mask
                mask = self.sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]
                
                masks.append(mask[0, 0].numpy())  # Take first mask
                
        elif SAM_SOURCE == "segment_anything":
            # Use segment-anything package
            self.sam_predictor.set_image(image_np)
            
            masks = []
            for prompt in prompts:
                mask, scores, logits = self.sam_predictor.predict(
                    point_coords=np.array([prompt]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                
                # Take the best mask
                best_mask = mask[np.argmax(scores)]
                masks.append(best_mask)
        
        return masks
    
    def dinov3_clustering_fallback(self, image, patch_features, num_clusters=5):
        """Fallback method using only DINOv3 features for segmentation."""
        h, w, c = patch_features.shape
        
        # Flatten features and apply K-means
        features_flat = patch_features.view(-1, c).cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_flat)
        cluster_map = cluster_labels.reshape(h, w)
        
        # Resize cluster map to original image size
        cluster_map_resized = cv2.resize(
            cluster_map.astype(np.uint8), 
            image.size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert clusters to binary masks
        masks = []
        for cluster_id in range(num_clusters):
            mask = (cluster_map_resized == cluster_id)
            masks.append(mask)
        
        return masks, cluster_map
    
    def segment_image(self, image_path, prompt_method="clustering", num_segments=5):
        """Complete segmentation pipeline combining DINOv3 and SAM."""
        # Load image
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(response.raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        print(f"Processing image of size: {image.size}")
        
        # Extract DINOv3 features
        print("Extracting DINOv3 features...")
        patch_features = self.extract_dinov3_features(image)
        print(f"Feature shape: {patch_features.shape}")
        
        if SAM_AVAILABLE:
            # Generate prompts from DINOv3 features
            print(f"Generating SAM prompts using {prompt_method}...")
            prompts, feature_map = self.generate_sam_prompts_from_dinov3(
                image, patch_features, method=prompt_method, num_clusters=num_segments
            )
            print(f"Generated {len(prompts)} prompts")
            
            # Run SAM segmentation
            print("Running SAM segmentation...")
            masks = self.segment_with_sam(image, prompts)
            
            return {
                'image': image,
                'masks': masks,
                'prompts': prompts,
                'feature_map': feature_map,
                'patch_features': patch_features
            }
        else:
            # Fallback to DINOv3-only clustering
            print("Using DINOv3-only clustering fallback...")
            masks, cluster_map = self.dinov3_clustering_fallback(
                image, patch_features, num_clusters=num_segments
            )
            
            return {
                'image': image,
                'masks': masks,
                'prompts': [],
                'feature_map': cluster_map,
                'patch_features': patch_features
            }

def visualize_results(results, output_path="dinov3_sam_result.png"):
    """Visualize the segmentation results."""
    image = results['image']
    masks = results['masks']
    prompts = results['prompts']
    feature_map = results['feature_map']
    
    num_masks = len(masks)
    cols = min(4, num_masks + 2)
    fig, axes = plt.subplots(2, cols, figsize=(20, 10))
    
    if cols == 1:
        axes = axes.reshape(2, 1)
    elif len(axes.shape) == 1:
        axes = axes.reshape(1, -1)
    
    # Original image with prompts
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image + Prompts")
    axes[0, 0].axis('off')
    
    # Draw prompts if available
    if prompts:
        for i, prompt in enumerate(prompts):
            axes[0, 0].plot(prompt[0], prompt[1], 'ro', markersize=8)
            axes[0, 0].text(prompt[0], prompt[1]-20, f'P{i+1}', 
                           color='red', fontweight='bold', fontsize=12)
    
    # Feature map
    axes[0, 1].imshow(feature_map, cmap='viridis')
    axes[0, 1].set_title("DINOv3 Feature Map")
    axes[0, 1].axis('off')
    
    # Individual masks
    for i, mask in enumerate(masks[:cols-2]):
        col = i + 2
        if col < cols:
            axes[0, col].imshow(image)
            if isinstance(mask, np.ndarray):
                colored_mask = np.zeros((*mask.shape, 3))
                colored_mask[mask] = np.random.rand(3)
                axes[0, col].imshow(colored_mask, alpha=0.6)
            axes[0, col].set_title(f"Segment {i+1}")
            axes[0, col].axis('off')
    
    # Combined visualization
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("All Segments Combined")
    axes[1, 0].axis('off')
    
    # Overlay all masks with different colors
    for i, mask in enumerate(masks):
        if isinstance(mask, np.ndarray):
            colored_mask = np.zeros((*mask.shape, 3))
            color = plt.cm.tab10(i % 10)[:3]
            colored_mask[mask] = color
            axes[1, 0].imshow(colored_mask, alpha=0.3)
    
    # Individual masks in bottom row
    for i, mask in enumerate(masks[:cols-1]):
        col = i + 1
        if col < cols:
            axes[1, col].imshow(mask, cmap='gray')
            axes[1, col].set_title(f"Mask {i+1}")
            axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="DINOv3 + SAM Segmentation")
    parser.add_argument('--image', type=str, 
                        default="http://images.cocodataset.org/val2017/000000039769.jpg",
                        help='Image path or URL')
    parser.add_argument('--method', choices=['clustering', 'attention'], default='clustering',
                        help='Method for generating SAM prompts from DINOv3 features')
    parser.add_argument('--num_segments', type=int, default=5,
                        help='Number of segments to generate')
    parser.add_argument('--output', type=str, default="dinov3_sam_result.png",
                        help='Output visualization path')
    parser.add_argument('--dinov3_model', type=str, 
                        default="facebook/dinov3-vits16-pretrain-lvd1689m",
                        help='DINOv3 model to use')
    parser.add_argument('--sam_model', type=str, 
                        default="facebook/sam-vit-base",
                        help='SAM model to use')
    args = parser.parse_args()
    
    print("="*60)
    print("DINOv3 + SAM Intelligent Segmentation")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Method: {args.method}")
    print(f"Number of segments: {args.num_segments}")
    if not SAM_AVAILABLE:
        print("⚠️ SAM not available - using DINOv3 clustering fallback")
    print("="*60)
    
    # Initialize pipeline
    segmenter = DINOv3SAMSegmentation(
        dinov3_model=args.dinov3_model,
        sam_model=args.sam_model
    )
    
    # Run segmentation
    results = segmenter.segment_image(
        args.image, 
        prompt_method=args.method,
        num_segments=args.num_segments
    )
    
    # Visualize results
    visualize_results(results, args.output)
    
    print(f"\nSegmentation completed!")
    print(f"Generated {len(results['masks'])} segments")
    if results['prompts']:
        print(f"Used {len(results['prompts'])} SAM prompts")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
