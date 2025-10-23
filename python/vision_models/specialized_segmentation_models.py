# Specialized Instance Segmentation Models: Mask R-CNN, DETR, RT-DETR
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import requests
from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection
import warnings
warnings.filterwarnings("ignore")

class MaskRCNNSegmentation:
    """Mask R-CNN for instance segmentation using torchvision."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Mask R-CNN on {device}...")
        
        # Load pre-trained Mask R-CNN
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        )
        self.model.to(device)
        self.model.eval()
        
        # COCO class names
        self.coco_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Predict instance masks using Mask R-CNN."""
        # Load image
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(response.raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        pred = predictions[0]
        
        # Filter by confidence
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        
        keep = scores >= confidence_threshold
        
        return {
            'image': image,
            'scores': scores[keep],
            'labels': labels[keep],
            'boxes': boxes[keep],
            'masks': masks[keep],
            'class_names': [self.coco_names[label] for label in labels[keep]]
        }

class DETRSegmentation:
    """DETR for instance segmentation using Hugging Face transformers."""
    
    def __init__(self, model_name="facebook/detr-resnet-50-panoptic"):
        print(f"Loading DETR model: {model_name}")
        self.pipe = pipeline(
            "image-segmentation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def predict(self, image_path):
        """Predict segmentation using DETR."""
        # Load image
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(response.raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Predict
        results = self.pipe(image)
        
        return {
            'image': image,
            'segments': results
        }

class RTDETRSegmentation:
    """RT-DETR for object detection (can be extended for segmentation)."""
    
    def __init__(self, model_name="PekingU/rtdetr_r50vd_coco_o365"):
        print(f"Loading RT-DETR model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.model.eval()
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Predict object detection using RT-DETR."""
        # Load image
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(response.raw).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        if torch.cuda.is_available():
            target_sizes = target_sizes.to("cuda")
        
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        return {
            'image': image,
            'scores': results['scores'].cpu().numpy(),
            'labels': results['labels'].cpu().numpy(),
            'boxes': results['boxes'].cpu().numpy()
        }

def visualize_maskrcnn_results(results, output_path="maskrcnn_result.png"):
    """Visualize Mask R-CNN results."""
    image = results['image']
    masks = results['masks']
    boxes = results['boxes']
    scores = results['scores']
    class_names = results['class_names']
    
    fig, axes = plt.subplots(1, min(4, len(masks) + 1), figsize=(20, 5))
    if len(masks) == 0:
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show masks
    for i, (mask, box, score, class_name) in enumerate(zip(masks[:3], boxes[:3], scores[:3], class_names[:3])):
        if i + 1 < len(axes):
            # Apply mask
            mask_binary = mask[0] > 0.5
            colored_mask = np.zeros((*mask_binary.shape, 3))
            colored_mask[mask_binary] = np.random.rand(3)
            
            axes[i + 1].imshow(image)
            axes[i + 1].imshow(colored_mask, alpha=0.5)
            axes[i + 1].set_title(f"{class_name}\nScore: {score:.2f}")
            axes[i + 1].axis('off')
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
            axes[i + 1].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Mask R-CNN results saved to {output_path}")
    plt.show()

def visualize_detr_results(results, output_path="detr_result.png"):
    """Visualize DETR segmentation results."""
    image = results['image']
    segments = results['segments']
    
    fig, axes = plt.subplots(1, min(4, len(segments) + 1), figsize=(20, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show segments
    for i, segment in enumerate(segments[:3]):
        if i + 1 < len(axes):
            axes[i + 1].imshow(segment['mask'])
            axes[i + 1].set_title(f"{segment['label']}\nScore: {segment['score']:.2f}")
            axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"DETR results saved to {output_path}")
    plt.show()

def visualize_rtdetr_results(results, output_path="rtdetr_result.png"):
    """Visualize RT-DETR detection results."""
    image = results['image']
    boxes = results['boxes']
    scores = results['scores']
    labels = results['labels']
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title("RT-DETR Object Detection")
    plt.axis('off')
    
    # Draw bounding boxes
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x1, y1-5, f"Class {label}: {score:.2f}", 
                color='red', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"RT-DETR results saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Specialized Instance Segmentation Models")
    parser.add_argument('--model', choices=['maskrcnn', 'detr', 'rtdetr'], required=True,
                        help='Choose segmentation model')
    parser.add_argument('--image', type=str, 
                        default="http://images.cocodataset.org/val2017/000000039769.jpg",
                        help='Image path or URL')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    args = parser.parse_args()
    
    print(f"Running {args.model.upper()} on image: {args.image}")
    
    if args.model == 'maskrcnn':
        model = MaskRCNNSegmentation()
        results = model.predict(args.image, confidence_threshold=args.confidence)
        output_path = args.output or "maskrcnn_result.png"
        visualize_maskrcnn_results(results, output_path)
        
        print(f"\nDetected {len(results['class_names'])} objects:")
        for i, (class_name, score) in enumerate(zip(results['class_names'], results['scores'])):
            print(f"  {i+1}. {class_name}: {score:.3f}")
    
    elif args.model == 'detr':
        model = DETRSegmentation()
        results = model.predict(args.image)
        output_path = args.output or "detr_result.png"
        visualize_detr_results(results, output_path)
        
        print(f"\nDetected {len(results['segments'])} segments:")
        for i, segment in enumerate(results['segments']):
            print(f"  {i+1}. {segment['label']}: {segment['score']:.3f}")
    
    elif args.model == 'rtdetr':
        model = RTDETRSegmentation()
        results = model.predict(args.image, confidence_threshold=args.confidence)
        output_path = args.output or "rtdetr_result.png"
        visualize_rtdetr_results(results, output_path)
        
        print(f"\nDetected {len(results['labels'])} objects:")
        for i, (label, score) in enumerate(zip(results['labels'], results['scores'])):
            print(f"  {i+1}. Class {label}: {score:.3f}")

if __name__ == "__main__":
    main()
