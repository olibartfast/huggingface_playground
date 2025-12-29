# https://huggingface.co/docs/transformers/model_doc/dinov2
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

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "feature_extraction",
            "depth_estimation",
            "classification",
            "instance_segmentation",
            "video_classification",
            "sparse_matching",
            "dense_matching",
            "instance_retrieval",
        ],
        default="feature_extraction",
        help="Task to perform with DINOv2"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
            "dinov2-small-imagenet1k-1-layer",
            "dinov2-base-imagenet1k-1-layer",
        ],
        default="dinov2-small",
        help="DINOv2 model to use (e.g., dinov2-small, dinov2-small-imagenet1k-1-layer for classification)"
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
        default="output.png",
        help="File path to save the output image"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (display output image or feature map)"
    )
    args = parser.parse_args()
    return args

def setup_model(model_name, task):
    """Sets up the processor and model based on the model name and task."""
    model_id = f"facebook/{model_name}"
    print(f"Loading {model_name} model for {task}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    print(f"Processor type: {type(processor)}")
    if task == "classification" and "imagenet1k" in model_name:
        model = AutoModelForImageClassification.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to CUDA")
    print(f"{model_name} model loaded successfully!")
    return processor, model

class DepthHead(nn.Module):
    """Custom depth estimation head for DINOv2."""
    def __init__(self, in_channels=384):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class ClassificationHead(nn.Module):
    """Custom classification head for DINOv2."""
    def __init__(self, in_channels=384, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class InstanceSegmentationHead(nn.Module):
    """Simplified instance segmentation head for DINOv2."""
    def __init__(self, in_channels=384, num_instances=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_instances, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))

def feature_extraction(image, processor, model, output_path, visualize=False):
    """Perform feature extraction with DINOv2 (per HF docs)."""
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    print("Extracting features...")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # [batch, 1 + num_patches, hidden_size]

    cls_token = last_hidden_states[:, 0, :]  # [batch, hidden_size]
    patch_features = last_hidden_states[:, 1:, :]  # [batch, num_patches, hidden_size]
    print("CLS token shape:", cls_token.shape)
    print("Patch features shape:", patch_features.shape)

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    embedding_path = os.path.splitext(output_path)[0] + "_embedding.txt"
    np.savetxt(embedding_path, cls_token.cpu().numpy())
    print(f"CLS embedding saved to {embedding_path}")

    if visualize or output_path:
        batch_size, num_patches, channels = patch_features.shape
        patch_size = int(num_patches ** 0.5)
        feature_map = patch_features.mean(dim=2).view(batch_size, patch_size, patch_size).cpu().numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8) * 255
        feature_image = Image.fromarray(feature_map[0].astype("uint8"))
        feature_image.save(output_path)
        print(f"Feature map saved to {output_path}")

        if visualize:
            plt.figure()
            plt.imshow(feature_image, cmap='viridis')
            plt.axis('off')
            plt.title("DINOv2 Feature Map (Mean Across Channels)")
            plt.show()
            plt.close()

    return cls_token, patch_features

def depth_estimation(image, processor, model, output_path, visualize=False):
    """Perform depth estimation with DINOv2 (requires fine-tuning)."""
    print("Note: Depth estimation with raw DINOv2 requires fine-tuning for accurate results.")
    in_channels = model.config.hidden_size
    depth_head = DepthHead(in_channels=in_channels)
    if torch.cuda.is_available():
        depth_head = depth_head.to("cuda")

    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    print("Extracting features for depth estimation...")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 1:, :]
        batch_size, num_patches, channels = features.shape
        patch_size = int(num_patches ** 0.5)
        features = features.permute(0, 2, 1).reshape(batch_size, channels, patch_size, patch_size)
        depth = depth_head(features)

    depth = torch.nn.functional.interpolate(depth, size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False)
    depth = depth.squeeze(1)

    depth_np = depth.detach().cpu().numpy().squeeze()
    depth_map = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-5) * 255
    depth_image = Image.fromarray(depth_map.astype(np.uint8))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    depth_image.save(output_path)
    print(f"Depth map saved to {output_path}")

    if visualize:
        plt.figure()
        plt.imshow(depth_image, cmap='viridis')
        plt.axis('off')
        plt.title("DINOv2 Depth Map (Illustrative)")
        plt.show()
        plt.close()

    return depth

def classification(image, processor, model, output_path, visualize=False):
    """Perform image classification with DINOv2."""
    if isinstance(model, AutoModelForImageClassification):
        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        print("Performing classification...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_class = np.argmax(probs)
            predicted_label = model.config.id2label[pred_class]
    else:
        print("Note: Classification with raw DINOv2 requires fine-tuning. Using demo classifier.")
        in_channels = model.config.hidden_size
        class_head = ClassificationHead(in_channels=in_channels, num_classes=2)
        if torch.cuda.is_available():
            class_head = class_head.to("cuda")

        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        print("Performing classification...")
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            probs = class_head(cls_embedding)[0].cpu().numpy()
            pred_class = np.argmax(probs)
            class_names = ["non-cat", "cat"]  # Demo classes
            predicted_label = class_names[pred_class]

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.splitext(output_path)[0] + "_classification.txt"
    with open(result_path, "w") as f:
        f.write(f"Predicted class: {predicted_label}, Confidence: {probs[pred_class]:.3f}\n")
    print(f"Classification result saved to {result_path}")

    if visualize or output_path:
        plt.figure()
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_label} ({probs[pred_class]:.3f})")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Classification visualization saved to {output_path}")

        if visualize:
            plt.show()
        plt.close()

    return predicted_label, probs

def instance_segmentation(image, processor, model, output_path, visualize=False):
    """Perform instance segmentation with DINOv2 (requires fine-tuning)."""
    print("Note: Instance segmentation with raw DINOv2 requires fine-tuning for accurate results.")
    in_channels = model.config.hidden_size
    seg_head = InstanceSegmentationHead(in_channels=in_channels, num_instances=5)
    if torch.cuda.is_available():
        seg_head = seg_head.to("cuda")

    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    print("Performing instance segmentation...")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 1:, :]
        batch_size, num_patches, channels = features.shape
        patch_size = int(num_patches ** 0.5)
        features = features.permute(0, 2, 1).reshape(batch_size, channels, patch_size, patch_size)
        masks = seg_head(features)

    masks = torch.nn.functional.interpolate(masks, size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False)
    masks = masks[0].cpu().numpy()  # [num_instances, height, width]

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(image)
    for i in range(masks.shape[0]):
        mask = masks[i] > 0.5
        plt.contour(mask, colors=[np.random.rand(3)], levels=[0.5], alpha=0.5)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Segmentation visualization saved to {output_path}")

    if visualize:
        plt.show()
    plt.close()

    return masks

def video_classification(video_path, processor, model, output_path, visualize=False):
    """Perform video classification with DINOv2 (requires fine-tuning)."""
    print("Note: Video classification with raw DINOv2 requires fine-tuning for accurate results.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    in_channels = model.config.hidden_size
    class_head = ClassificationHead(in_channels=in_channels, num_classes=2)
    if torch.cuda.is_available():
        class_head = class_head.to("cuda")

    embeddings = []
    first_frame = None
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame is None:
            first_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding)

    cap.release()
    if not embeddings:
        raise ValueError("No frames processed from video")

    embeddings = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)  # Average embeddings
    probs = class_head(embeddings)[0].cpu().numpy()
    pred_class = np.argmax(probs)
    class_names = ["non-action", "action"]  # Demo classes
    predicted_label = class_names[pred_class]

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.splitext(output_path)[0] + "_video_classification.txt"
    with open(result_path, "w") as f:
        f.write(f"Predicted class: {predicted_label}, Confidence: {probs[pred_class]:.3f}\n")
    print(f"Video classification result saved to {result_path}")

    if visualize or output_path:
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(first_frame_rgb)
        plt.title(f"Predicted: {predicted_label} ({probs[pred_class]:.3f})")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Video classification visualization saved to {output_path}")

        if visualize:
            plt.show()
        plt.close()

    return predicted_label, probs

def sparse_matching(image1, image2, processor, model, output_path, visualize=False):
    """Perform sparse matching between two images with DINOv2."""
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")
    if torch.cuda.is_available():
        inputs1 = inputs1.to("cuda")
        inputs2 = inputs2.to("cuda")

    print("Extracting features for sparse matching...")
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        features1 = outputs1.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_size]
        features2 = outputs2.last_hidden_state[:, 1:, :]

    features1 = features1[0].cpu().numpy()  # [num_patches, hidden_size]
    features2 = features2[0].cpu().numpy()
    num_patches = features1.shape[0]
    patch_size = int(num_patches ** 0.5)

    # Select top-k keypoints based on feature magnitude
    k = 50
    magnitudes1 = np.linalg.norm(features1, axis=1)
    magnitudes2 = np.linalg.norm(features2, axis=1)
    keypoints1 = np.argsort(magnitudes1)[-k:]
    keypoints2 = np.argsort(magnitudes2)[-k:]

    # Compute distances between keypoints
    dists = cdist(features1[keypoints1], features2[keypoints2], metric='cosine')
    matches = np.argmin(dists, axis=1)
    keypoints2 = keypoints2[matches]

    # Convert patch indices to image coordinates
    coords1 = np.array([(i // patch_size * 14, i % patch_size * 14) for i in keypoints1])  # 14x14 patches
    coords2 = np.array([(i // patch_size * 14, i % patch_size * 14) for i in keypoints2])

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.scatter(coords1[:, 1], coords1[:, 0], c='r', s=10)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.scatter(coords2[:, 1], coords2[:, 0], c='b', s=10)
    for i in range(len(coords1)):
        plt.plot([coords1[i, 1] + image1.size[0], coords2[i, 1]], [coords1[i, 0], coords2[i, 0]], 'g-')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Sparse matching visualization saved to {output_path}")

    if visualize:
        plt.show()
    plt.close()

    return coords1, coords2

def dense_matching(image1, image2, processor, model, output_path, visualize=False):
    """Perform dense matching between two images with DINOv2."""
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")
    if torch.cuda.is_available():
        inputs1 = inputs1.to("cuda")
        inputs2 = inputs2.to("cuda")

    print("Extracting features for dense matching...")
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        features1 = outputs1.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_size]
        features2 = outputs2.last_hidden_state[:, 1:, :]

    features1 = features1[0]  # [num_patches, hidden_size]
    features2 = features2[0]
    num_patches = features1.shape[0]
    patch_size = int(num_patches ** 0.5)

    # Compute cosine similarity
    features1_norm = features1 / torch.norm(features1, dim=1, keepdim=True)
    features2_norm = features2 / torch.norm(features2, dim=1, keepdim=True)
    similarity = torch.matmul(features1_norm, features2_norm.T)  # [num_patches, num_patches]
    matches = similarity.argmax(dim=1).cpu().numpy()  # Best match for each patch in image1

    # Visualize correspondence for a sample point
    sample_idx = num_patches // 2  # Middle patch
    match_idx = matches[sample_idx]
    coord1 = (sample_idx // patch_size * 14, sample_idx % patch_size * 14)
    coord2 = (match_idx // patch_size * 14, match_idx % patch_size * 14)

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.scatter([coord1[1]], [coord1[0]], c='r', s=50)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.scatter([coord2[1]], [coord2[0]], c='b', s=50)
    plt.plot([coord1[1] + image1.size[0], coord2[1]], [coord1[0], coord2[0]], 'g-')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Dense matching visualization saved to {output_path}")

    if visualize:
        plt.show()
    plt.close()

    return matches

def instance_retrieval(image1, image2, processor, model, output_path, visualize=False):
    """Perform instance retrieval with DINOv2."""
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")
    if torch.cuda.is_available():
        inputs1 = inputs1.to("cuda")
        inputs2 = inputs2.to("cuda")

    print("Extracting features for instance retrieval...")
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        cls1 = outputs1.last_hidden_state[:, 0, :]  # [1, hidden_size]
        cls2 = outputs2.last_hidden_state[:, 0, :]

    cos_sim = torch.nn.functional.cosine_similarity(cls1, cls2, dim=1).item()
    print(f"Cosine similarity: {cos_sim:.3f}")

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.splitext(output_path)[0] + "_retrieval.txt"
    with open(result_path, "w") as f:
        f.write(f"Cosine similarity between images: {cos_sim:.3f}\n")
    print(f"Retrieval result saved to {result_path}")

    if visualize or output_path:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image1)
        plt.title("Query Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image2)
        plt.title(f"Reference Image\nSimilarity: {cos_sim:.3f}")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Retrieval visualization saved to {output_path}")

        if visualize:
            plt.show()
        plt.close()

    return cos_sim

def main(args):
    """Main function to perform the specified task with DINOv2."""
    print("Task selected:", args.task)
    print("Model selected:", args.model)
    print("Image URL:", args.image_url)
    print("Second Image URL:", args.second_image_url)
    print("Video Path:", args.video_path)
    print("Output image path:", args.output_image)
    if args.visualize:
        print("Visualization enabled (output will be displayed)")

    try:
        processor, model = setup_model(args.model, args.task)
        if processor is None or model is None:
            raise Exception("Model setup failed.")

        if args.task != "video_classification":
            print("Downloading and opening first image...")
            image1 = Image.open(requests.get(args.image_url, stream=True).raw)
            print("First image loaded successfully!")
            image2 = None
            if args.task in ["sparse_matching", "dense_matching", "instance_retrieval"]:
                if args.second_image_url is None:
                    raise ValueError("Second image URL required for this task")
                print("Downloading and opening second image...")
                image2 = Image.open(requests.get(args.second_image_url, stream=True).raw)
                print("Second image loaded successfully!")

        if args.task == "feature_extraction":
            cls_token, patch_features = feature_extraction(
                image1, processor, model, args.output_image, args.visualize
            )
        elif args.task == "depth_estimation":
            depth = depth_estimation(
                image1, processor, model, args.output_image, args.visualize
            )
        elif args.task == "classification":
            predicted_label, probs = classification(
                image1, processor, model, args.output_image, args.visualize
            )
        elif args.task == "instance_segmentation":
            masks = instance_segmentation(
                image1, processor, model, args.output_image, args.visualize
            )
        elif args.task == "video_classification":
            if args.video_path is None:
                raise ValueError("Video path required for video_classification")
            predicted_label, probs = video_classification(
                args.video_path, processor, model, args.output_image, args.visualize
            )
        elif args.task == "sparse_matching":
            coords1, coords2 = sparse_matching(
                image1, image2, processor, model, args.output_image, args.visualize
            )
        elif args.task == "dense_matching":
            matches = dense_matching(
                image1, image2, processor, model, args.output_image, args.visualize
            )
        elif args.task == "instance_retrieval":
            cos_sim = instance_retrieval(
                image1, image2, processor, model, args.output_image, args.visualize
            )

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)