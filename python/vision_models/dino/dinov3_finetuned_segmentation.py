# Fine-tuned DINOv3 Segmentation Head for Instance Segmentation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import argparse

class COCOInstanceDataset(Dataset):
    """Dataset for COCO instance segmentation."""
    
    def __init__(self, coco_root, ann_file, transforms=None, max_samples=None):
        self.coco_root = coco_root
        self.coco = COCO(ann_file)
        self.transforms = transforms
        
        # Get image IDs that have annotations
        self.img_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_path = os.path.join(self.coco_root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create instance masks
        masks = []
        for ann in anns:
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                masks.append(mask)
        
        if len(masks) == 0:
            # Create dummy mask if no annotations
            masks = [np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)]
        
        masks = np.stack(masks, axis=0)  # [num_instances, H, W]
        
        if self.transforms:
            image = self.transforms(image)
            # Resize masks to match image processing
            masks = torch.from_numpy(masks).float()
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(0), size=(224, 224), mode='nearest'
            ).squeeze(0)
        
        return image, masks

class DINOv3SegmentationHead(nn.Module):
    """Trainable segmentation head for DINOv3 features."""
    
    def __init__(self, in_channels=384, num_classes=91, feature_size=14):
        super().__init__()
        self.feature_size = feature_size
        
        # Segmentation decoder
        self.decoder = nn.Sequential(
            # Upsampling layers
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56x56 -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112x112 -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, patch_features):
        """
        Args:
            patch_features: [B, H, W, C] from DINOv3
        Returns:
            masks: [B, num_classes, 224, 224]
        """
        # Permute to [B, C, H, W]
        x = patch_features.permute(0, 3, 1, 2)
        masks = self.decoder(x)
        return masks

class DINOv3InstanceSegmentation:
    """Complete pipeline for DINOv3-based instance segmentation."""
    
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda"):
        self.device = device
        
        # Load DINOv3 backbone (frozen)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(device)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Segmentation head (trainable)
        hidden_size = self.backbone.config.hidden_size
        self.seg_head = DINOv3SegmentationHead(
            in_channels=hidden_size,
            num_classes=91  # COCO classes
        ).to(device)
        
        self.backbone.eval()
    
    def extract_features(self, images):
        """Extract patch features from DINOv3."""
        with torch.no_grad():
            outputs = self.backbone(pixel_values=images)
            last_hidden_states = outputs.last_hidden_state
            
            # Extract patch features (skip CLS and register tokens)
            num_register_tokens = getattr(self.backbone.config, 'num_register_tokens', 0)
            patch_features_flat = last_hidden_states[:, 1+num_register_tokens:, :]
            
            # Reshape to spatial dimensions
            batch_size = patch_features_flat.shape[0]
            feature_dim = patch_features_flat.shape[-1]
            spatial_size = int(np.sqrt(patch_features_flat.shape[1]))
            
            patch_features = patch_features_flat.view(
                batch_size, spatial_size, spatial_size, feature_dim
            )
            
            return patch_features
    
    def forward(self, images):
        """Forward pass through the complete pipeline."""
        patch_features = self.extract_features(images)
        masks = self.seg_head(patch_features)
        return masks
    
    def train_model(self, train_dataset, val_dataset=None, epochs=10, lr=1e-4):
        """Train the segmentation head."""
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        
        optimizer = optim.Adam(self.seg_head.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        self.seg_head.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, target_masks) in enumerate(train_loader):
                images = images.to(self.device)
                target_masks = target_masks.to(self.device)
                
                # Forward pass
                pred_masks = self.forward(images)
                
                # Compute loss (simplified - you may want to use focal loss, dice loss, etc.)
                loss = criterion(pred_masks, target_masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'dinov3_seg_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'seg_head_state_dict': self.seg_head.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.seg_head.load_state_dict(checkpoint['seg_head_state_dict'])
        print(f"Checkpoint loaded from {path}")
    
    def predict(self, image_path, threshold=0.5):
        """Predict instance masks for a single image."""
        self.seg_head.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        images = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            masks = self.forward(images)
        
        # Post-process masks
        masks = masks.squeeze(0).cpu().numpy()  # [num_classes, H, W]
        binary_masks = (masks > threshold).astype(np.uint8)
        
        return image, binary_masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--coco_root', type=str, help='Path to COCO dataset root')
    parser.add_argument('--ann_file', type=str, help='Path to COCO annotation file')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_samples', type=int, help='Limit training samples for testing')
    args = parser.parse_args()
    
    # Initialize model
    model = DINOv3InstanceSegmentation()
    
    if args.mode == 'train':
        if not args.coco_root or not args.ann_file:
            print("Error: --coco_root and --ann_file required for training")
            return
        
        # Prepare dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = COCOInstanceDataset(
            args.coco_root, args.ann_file, 
            transforms=transform, max_samples=args.max_samples
        )
        
        # Train model
        model.train_model(train_dataset, epochs=args.epochs)
        
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: --image_path required for prediction")
            return
        
        if args.checkpoint:
            model.load_checkpoint(args.checkpoint)
        
        # Predict
        image, masks = model.predict(args.image_path)
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show some predicted masks
        for i in range(min(2, masks.shape[0])):
            plt.subplot(1, 3, i + 2)
            plt.imshow(masks[i], cmap='hot')
            plt.title(f'Predicted Mask {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('dinov3_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
