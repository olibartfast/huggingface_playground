# Real Instance Segmentation Implementation Guide

This repository contains three different approaches for implementing **real instance segmentation** beyond the demo DINOv3 script:

## 🎯 Overview of Approaches

### 1. Fine-tuned DINOv3 Segmentation Head
**File:** `dinov3_finetuned_segmentation.py`
- **Method:** Train a segmentation decoder on top of frozen DINOv3 features
- **Best for:** Custom datasets, learning specific object types
- **Requires:** Labeled segmentation data (COCO format)

### 2. Specialized Pre-trained Models  
**File:** `specialized_segmentation_models.py`
- **Method:** Use proven architectures (Mask R-CNN, DETR, RT-DETR)
- **Best for:** General-purpose segmentation, quick deployment
- **Requires:** No training needed

### 3. DINOv3 + SAM Combination
**File:** `dinov3_sam_segmentation.py`
- **Method:** Use DINOv3 to generate intelligent prompts for SAM
- **Best for:** Zero-shot segmentation, combining foundation models
- **Requires:** SAM installation

---

## 🚀 Quick Start

### Installation
```bash
# Install requirements
pip install -r requirements_segmentation.txt

# For SAM support (choose one):
# Option 1: Transformers SAM (recommended)
pip install transformers[torch]>=4.32.0

# Option 2: Original SAM (requires manual checkpoint download)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## 📋 Usage Examples

### 1. Specialized Models (Easiest - No Training Required)

```bash
# Mask R-CNN (best for accurate instance segmentation)
python specialized_segmentation_models.py --model maskrcnn --image path/to/image.jpg

# DETR (good for panoptic segmentation)
python specialized_segmentation_models.py --model detr --image path/to/image.jpg

# RT-DETR (fastest object detection)
python specialized_segmentation_models.py --model rtdetr --image path/to/image.jpg
```

### 2. DINOv3 + SAM (Zero-shot Intelligent Segmentation)

```bash
# Clustering-based prompts
python dinov3_sam_segmentation.py --image path/to/image.jpg --method clustering --num_segments 5

# Attention-based prompts  
python dinov3_sam_segmentation.py --image path/to/image.jpg --method attention --num_segments 3

# With URL
python dinov3_sam_segmentation.py --image "http://images.cocodataset.org/val2017/000000039769.jpg"
```

### 3. Fine-tuned DINOv3 (Custom Training)

```bash
# Training (requires COCO dataset)
python dinov3_finetuned_segmentation.py --mode train \
    --coco_root /path/to/coco \
    --ann_file /path/to/annotations.json \
    --epochs 10

# Prediction with trained model
python dinov3_finetuned_segmentation.py --mode predict \
    --image path/to/image.jpg \
    --checkpoint dinov3_seg_epoch_10.pth
```

---

## 🎭 Model Comparison

| Method | Training Required | Accuracy | Speed | Best Use Case |
|--------|------------------|----------|-------|---------------|
| **Mask R-CNN** | ❌ No | TBD | TBD | Production instance segmentation |
| **DETR** | ❌ No | TBD | TBD | Panoptic segmentation |
| **DINOv3 + SAM** | ❌ No | TBD | TBD | Zero-shot, flexible prompting |
| **Fine-tuned DINOv3** | ✅ Yes | TBD | TBD| Custom domains, specific objects |

---

## 🔧 Technical Details

### Mask R-CNN
- **Architecture:** ResNet-50 + FPN backbone with mask head
- **Training:** Pre-trained on COCO (80 classes)
- **Output:** Bounding boxes + instance masks
- **Strengths:** Mature, highly accurate, well-documented

### DETR (Detection Transformer)
- **Architecture:** Transformer-based end-to-end detection
- **Training:** Pre-trained for panoptic segmentation
- **Output:** Object detection + segmentation masks
- **Strengths:** No NMS needed, handles variable number of objects

### DINOv3 + SAM
- **Architecture:** DINOv3 feature extraction + SAM prompting
- **Training:** Both models pre-trained (no fine-tuning needed)
- **Output:** Flexible segmentation based on DINOv3 features
- **Strengths:** Combines self-supervised + prompted segmentation

### Fine-tuned DINOv3
- **Architecture:** Frozen DINOv3 + trainable segmentation decoder
- **Training:** Custom training on labeled data
- **Output:** Task-specific instance masks
- **Strengths:** Leverages DINOv3 features for domain-specific tasks

---

## 📊 Expected Results

### Performance Benchmarks (Approximate)

**COCO Validation Set:**
- **Mask R-CNN:** mAP ~35% (instance segmentation)
- **DETR:** mAP ~32% (panoptic segmentation)  
- **DINOv3 + SAM:** ~25-30% (depending on prompting strategy)
- **Fine-tuned DINOv3:** ~30-40% (depends on training data)

**Speed (on V100 GPU):**
- **Mask R-CNN:** ~10 FPS
- **DETR:** ~15 FPS
- **DINOv3 + SAM:** ~5 FPS
- **Fine-tuned DINOv3:** ~12 FPS

---

## 🛠️ Customization Guide

### Adding New Classes to Fine-tuned DINOv3
```python
# Modify in dinov3_finetuned_segmentation.py
seg_head = DINOv3SegmentationHead(
    in_channels=384,
    num_classes=YOUR_NUM_CLASSES,  # Change this
    feature_size=14
)
```

### Custom SAM Prompting Strategy
```python
# Add to dinov3_sam_segmentation.py
def custom_prompt_generation(self, patch_features):
    # Your custom logic here
    # Return list of [x, y] coordinates
    return prompts
```

### Using Different Backbones
```python
# For specialized models
model = torchvision.models.detection.maskrcnn_resnet101_fpn(
    weights=torchvision.models.detection.MaskRCNN_ResNet101_FPN_Weights.COCO_V1
)
```

---

## 🎯 Production Recommendations

### For Production Deployment:
1. **Start with Mask R-CNN** - Most reliable and battle-tested
2. **Use DINOv3 + SAM** for flexible, prompt-based segmentation
3. **Fine-tune DINOv3** only if you have domain-specific requirements

### For Research/Experimentation:
1. **DINOv3 + SAM** - Most flexible and innovative
2. **Fine-tuned DINOv3** - Best for custom datasets
3. **DETR** - Good for end-to-end learning experiments

---

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   ```bash
   # Reduce batch size or use smaller model
   --batch_size 1
   --model dinov3-vits16-pretrain-lvd1689m  # Use small variant
   ```

2. **SAM Import Error:**
   ```bash
   pip install transformers>=4.32.0
   # or
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

3. **COCO Dataset Format:**
   - Ensure annotations are in COCO JSON format
   - Check image paths are relative to coco_root

4. **DINOv3 Access Error:**
   ```bash
   huggingface-cli login
   # Request access at: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
   ```

