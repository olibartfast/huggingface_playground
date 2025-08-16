# DINOv3 Vision Foundation Model Script

This script provides a robust interface for working with DINOv3 vision foundation models from Meta/Facebook.

## Features

- **Feature Extraction**: Extract CLS tokens, patch features, and register tokens
- **Instance Segmentation**: Demo segmentation with untrained head (educational purposes)
- **Quantization Support**: Reduce memory usage with Int4 quantization
- **Local/Remote Images**: Support for both URLs and local image files
- **Error Handling**: Comprehensive error handling and validation
- **Resource Management**: Automatic cleanup of GPU memory

## Installation

```bash
pip install -r requirements.dinov3.txt
```

## Usage

### Basic Feature Extraction
```bash
python3 dinov3.py --task feature_extraction --visualize
```

### With Local Image
```bash
python3 dinov3.py --task feature_extraction --local_image /path/to/image.jpg --visualize
```

### Instance Segmentation Demo
```bash
python3 dinov3.py --task instance_segmentation --visualize
```

### With Quantization (for large models)
```bash
python3 dinov3.py --task feature_extraction --model dinov3-vitl16-pretrain-lvd1689m --use_quantization
```

### Debug Mode
```bash
python3 dinov3.py --task feature_extraction --debug
```

## Available Models

- `dinov3-vits16-pretrain-lvd1689m` (Small, fastest)
- `dinov3-vitb16-pretrain-lvd1689m` (Base, good balance)
- `dinov3-vitl16-pretrain-lvd1689m` (Large, better quality)
- `dinov3-vitg16-pretrain-lvd1689m` (Giant, best quality)
- `dinov3-vit7b16-pretrain-lvd1689m` (7B parameters, use quantization)
- `dinov3-vitsplus-pretrain-lvd1689m`
- `dinov3-convnext-base-pretrain-lvd1689m`
- `dinov3-convnext-large-pretrain-lvd1689m`

## Key Improvements

1. **Better Error Handling**: Graceful handling of CUDA OOM, missing models, etc.
2. **Device Management**: Proper device detection and tensor movement
3. **Resource Cleanup**: Automatic GPU memory cleanup
4. **Input Validation**: Model existence validation and input checking
5. **Interactive Warnings**: Clear warnings about untrained components
6. **Local File Support**: Can process local images in addition to URLs
7. **Debug Mode**: Detailed error messages when needed

## Important Notes

- **Instance Segmentation**: The segmentation head is UNTRAINED and produces random results. This is for educational purposes only.
- **Memory Requirements**: Large models require significant GPU memory. Use quantization for memory-constrained systems.
- **Register Tokens**: DINOv3's key innovation - dedicated memory slots for global information.

## Output Files

- Feature extraction saves:
  - `*_cls_embedding.txt`: CLS token embeddings
  - `*_register_tokens.txt`: Register token embeddings (if available)
  - Visualization PNG file

## Hardware Requirements

- **Small models (vits16)**: 2-4GB GPU memory
- **Base models (vitb16)**: 4-8GB GPU memory  
- **Large models (vitl16)**: 8-16GB GPU memory
- **Giant models (vitg16)**: 16-32GB GPU memory
- **7B model**: 32GB+ GPU memory (use quantization)

## Troubleshooting

1. **CUDA OOM**: Use `--use_quantization` flag
2. **Model not found**: Check model name spelling
3. **Import errors**: Install all requirements
4. **Slow performance**: Ensure CUDA is available and properly installed
