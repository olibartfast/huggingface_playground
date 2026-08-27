# Hugging Face Experimental Playground

C++ and Python implementations for Hugging Face models, focusing on computer vision, multimodal tasks, and natural language processing.

## Key Features

- **Multi-Modal Models**: Qwen2.5-VL, CLIP, Grounding DINO, and OwlViT/OwlV2
- **Vision Models**: Depth estimation, DINOv3, ViTPose, RT-DETR, and SAM2
- **Pose Estimation**: ViTPose for human pose estimation
- **Video Classification**: C++ implementation using Triton Inference Server, plus a Python HMDB-51 benchmark for V-JEPA 2, VideoPrism and PE Video
- **Hugging Face Inference Provider Client**: C++ Client for Hugging Face Serverless API
- **Visualization Tools**: Built-in plotting and visualization
- **Modular Architecture**: Specialized requirements for different model families

## Repository Structure

### C++ examples
- **huggingface-inference-provider-cpp-client**: C++ client for Hugging Face Serverless Inference API. Supports object detection, image classification, image segmentation, and image-text-to-text generation. Uses `libcurl`, `OpenCV`, and `nlohmann/json`.
- **pose_estimation**: C++ application for pose estimation using ViTPose models exported to ONNX Runtime.
- **video_classification**: C++ application for video classification (e.g., VideoMAE) using Triton Inference Server, OpenCV, and C++20.

### Benchmarks
- **video_classification/python/hmdb51**: HMDB-51 frozen-probe benchmark for three recent video backbones. None of them ship an HMDB-51 head, so the encoder is frozen, clip embeddings are cached once, and a linear probe is fitted on the cached features. Includes a dataset fetcher, resumable extraction, and a smoke test that validates every backbone before a long run. See its [README](video_classification/python/hmdb51/README.md).

  Official split 1 (3570 train / 1530 test), linear probe, mean of 5 runs on an RTX 3060 Laptop:

  | Backbone | Dim | top-1 | top-5 | Latency/clip |
  |---|---|---|---|---|
  | PE Video (`facebook/pe-av-large-16-frame`) | 1792 | **77.23 ± 0.09** | 96.08 | 535 ms |
  | PE Video (`facebook/pe-av-small-16-frame`) | 768 | 76.04 ± 0.09 | 94.84 | ~535 ms |
  | VideoPrism base (`google/videoprism-base-f16r288`) | 768 | 63.18 ± 0.18 | 90.92 | **165 ms** |
  | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) | 1024 | 62.84 ± 0.29 | 90.07 | 970 ms |

  This ranks *mean-pooled linear probing*, not backbone capability — V-JEPA 2 is penalised by averaging ~8192 patch tokens, while its own classifier uses an attentive pooler. Note that PE's small/base/large labels size only the temporal fusion encoder; all three share one PE-Core Large ViT backbone, so a smaller checkpoint buys capacity, not speed.

### Python examples
- **multimodal_models**: Python scripts for multimodal tasks, including:
  - `grounding_dino.py`: Zero-shot object detection with text prompts.
  - `owl.py`: Zero-shot object detection using OwlViT or OwlV2 models.
  - `qwen2.5-vl.py`: Vision-language inference with single and multi-image support.
  - `qwen2.5-vl_examples.py`: Additional examples for Qwen2.5-VL models.
  - `qwen2.5-vl_object_detection.py`: Object detection using Qwen2.5-VL.
  - `object_detection_example.py`: Example for object detection with Qwen2.5-VL.
  - `qwen2.5-vl_object_detection_example.py`: Another example for object detection.
  - `clip.py`: Image-text similarity, zero-shot classification, and feature extraction.
- **vision_models**: Python scripts for vision-specific tasks:
  - `dinov3.py` & `dinov2.py`: State-of-the-art self-supervised vision transformers for features and classification.
  - `dinov3_finetuned_segmentation.py`: Finetuned DINOv3 for segmentation tasks.
  - `dinov3_sam_segmentation.py`: Integration of DINOv3 with SAM for segmentation.
  - `vitpose.py`: High-accuracy pose estimation.
  - `specialized_segmentation_models.py`: Instance segmentation using Mask R-CNN, DETR, and RT-DETR.
  - `rtdetrv2.py`: Object detection using RT-DETRv2 models.
  - `samv2.py`: Image segmentation using SAM2, integrated with RT-DETRv2.
  - `deep_anything_v2.py`: Monocular depth estimation using Depth Anything V2.
- **other_examples**: Additional Python scripts demonstrating various Hugging Face tasks:
  - `automatic_speech_recognition.py`: Audio processing using the LibriSpeech dataset.
  - `nlp_chatbot.py`: Conversational AI using Blenderbot.
  - `object_detection.py`: Basic object detection with DETR.
  - `sentence_embeddings.py`: Text embeddings using Sentence Transformers.
  - `translation.py`: Text translation with NLLB-200.
  - `zeroshot_audio_classification.py`: Zero-shot audio classification using CLAP.

### Notebooks
  - `dinov3_comprehensive_tutorial.ipynb`: A deep dive into DINOv3 features and applications.

## Getting Started

### Prerequisites
- **C++ Clients**:
  - CMake 3.20+ (3.25+ for video classification)
  - C++17/20 compatible compiler
  - libcurl, OpenCV, nlohmann/json
  - Triton Inference Server (for video classification)
- **Python Scripts**:
  - Python 3.12+
  - Install dependencies from `requirements.txt` in the respective directories
  - PyTorch (CPU or GPU)

### Setup Instructions
1. **C++ Inference Client**:
   - Navigate to `huggingface-inference-provider-cpp-client`.
   - Create a build directory and run:
     ```bash
     mkdir build && cd build
     cmake ..
     make
     ```
   - Set the `HF_TOKEN` environment variable:
     ```bash
     export HF_TOKEN=<your_huggingface_token>
     ```

2. **Video Classification (C++)**:
   - Navigate to `video_classification`.
   - Use CMake Presets:
     ```bash
     cmake --preset=release
     cmake --build --preset=release
     ```

3. **Pose Estimation (C++)**:
   - Navigate to `pose_estimation`.
   - Create a build directory and run:
     ```bash
     mkdir build && cd build
     cmake ..
     make
     ```

4. **Python Environment**:
   - Follow instructions in `other_examples/setup_huggingface_venv.md` to set up a virtual environment:
     ```bash
     chmod +x other_examples/setup_huggingface_venv.sh
     ./other_examples/setup_huggingface_venv.sh
     source huggingface_venv/bin/activate
     ```
   - Install additional dependencies for specific scripts (e.g., `sam2` for `vision_models/samv2.py`).

5. **Running Python Scripts**:
   - Example for `vision_models/rtdetrv2.py`:
     ```bash
     python vision_models/rtdetrv2.py --image_url <url> --output_dir output
     ```
   - Example for `multimodal_models/grounding_dino.py`:
     ```bash
     python multimodal_models/grounding_dino.py --model grounding-dino-base --text_labels "cat" "dog"
     ```
   - Example for `multimodal_models/qwen2.5-vl.py` (single image):
     ```bash
     python multimodal_models/qwen2.5-vl.py --model Qwen/Qwen2.5-VL-3B-Instruct --prompt "Describe this image in detail"
     ```
   - Example for `multimodal_models/qwen2.5-vl.py` (multi-image support):
     ```bash
     python multimodal_models/qwen2.5-vl.py --multi_image --image_urls "url1.jpg" "url2.jpg" --prompt "Compare these images"
     ```
   - Example for `multimodal_models/clip.py`:
     ```bash
     python multimodal_models/clip.py --task zero_shot --candidate_labels "cat" "dog" "bird"
     ```
   - Example for `vision_models/deep_anything_v2.py`:
     ```bash
     python vision_models/deep_anything_v2.py --model Depth-Anything-V2-Base-hf --visualize
     ```
   - Example for `vision_models/dinov2.py`:
     ```bash
     python vision_models/dinov2.py --task feature_extraction --visualize_features
     ```

## Model Capabilities

### Vision-Language Models
- **Qwen2.5-VL**: 3B, 7B, and 72B variants for multi-image conversations and visual Q&A
- **CLIP**: Zero-shot image classification and image-text similarity
- **Grounding DINO**: Text-prompted object detection
- **OwlViT/OwlV2**: Open-vocabulary object detection

### Vision Models  
- **Depth Anything V2**: Monocular depth estimation (Small to Giant variants)
- **DINOv2**: Self-supervised vision features and classification
- **RT-DETRv2**: Real-time object detection
- **SAM2**: Universal object segmentation

### Video Models
- **VideoMAE / ViViT / TimeSformer**: Kinetics-400 classification served through Triton (C++)
- **V-JEPA 2**: Self-supervised video encoder (no released HMDB-51 head; probed)
- **VideoPrism**: Factorised spatio-temporal encoder (probed)
- **PE Video**: Perception Encoder video tower, CLIP-style video/text (probed)

### Inference Methods
- **Pipeline API**: High-level interface
- **AutoModel**: Lower-level control with custom processing
- **Custom Classes**: Specialized implementations

## Learn
- [Hugging Face Learn](https://huggingface.co/learn): Tutorials and courses on machine learning.
- [Hugging Face Documentation](https://huggingface.co/docs): Guides for the Hub, Transformers, Diffusers, and more.
- [Serverless API](https://huggingface.co/docs/api-inference/index): Documentation for the Inference API used by the C++ client.
- [Vision Transformers](https://huggingface.co/docs/transformers/model_doc/vit): Understanding Vision Transformer architectures.
- [Multi-Modal Models](https://huggingface.co/docs/transformers/model_doc/clip): CLIP and other vision-language models.
- [Depth Estimation](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2): Monocular depth estimation techniques.
- [Self-Supervised Learning](https://huggingface.co/docs/transformers/model_doc/dinov2): DINOv2 and representation learning.
- [Video Classification](https://huggingface.co/docs/transformers/tasks/video_classification): Task guide for video models.
- [V-JEPA 2](https://huggingface.co/docs/transformers/model_doc/vjepa2), [VideoPrism](https://huggingface.co/docs/transformers/model_doc/videoprism), [PE Video](https://huggingface.co/docs/transformers/model_doc/pe_video): The three backbones covered by the HMDB-51 benchmark.
- [Deep Learning Containers](https://huggingface.co/docs/sagemaker/index#deep-learning-containers): Amazon SageMaker and Google Cloud integrations.
- [PyTorch Tutorials](https://github.com/philschmid/deep-learning-pytorch-huggingface): Deep learning with PyTorch and Hugging Face.
- [NVIDIA Triton Server](https://github.com/triton-inference-server/tutorials/tree/main/HuggingFace): Deploying models with Triton.
- [Transformers Notebooks](https://github.com/qubvel/transformers-notebooks/tree/main/notebooks): Example notebooks for Transformers.
- [Transformers Server](https://huggingface.co/docs/transformers/main/serving): Serve a transformer model. 

## Courses
- [DeepLearning.AI: Open Source Models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/)
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)
- [Hugging Face ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course/unit0/introduction)
- [Udemy: Transformers in Computer Vision](https://www.udemy.com/course/transformers-in-computer-vision-english-version)

