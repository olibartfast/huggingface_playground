# Hugging Face Experimental Playground

C++ and Python implementations for Hugging Face models, focusing on computer vision, multimodal tasks, and natural language processing.

## Key Features

- **Multi-Modal Models**: Qwen2.5-VL, CLIP, Grounding DINO, and OwlViT/OwlV2
- **Vision Models**: Depth estimation, self-supervised learning, object detection, and segmentation
- **Multiple Inference Methods**: Pipeline, AutoModel, and custom implementations
- **C++ Client**: High-performance client for Hugging Face Serverless API
- **Visualization Tools**: Built-in plotting and visualization
- **Modular Architecture**: Specialized requirements for different model families

## Repository Structure

- **huggingface-cpp-inference-client**: C++ client for Hugging Face Serverless Inference API. Supports object detection, image classification, image segmentation, and image-text-to-text generation. Uses `libcurl`, `OpenCV`, and `nlohmann/json`.
- **multimodal_models**: Python scripts for multimodal tasks, including:
  - `grounding_dino.py`: Zero-shot object detection with text prompts.
  - `owl.py`: Zero-shot object detection using OwlViT or OwlV2 models.
  - `qwen2.5-vl.py`: Vision-language inference with single and multi-image support.
  - `qwen2.5-vl_examples.py`: Additional examples for Qwen2.5-VL models.
  - `clip.py`: Image-text similarity, zero-shot classification, and feature extraction.
- **vision_models**: Python scripts for vision-specific tasks:
  - `rtdetrv2.py`: Object detection using RT-DETRv2 models.
  - `samv2.py`: Image segmentation using SAM2, integrated with RT-DETRv2.
  - `deep_anything_v2.py`: Monocular depth estimation using Depth Anything V2.
  - `dinov2.py`: Self-supervised vision transformer for feature extraction and classification.
- **other_examples**: Additional Python scripts demonstrating various Hugging Face tasks:
  - `automatic_speech_recognition.py`: Audio processing using the LibriSpeech dataset.
  - `nlp_chatbot.py`: Conversational AI using Blenderbot.
  - `object_detection.py`: Basic object detection with DETR.
  - `sentence_embeddings.py`: Text embeddings using Sentence Transformers.
  - `translation.py`: Text translation with NLLB-200.
  - `zeroshot_audio_classification.py`: Zero-shot audio classification using CLAP.
  - `setup_huggingface_venv.sh` and `setup_huggingface_venv.md`: Scripts and instructions for setting up a Python virtual environment for Hugging Face libraries.
  - Requirements files (`requirements.txt`, `requirements.sentence_embeddings.txt`, `requirements.zeroshot_audio_classification.txt`) for dependency management.
  - Additional specialized requirements files in multimodal_models directory (`requirements.clip.txt`, `requirements.qwen2.5-vl.txt`) for specific model dependencies.

## Getting Started

### Prerequisites
- **C++ Client**:
  - CMake 3.20+
  - libcurl, OpenCV, nlohmann/json
  - cpp-base64 and cxxopts (fetched via CMake)
  - Hugging Face API token (`HF_TOKEN`) set as an environment variable
- **Python Scripts**:
  - Python 3.12+
  - Install dependencies from `requirements.txt` in the respective directories
  - PyTorch (CPU or GPU, depending on hardware and model requirements)
  - Additional libraries like `sam2`, `sentence-transformers`, `soundfile`, `librosa`, `pydub`, and `pyaudio` for specific scripts
  - For vision-language models: `transformers>=4.37.0`, `qwen-vl-utils` for Qwen2.5-VL
  - For depth estimation: `transformers[vision]` with appropriate torch versions

### Setup Instructions
1. **C++ Client**:
   - Navigate to `huggingface-cpp-inference-client`.
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
   - Run the client with:
     ```bash
     ./huggingface_app --input <image_path> --task <task_type> --model <model_url>
     ```
     Example:
     ```bash
     ./huggingface_app --input image.jpg --task object-detection --model https://api-inference.huggingface.co/models/facebook/detr-resnet-50
     ```

2. **Python Environment**:
   - Follow instructions in `other_examples/setup_huggingface_venv.md` to set up a virtual environment:
     ```bash
     chmod +x other_examples/setup_huggingface_venv.sh
     ./other_examples/setup_huggingface_venv.sh
     source huggingface_venv/bin/activate
     ```
   - Install additional dependencies for specific scripts (e.g., `sam2` for `vision_models/samv2.py`).

3. **Running Python Scripts**:
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

