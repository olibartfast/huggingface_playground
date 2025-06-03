# Hugging Face Experimental Playground

This repository serves as a playground for experimenting with models and tasks from [Hugging Face](https://huggingface.co/tasks) and related resources. It includes a mix of C++ and Python implementations for exploring machine learning models, particularly focusing on computer vision, multimodal tasks, and natural language processing.

## Repository Structure

- **huggingface-cpp-inference-client**: A C++ client for interacting with Hugging Face's Serverless Inference API. It supports tasks like object detection, image classification, image segmentation, and image-text-to-text generation using models such as DETR, ViT, SegFormer, and Llama-3.2. The client leverages `libcurl`, `OpenCV`, and `nlohmann/json` for HTTP requests, image processing, and JSON handling, respectively.
- **multimodal_models**: Python scripts for multimodal tasks, including:
  - `grounding_dino.py`: Zero-shot object detection using Grounding DINO models (`grounding-dino-tiny` or `grounding-dino-base`) with text prompts.
  - `owl.py`: Zero-shot object detection using OwlViT or OwlV2 models, supporting flexible text-based object detection.
- **vision_models**: Python scripts for vision-specific tasks:
  - `rtdetrv2.py`: Object detection using RT-DETRv2 models, with visualization of bounding boxes.
  - `samv2.py`: Image segmentation using SAM2 (Segment Anything Model 2), integrated with RT-DETRv2 for bounding box prompts.
- **other_examples**: Additional Python scripts demonstrating various Hugging Face tasks:
  - `automatic_speech_recognition.py`: Audio processing using the LibriSpeech dataset.
  - `nlp_chatbot.py`: Conversational AI using Blenderbot.
  - `object_detection.py`: Basic object detection with DETR.
  - `sentence_embeddings.py`: Text embeddings using Sentence Transformers.
  - `translation.py`: Text translation with NLLB-200.
  - `zeroshot_audio_classification.py`: Zero-shot audio classification using CLAP.
  - `setup_huggingface_venv.sh` and `setup_huggingface_venv.md`: Scripts and instructions for setting up a Python virtual environment for Hugging Face libraries.
  - Requirements files (`requirements.txt`, `requirements.sentence_embeddings.txt`, `requirements.zeroshot_audio_classification.txt`) for dependency management.

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
  - PyTorch (CPU or GPU, depending on hardware)
  - Additional libraries like `sam2`, `sentence-transformers`, `soundfile`, `librosa`, `pydub`, and `pyaudio` for specific scripts

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

## Learn
- [Hugging Face Learn](https://huggingface.co/learn): Tutorials and courses on machine learning.
- [Hugging Face Documentation](https://huggingface.co/docs): Guides for the Hub, Transformers, Diffusers, and more.
- [Serverless API](https://huggingface.co/docs/api-inference/index): Documentation for the Inference API used by the C++ client.
- [Deep Learning Containers](https://huggingface.co/docs/sagemaker/index#deep-learning-containers): Amazon SageMaker and Google Cloud integrations.
- [PyTorch Tutorials](https://github.com/philschmid/deep-learning-pytorch-huggingface): Deep learning with PyTorch and Hugging Face.
- [NVIDIA Triton Server](https://github.com/triton-inference-server/tutorials/tree/main/HuggingFace): Deploying models with Triton.
- [Transformers Notebooks](https://github.com/qubvel/transformers-notebooks/tree/main/notebooks): Example notebooks for Transformers.

## Courses
- [DeepLearning.AI: Open Source Models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/)
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)
- [Hugging Face ML for 3D Course](https://huggingface.co/learn/ml-for-3d-course/unit0/introduction)
- [Udemy: Transformers in Computer Vision](https://www.udemy.com/course/transformers-in-computer-vision-english-version)

