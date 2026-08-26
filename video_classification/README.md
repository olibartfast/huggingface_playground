# Video Classification Project

This project implements a C++ video classification application that performs inference on [Hugging Face video models](https://huggingface.co/docs/transformers/tasks/video_classification)  using Triton Inference Server and OpenCV.

> **Looking for the HMDB-51 benchmark?** See [`python/hmdb51`](python/hmdb51/README.md).
> It evaluates V-JEPA 2, VideoPrism and PE Video with frozen encoders and a linear
> probe. Those backbones ship no HMDB-51 classification head, so they cannot be
> served by the Triton pipeline documented here until a head is trained on the
> cached features.


## Prerequisites

- **Required**:
  - C++ Compiler with C++20 support (GCC 11+, Clang 14+, MSVC 19.30+)
  - CMake 3.25 or later
  - Git
  - Triton Inference Server (running and serving the `videomae_large` model)

- **Managed via vcpkg (automatically installed)**:
  - fmt
  - rapidjson
  - gtest
  - spdlog

- **System Dependencies** (Ensure these are installed):
  - OpenCV (Development libraries)
  - curl (for Http client)

## Building the Project

This project uses CMake Presets for easy configuration. The vcpkg dependency manager handles most external libraries.

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd video_classification
   ```

2. **Set Triton Client Path** (if not in default location):
   ```bash
   export TRITON_CLIENT_ROOT=/path/to/triton_client_libs/install
   # OR set via CMake cache variable
   ```

3. **Configure**:
   ```bash
   cmake --preset=debug
   # OR for release
   cmake --preset=release
   # OR specify Triton path directly
   cmake --preset=debug -DTRITON_CLIENT_ROOT=/path/to/triton
   ```

4. **Build**:
   ```bash
   cmake --build --preset=debug
   # OR for release
   cmake --build --preset=release
   ```

## Running the Application

The main executable `video_classification_app` takes a video file as input.

```bash
./build/debug/src/app/video_classification_app [options] <video_path>
```

### Options:
- `-m <model_name>`: Model name on Triton server (default: `videomae_large`)
- `-u <url>`: Triton server URL (default: `http://localhost:8000`)
- `-b <batch_size>`: Batch size (default: 1)
- `-l <labels_file>`: Path to labels file (default: `labels/kinetics400.txt`)
- `-c <config_file>`: Path to model configuration file (optional)
- `-t <model_type>`: Model type: `videomae`, `vivit`, or `timesformer` (default: `videomae`)

### Examples:
```bash
# Basic usage with default VideoMAE settings
./build/debug/src/app/video_classification_app /path/to/my/video.mp4

# Specify model type
./build/debug/src/app/video_classification_app -t vivit -m vivit_model /path/to/my/video.mp4

# Use custom config file
./build/debug/src/app/video_classification_app -c configs/custom_model.json /path/to/my/video.mp4

# Full example with all options
./build/debug/src/app/video_classification_app \
  -m videomae_large \
  -u http://localhost:8000 \
  -b 1 \
  -l labels/kinetics400.txt \
  -t videomae \
  /path/to/my/video.mp4
```

## Configuration Files

Model configurations can be specified via JSON files in the `configs/` directory. See `configs/videomae.json`, `configs/vivit.json`, and `configs/timesformer.json` for examples.

Example configuration:
```json
{
  "model_type": "videomae",
  "image_size": 224,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}
```

## Testing

Unit tests are managed by GoogleTest.

1. **Build Tests**:
   The tests are built as part of the main build (enabled by default).

2. **Run Tests**:
   ```bash
   cd build/debug
   ctest --output-on-failure
   ```

 # Resources
 - https://huggingface.co/docs/transformers/tasks/video_classification
 - https://huggingface.co/docs/transformers/model_doc/vjepa2
 - https://huggingface.co/docs/transformers/model_doc/pe_video
 - https://huggingface.co/docs/transformers/model_doc/videomae
 - https://huggingface.co/docs/transformers/model_doc/vivit
 - https://huggingface.co/docs/transformers/model_doc/timesformer
 
