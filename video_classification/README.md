# Video Classification Project

This project implements a C++ video classification application that performs inference on [Hugging Face video models](https://huggingface.co/docs/transformers/tasks/video_classification)  using Triton Inference Server and OpenCV.


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

2. **Configure**:
   ```bash
   cmake --preset=debug
   # OR for release
   cmake --preset=release
   ```

3. **Build**:
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

### Example:
```bash
./build/debug/src/app/video_classification_app -l labels/kinetics400.txt /path/to/my/video.mp4
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
 - https://huggingface.co/docs/transformers/model_doc/videomae
 - https://huggingface.co/docs/transformers/model_doc/vivit
 - https://huggingface.co/docs/transformers/model_doc/timesformer
