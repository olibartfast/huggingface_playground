# Hugging Face C++ Inference Client

A modern C++17 client library for interacting with the Hugging Face Inference API, designed for computer vision tasks with built-in visualization capabilities.

## Features

- **Multi-task Support**: Object detection, image classification, image segmentation, and image-to-text
- **Visual Results**: Built-in OpenCV integration for visualizing detection results with bounding boxes and segmentation masks
- **Modern C++ Design**: Uses C++17 features, smart pointers, and RAII principles
- **Factory Pattern**: Extensible architecture for adding new tasks
- **Error Handling**: Robust error handling with std::variant for result types
- **Cross-platform**: CMake build system with automatic dependency management

## Supported Tasks

### Object Detection
- **Models**: Facebook DETR ResNet-50
- **Features**: Bounding box visualization, confidence thresholding
- **Output**: JSON with detected objects, confidence scores, and coordinates

### Image Classification
- **Models**: Google ViT (Vision Transformer)
- **Features**: Top-K classification results
- **Output**: JSON with class labels and confidence scores

### Image Segmentation
- **Models**: NVIDIA SegFormer
- **Features**: Semantic and instance segmentation with colored mask overlay
- **Output**: Base64-encoded masks with labels

### Image-to-Text
- **Models**: Meta Llama 3.2 Vision
- **Features**: Image captioning and visual question answering
- **Output**: Generated text descriptions

## Prerequisites

### Dependencies
- **CMake**: 3.20 or higher
- **C++17**: Compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- **OpenCV**: 4.x for image processing and visualization
- **libcurl**: For HTTP requests
- **nlohmann/json**: JSON parsing (auto-downloaded)

### External Libraries (Auto-downloaded)
- **cpp-base64**: Base64 encoding/decoding
- **cxxopts**: Command-line argument parsing

## Installation

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake libcurl4-openssl-dev libopencv-dev nlohmann-json3-dev
```

### macOS (with Homebrew)
```bash
brew install cmake curl opencv nlohmann-json
```

### Build from Source
```bash
# Clone the repository
git clone <repository-url>
cd huggingface-cpp-inference-client

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Configuration

### Environment Setup
Set your Hugging Face API token:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

## Usage

### Command Line Interface
```bash
./huggingface_app --input <image_path> --task <task_type> [--model <model_name>]
```

### Examples

#### Object Detection
```bash
./huggingface_app --input cat.jpeg --task object-detection
```

#### Image Classification
```bash
./huggingface_app --input cat.jpeg --task image-classification
```

#### Image Segmentation
```bash
./huggingface_app --input cat.jpeg --task image-segmentation
```

#### Image-to-Text
```bash
./huggingface_app --input cat.jpeg --task image-text-to-text
```

### Output
- **Console**: JSON results with detection data
- **Visual**: OpenCV windows showing annotated images
- **Files**: Saved result images with prefix `result_`

## Architecture

### Core Components

#### HuggingFaceTask (Base Class)
- Abstract base class for all inference tasks
- Handles authentication and API communication
- Template method pattern for extensibility

#### Task Implementations
- `ObjectDetection`: DETR-based object detection
- `ImageClassification`: Vision Transformer classification
- `ImageSegmentation`: SegFormer semantic segmentation
- `ImageTextToText`: Multimodal vision-language models

#### Utilities
- `CurlWrapper`: HTTP client abstraction
- `ImageProcessing`: Base64 encoding/decoding utilities
- `HuggingFaceTaskFactory`: Factory for creating task instances

### Design Patterns
- **Factory Pattern**: Task creation and management
- **Template Method**: Common API interaction flow
- **RAII**: Automatic resource management
- **Variant Types**: Type-safe error handling

## API Reference

### Core Classes

```cpp
// Base task class
class HuggingFaceTask {
public:
    HuggingFaceTask(const std::string& endpoint, const std::string& authToken);
    virtual std::string execute() = 0;
protected:
    virtual nlohmann::json preparePayload() const = 0;
};

// Factory for task creation
class HuggingFaceTaskFactory {
public:
    static std::unique_ptr<HuggingFaceTask> createTask(
        const std::string& taskType,
        const std::string& endpoint, 
        const std::string& authToken,
        const nlohmann::json& params
    );
};
```

### Task-Specific Parameters

#### Object Detection
```json
{
    "image_path": "path/to/image.jpg",
    "threshold": 0.7
}
```

#### Image Segmentation
```json
{
    "image_path": "path/to/image.jpg",
    "mask_threshold": 0.7,
    "overlap_mask_area_threshold": 0.5,
    "subtask": "semantic",
    "threshold": 0.9
}
```

## Development

### Adding New Tasks

1. **Create Header**: Define task class inheriting from `HuggingFaceTask`
2. **Implement Methods**: Override `preparePayload()` and `execute()`
3. **Register Factory**: Add task creation logic to factory
4. **Update CMake**: Add source files to build system

### Example Task Implementation
```cpp
class CustomTask : public HuggingFaceTask {
private:
    std::string imagePath;
    // Task-specific parameters
    
protected:
    nlohmann::json preparePayload() const override {
        // Prepare API request payload
    }
    
public:
    CustomTask(const std::string& endpoint, const std::string& authToken, 
               const nlohmann::json& params);
    std::string execute() override;
};
```

## Performance Notes

- **Model Loading**: First request per model may have cold start latency
- **Image Size**: Larger images increase processing time and bandwidth
- **Batch Processing**: Single image per request (batch support planned)
- **Memory Usage**: OpenCV operations require sufficient RAM for image processing

## Error Handling

The client provides comprehensive error handling:
- **Network Errors**: Connection timeouts, DNS resolution failures
- **API Errors**: Invalid tokens, model unavailability, quota limits
- **File Errors**: Missing images, unsupported formats
- **JSON Parsing**: Malformed API responses

## Troubleshooting

### Common Issues

1. **Missing HF_TOKEN**: Set environment variable with valid token
2. **OpenCV Display**: Ensure X11 forwarding for remote systems
3. **Model Unavailability**: Some models may be temporarily unavailable
4. **Large Images**: Resize images if processing times are excessive

### Debug Mode
Enable verbose output by setting:
```bash
export HF_DEBUG=1
```

## Roadmap

- [ ] Batch processing support
- [ ] Async/concurrent requests
- [ ] Additional vision tasks (OCR, face detection)
- [ ] Model caching for faster repeated inference
- [ ] Python bindings
- [ ] Docker containerization
- [ ] Performance benchmarking suite

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-task`)
3. Commit changes (`git commit -am 'Add new task'`)
4. Push to branch (`git push origin feature/new-task`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For providing the Inference API
- **OpenCV**: Computer vision library
- **nlohmann/json**: Modern JSON library for C++
- **cpp-base64**: Base64 encoding utilities
- **cxxopts**: Command-line parsing library
