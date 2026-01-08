
#include "video_classification/triton_client.hpp"
#include "video_classification/videomae_image_processor.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <stdexcept>
#include <vector>

#include "video_classification/video_utils.hpp"

int main(int argc, char **argv) {
  std::string model_name = "videomae_large";
  std::string url = "http://localhost:8000";
  std::string video_path;
  std::string labels_file = "labels/kinetics400.txt"; 
  int batch_size = 1;
  int window_size = 16;

  // Parse command-line arguments
  int opt;
  while ((opt = getopt(argc, argv, "m:u:b:l:")) != -1) {
    switch (opt) {
    case 'm':
      model_name = optarg;
      break;
    case 'u':
      url = optarg;
      break;
    case 'b':
      batch_size = std::atoi(optarg);
      break;
    case 'l':
      labels_file = optarg;
      break;
    default:
      std::cerr << "Usage: " << argv[0]
                << " [-m model] [-u url] [-b batch_size] [-l labels_file] "
                   "<video_path>\n";
      exit(1);
    }
  }
  if (optind >= argc) {
    std::cerr << "Error: Video file must be specified\n";
    exit(1);
  }
  video_path = argv[optind];
  if (batch_size <= 0) {
    std::cerr << "Error: Batch size must be > 0\n";
    exit(1);
  }

  try {
    // Initialize Triton client
    TritonClient client(url, labels_file);

    // Get model info
    ModelInfo model_info;
    client.get_model_info(model_name, model_info);

    // Initialize processor
    std::string config_json =
        R"({"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})";
    rapidjson::Document config;
    config.Parse(config_json.c_str());
    if (config.HasParseError()) {
      throw std::runtime_error("Failed to parse config JSON");
    }
    VideoMAEImageProcessor processor(config);

    // Read video frames at 1 FPS
    auto frames = read_video_frames(video_path, window_size);
    frames = pad_video_frames(frames, window_size);

    // Verify frame count
    if (frames.size() != static_cast<size_t>(window_size)) {
      throw std::runtime_error("Expected " + std::to_string(window_size) +
                               " frames, got " + std::to_string(frames.size()));
    }

    // Preprocess frames
    auto pixel_values = processor.process(frames, model_info.input_c_,
                                          model_info.input_format_);

    // Validate input data size
    const size_t expected_elements = static_cast<size_t>(batch_size) *
                                     static_cast<size_t>(window_size) *
                                     static_cast<size_t>(model_info.input_c_) *
                                     static_cast<size_t>(model_info.input_h_) *
                                     static_cast<size_t>(model_info.input_w_);
    if (pixel_values.size() != expected_elements) {
      throw std::runtime_error("Invalid input data size: expected " +
                               std::to_string(expected_elements) +
                               " elements, got " +
                               std::to_string(pixel_values.size()));
    }

    // Set input shape
    std::vector<int64_t> shape = {batch_size, window_size, model_info.input_c_,
                                  model_info.input_h_, model_info.input_w_};

    // Perform inference
    auto results = client.infer(pixel_values, model_name, model_info, shape);

    // Output results
    std::cout << "Predictions for video '" << video_path << "':\n";
    for (const auto &result : results) {
      std::cout << "  " << result.label << ": " << result.probability << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}