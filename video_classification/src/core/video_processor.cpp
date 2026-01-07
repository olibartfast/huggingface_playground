#include "video_classification/video_processor.hpp"
#include <algorithm>
#include <cmath>

VideoProcessor::VideoProcessor() {}

VideoProcessor::~VideoProcessor() {
  if (cap.isOpened()) {
    cap.release();
  }
}

bool VideoProcessor::openVideo(const std::string &videoPath) {
  cap.open(videoPath);
  if (!cap.isOpened()) {
    return false;
  }

  info.totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  info.fps = cap.get(cv::CAP_PROP_FPS);
  info.duration = info.totalFrames / info.fps;
  return true;
}

VideoProcessor::VideoInfo VideoProcessor::getVideoInfo() const { return info; }

std::vector<VideoProcessor::WindowIndices>
VideoProcessor::splitVideoIntoWindows(int windowSize, float samplingFps) const {
  std::vector<WindowIndices> windows;
  int interval = std::max(static_cast<int>(info.fps / samplingFps), 1);
  std::vector<int> sampledIndices;

  for (int i = 0; i < info.totalFrames; i += interval) {
    sampledIndices.push_back(i);
  }

  for (size_t i = 0;
       i + static_cast<size_t>(windowSize) <= sampledIndices.size();
       i += static_cast<size_t>(windowSize)) {
    WindowIndices window;
    window.indices.assign(
        sampledIndices.begin() + static_cast<long>(i),
        sampledIndices.begin() +
            static_cast<long>(i + static_cast<size_t>(windowSize)));
    window.startTime = window.indices.front() / info.fps;
    window.endTime = window.indices.back() / info.fps;
    windows.push_back(window);
  }

  // Handle last window if needed
  if (!sampledIndices.empty() && windows.empty()) {
    WindowIndices window;
    window.indices = sampledIndices;
    window.startTime = window.indices.front() / info.fps;
    window.endTime = window.indices.back() / info.fps;
    windows.push_back(window);
  }

  return windows;
}

std::vector<cv::Mat>
VideoProcessor::extractFrames(const std::vector<int> &indices) {
  std::vector<cv::Mat> frames;
  int currentFrame = 0;

  for (int targetFrame : indices) {
    if (targetFrame < currentFrame) {
      cap.set(cv::CAP_PROP_POS_FRAMES, targetFrame);
      currentFrame = targetFrame;
    }

    while (currentFrame < targetFrame) {
      cap.grab();
      currentFrame++;
    }

    cv::Mat frame;
    if (cap.read(frame)) {
      frames.push_back(frame);
    }
    currentFrame++;
  }

  return frames;
}

// ===== IMAGE PROCESSOR CONFIGURATION =====
// Processor type: VideoMAEImageProcessor
// Image size: {'height': 224, 'width': 224}
// Image mean: [0.485, 0.456, 0.406]
// Image std: [0.229, 0.224, 0.225]
// Do rescale: True
// Rescale factor: 0.00392156862745098
// Do normalize: True
// Do resize: True
// Resample: 2
// Do center crop: True
// Crop size: {'height': 224, 'width': 224}

// All processor attributes:
//   crop_size: {'height': 224, 'width': 224}
//   do_center_crop: True
//   do_normalize: True
//   do_rescale: True
//   do_resize: True
//   image_mean: [0.485, 0.456, 0.406]
//   image_processor_type: VideoMAEImageProcessor
//   image_std: [0.229, 0.224, 0.225]
//   model_input_names: ['pixel_values']
//   resample: 2
//   rescale_factor: 0.00392156862745098
//   size: {'height': 224, 'width': 224}

std::vector<float>
VideoProcessor::preprocessFrames(const std::vector<cv::Mat> &frames,
                                 int targetSize) {
  // Initialize output vector for the preprocessed data
  // Shape: [1, num_frames, 3, height, width]
  std::vector<float> preprocessedData;
  preprocessedData.reserve(frames.size() * 3 * static_cast<size_t>(targetSize) *
                           static_cast<size_t>(targetSize));

  // VideoMAE preprocessing parameters (matching HuggingFace)
  const cv::Scalar mean(0.485, 0.456, 0.406); // RGB order
  const cv::Scalar std(0.229, 0.224, 0.225);
  const double rescale_factor = 0.00392156862745098; // 1/255

  for (const auto &frame : frames) {
    cv::Mat resized, cropped, float_img, normalized;

    // Step 1: Resize (maintaining aspect ratio if needed)
    // HuggingFace uses BILINEAR resampling (resample=2)
    cv::resize(frame, resized, cv::Size(targetSize, targetSize), 0, 0,
               cv::INTER_LINEAR);

    // Step 2: Center crop to exact target size (224x224)
    // If resize already gives us 224x224, this is essentially a no-op
    int crop_x = (resized.cols - targetSize) / 2;
    int crop_y = (resized.rows - targetSize) / 2;
    if (crop_x > 0 || crop_y > 0) {
      cv::Rect crop_rect(crop_x, crop_y, targetSize, targetSize);
      cropped = resized(crop_rect);
    } else {
      cropped = resized;
    }

    // Step 3: Convert BGR to RGB (OpenCV uses BGR, HuggingFace expects RGB)
    cv::Mat rgb_frame;
    cv::cvtColor(cropped, rgb_frame, cv::COLOR_BGR2RGB);

    // Step 4: Rescale - convert to float and apply rescale factor
    rgb_frame.convertTo(float_img, CV_32F);
    float_img *= rescale_factor; // Scale from [0,255] to [0,1]

    // Step 5: Normalize using ImageNet mean and std (in RGB order)
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels); // R, G, B channels

    // Normalize each channel: (pixel - mean) / std
    for (int c = 0; c < 3; c++) {
      channels[static_cast<size_t>(c)] =
          (channels[static_cast<size_t>(c)] - mean[c]) / std[c];
    }

    // Step 6: Convert to CHW format (Channel-Height-Width)
    // HuggingFace expects CHW format: [C, H, W]
    for (int c = 0; c < 3; c++) {
      const float *channelData = channels[static_cast<size_t>(c)].ptr<float>();
      preprocessedData.insert(preprocessedData.end(), channelData,
                              channelData +
                                  static_cast<size_t>(targetSize) *
                                      static_cast<size_t>(targetSize));
    }
  }

  return preprocessedData;
}

std::vector<cv::Mat>
VideoProcessor::padVideoFrames(const std::vector<cv::Mat> &frames,
                               int targetLength) {
  std::vector<cv::Mat> paddedFrames = frames;

  if (frames.size() >= static_cast<size_t>(targetLength)) {
    paddedFrames.resize(static_cast<size_t>(targetLength));
    return paddedFrames;
  }

  // Pad with last frame
  const cv::Mat &lastFrame = frames.back();
  while (paddedFrames.size() < static_cast<size_t>(targetLength)) {
    paddedFrames.push_back(lastFrame.clone());
  }

  return paddedFrames;
}