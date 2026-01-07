#include "video_classification/video_utils.hpp"
#include <iostream>
#include <stdexcept>

std::vector<cv::Mat> read_video_frames(const std::string &video_path,
                                       int target_frames) {
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open video: " + video_path);
  }

  // Get video properties
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) {
    cap.release();
    throw std::runtime_error("Invalid FPS for video: " + video_path);
  }
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  double duration = total_frames / fps; // Duration in seconds

  // Calculate frame indices for 1 FPS sampling
  std::vector<int> indices;
  int available_seconds = std::min(static_cast<int>(duration), target_frames);
  for (int i = 0; i < available_seconds; ++i) {
    int frame_idx = static_cast<int>(i * fps);
    if (frame_idx < total_frames) {
      indices.push_back(frame_idx);
    }
  }

  // Read frames
  std::vector<cv::Mat> frames;
  for (int idx : indices) {
    cap.set(cv::CAP_PROP_POS_FRAMES, idx);
    cv::Mat frame;
    if (!cap.read(frame)) {
      std::cerr << "Warning: Failed to read frame at index " << idx
                << " (time: " << idx / fps << "s)" << std::endl;
      continue;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frames.push_back(frame.clone());
  }
  cap.release();

  if (frames.empty()) {
    throw std::runtime_error("No frames could be read from video: " +
                             video_path);
  }
  return frames;
}

std::vector<cv::Mat> pad_video_frames(const std::vector<cv::Mat> &frames,
                                      int target_length) {
  if (frames.empty()) {
    throw std::runtime_error("Cannot pad empty frames; ensure video has at "
                             "least one readable frame");
  }
  std::vector<cv::Mat> padded = frames;
  while (padded.size() < static_cast<size_t>(target_length)) {
    padded.push_back(padded.back().clone()); // Duplicate last frame
  }
  return padded;
}
