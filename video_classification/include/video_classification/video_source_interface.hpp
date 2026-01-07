#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace video_classification {

class IVideoSource {
public:
  virtual ~IVideoSource() = default;

  virtual bool open(const std::string &path) = 0;
  virtual bool isOpened() const = 0;
  virtual void release() = 0;
  virtual bool read(cv::OutputArray image) = 0;
  virtual double get(int propId) const = 0;
  virtual bool set(int propId, double value) = 0;

  // Helper to grab a frame (for efficiency if needed, mirroring
  // cv::VideoCapture::grab)
  virtual bool grab() = 0;
};

} // namespace video_classification
