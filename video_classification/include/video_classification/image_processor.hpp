#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ImageProcessor {
public:
  virtual ~ImageProcessor() = default;
  virtual std::vector<float> process(const std::vector<cv::Mat> &frames,
                                     int channels,
                                     const std::string &format) = 0;
};
