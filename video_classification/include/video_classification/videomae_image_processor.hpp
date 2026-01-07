#pragma once
#include "image_processor.hpp"
#include <rapidjson/document.h>
#include <string>
#include <vector>

class VideoMAEImageProcessor : public ImageProcessor {
public:
  explicit VideoMAEImageProcessor(const rapidjson::Document &config);

  std::vector<float> process(const std::vector<cv::Mat> &frames, int channels,
                             const std::string &format) override;

private:
  int image_size;
  std::vector<float> mean;
  std::vector<float> std;
};
