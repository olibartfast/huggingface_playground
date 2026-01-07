#include "video_classification/videomae_image_processor.hpp"

#include <opencv2/opencv.hpp>

VideoMAEImageProcessor::VideoMAEImageProcessor(
    const rapidjson::Document &config) {
  image_size = config.HasMember("image_size") && config["image_size"].IsInt()
                   ? config["image_size"].GetInt()
                   : 224;
  mean = {0.485f, 0.456f, 0.406f};
  if (config.HasMember("mean") && config["mean"].IsArray()) {
    mean.clear();
    for (rapidjson::SizeType i = 0; i < config["mean"].Size() && i < 3; ++i) {
      if (config["mean"][i].IsFloat() || config["mean"][i].IsDouble()) {
        mean.push_back(config["mean"][i].GetFloat());
      }
    }
  }
  std = {0.229f, 0.224f, 0.225f};
  if (config.HasMember("std") && config["std"].IsArray()) {
    std.clear();
    for (rapidjson::SizeType i = 0; i < config["std"].Size() && i < 3; ++i) {
      if (config["std"][i].IsFloat() || config["std"][i].IsDouble()) {
        std.push_back(config["std"][i].GetFloat());
      }
    }
  }
}

std::vector<float>
VideoMAEImageProcessor::process(const std::vector<cv::Mat> &frames,
                                int channels, const std::string &format) {
  std::vector<float> pixel_values;
  for (const auto &frame : frames) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(image_size, image_size));

    cv::Mat float_frame;
    resized.convertTo(float_frame, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels_vec(static_cast<size_t>(channels));
    cv::split(float_frame, channels_vec);
    for (int c = 0; c < channels; ++c) {
      channels_vec[static_cast<size_t>(c)] =
          (channels_vec[static_cast<size_t>(c)] -
           mean[static_cast<size_t>(c)]) /
          std[static_cast<size_t>(c)];
      if (format == "FORMAT_NCHW" || format == "FORMAT_NONE") {
        // Ensure contiguous memory access via pointer if needed, or just
        // iterate Mat::data is uchar*, for float we need to cast or use
        // ptr<float> Using insert is cleaner but we need to be careful with
        // types. OpenCv Mat is row-major. insert expects iterators.
        if (channels_vec[static_cast<size_t>(c)].isContinuous()) {
          const float *ptr = channels_vec[static_cast<size_t>(c)].ptr<float>();
          pixel_values.insert(pixel_values.end(), ptr,
                              ptr + image_size * image_size);
        } else {
          // Fallback for non-continuous, though split usually produces
          // continuous
          for (int r = 0; r < channels_vec[static_cast<size_t>(c)].rows; ++r) {
            const float *ptr =
                channels_vec[static_cast<size_t>(c)].ptr<float>(r);
            pixel_values.insert(pixel_values.end(), ptr,
                                ptr +
                                    channels_vec[static_cast<size_t>(c)].cols);
          }
        }
      }
    }
    if (format == "FORMAT_NHWC") {
      for (int h = 0; h < image_size; ++h) {
        for (int w = 0; w < image_size; ++w) {
          for (int c = 0; c < channels; ++c) {
            pixel_values.push_back(
                channels_vec[static_cast<size_t>(c)].at<float>(h, w));
          }
        }
      }
    }
  }
  return pixel_values;
}
