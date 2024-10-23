#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base64.h"

class ImageProcessing {
public:
    ImageProcessing() = default;
    ~ImageProcessing() = default;

    static std::string encodeImage(const std::string& image_path, int target_size, bool resize = false);

    // Public methods for testing
    static cv::Mat readImage(const std::string& image_path);
    static cv::Mat resizeImage(const cv::Mat& image, int target_size);
    static cv::Mat createSquareCanvas(const cv::Mat& image, int target_size);
    static std::vector<unsigned char> encodeToJpg(const cv::Mat& image);
    static std::string encodeToBase64(const std::vector<unsigned char>& data);

private:
    static cv::Size calculateNewSize(const cv::Mat& image, int target_size);
};
