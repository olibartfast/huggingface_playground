#include "image_processing.hpp"
#include "base64.h" 
#include <stdexcept>

std::string ImageProcessing::encodeImage(const std::string& image_path, int target_size, bool resize) {
    cv::Mat image = readImage(image_path);

    if (resize) {
        cv::Mat resized_image = resizeImage(image, target_size);
        cv::Mat squared_image = createSquareCanvas(resized_image, target_size);
        std::vector<unsigned char> jpg_data = encodeToJpg(squared_image);
        return encodeToBase64(jpg_data);
    } else {
        std::vector<unsigned char> jpg_data = encodeToJpg(image);
        return encodeToBase64(jpg_data);
    }
}

cv::Mat ImageProcessing::readImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to open image file: " + image_path);
    }
    return image;
}

cv::Mat ImageProcessing::resizeImage(const cv::Mat& image, int target_size) {
    cv::Size new_size = calculateNewSize(image, target_size);
    cv::Mat resized_image;
    cv::resize(image, resized_image, new_size);
    return resized_image;
}

cv::Mat ImageProcessing::createSquareCanvas(const cv::Mat& image, int target_size) {
    int top = (target_size - image.rows) / 2;
    int bottom = target_size - image.rows - top;
    int left = (target_size - image.cols) / 2;
    int right = target_size - image.cols - left;

    cv::Mat squared_image;
    cv::copyMakeBorder(image, squared_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return squared_image;
}

std::vector<unsigned char> ImageProcessing::encodeToJpg(const cv::Mat& image) {
    std::vector<unsigned char> buf;
    cv::imencode(".jpg", image, buf);
    return buf;
}

std::string ImageProcessing::encodeToBase64(const std::vector<unsigned char>& data) {
    return base64_encode(data.data(), data.size());
}

std::vector<unsigned char> ImageProcessing::decodeBase64(const std::string& encoded_string) {
    std::string decoded = base64_decode(encoded_string);
    return std::vector<unsigned char>(decoded.begin(), decoded.end());
}

cv::Size ImageProcessing::calculateNewSize(const cv::Mat& image, int target_size) {
    float aspect_ratio = (float)image.cols / (float)image.rows;
    int new_width, new_height;

    if (aspect_ratio >= 1.0f) {
        new_width = target_size;
        new_height = static_cast<int>(target_size / aspect_ratio);
    } else {
        new_height = target_size;
        new_width = static_cast<int>(target_size * aspect_ratio);
    }

    return cv::Size(new_width, new_height);
}