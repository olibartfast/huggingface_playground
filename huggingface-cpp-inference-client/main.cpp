#include <iostream>
#include <string>
#include <string_view>
#include <optional>
#include <variant>
#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

// Assuming these are your custom headers
#include "huggingface_task.hpp"

namespace fs = std::filesystem;

std::optional<std::string> getEnvVar(std::string_view key) {
    if (auto val = std::getenv(key.data()))
        return std::string(val);
    return std::nullopt;
}

template<typename T>
using Result = std::variant<T, std::string>;

void drawBoundingBoxes(const nlohmann::json& result, const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return;
    }

    for (const auto& detection : result) {
        std::string label = detection["label"];
        float score = detection["score"];
        auto box = detection["box"];

        int xmin = box["xmin"];
        int ymin = box["ymin"];
        int xmax = box["xmax"];
        int ymax = box["ymax"];

        cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);

        std::string label_with_score = label + " " + std::to_string(score).substr(0, 4);
        cv::putText(image, label_with_score, cv::Point(xmin, ymin - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Object Detection Result", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Optionally, save the image
    cv::imwrite("result_" + std::filesystem::path(imagePath).filename().string(), image);
}

template<typename T>
void processResult(const Result<T>& result, const std::string& taskType, const std::string& imagePath) {
    if (auto value = std::get_if<T>(&result)) {
        std::cout << "Success: " << value->dump(2) << std::endl;
        if (taskType == "object-detection") {
            drawBoundingBoxes(*value, imagePath);
        }
    } else {
        std::cerr << "Error: " << std::get<std::string>(result) << std::endl;
    }
}

Result<nlohmann::json> executeTask(const std::string& taskType, 
                                   const std::string& model, 
                                   const std::string& authToken, 
                                   const nlohmann::json& params) {
    try {
        auto task = HuggingFaceTaskFactory::createTask(taskType, model, authToken, params);
        std::string response = task->execute();
        return nlohmann::json::parse(response);
    } catch (const std::exception& e) {
        return std::string(e.what());
    }
}


int main() {
    auto authToken = getEnvVar("HF_TOKEN");
    if (!authToken) {
        std::cerr << "Error: HF_TOKEN environment variable is not set." << std::endl;
        return 1;
    }

    const fs::path image_path = "cat.jpeg";
    if (!fs::exists(image_path)) {
        std::cerr << "Error: Image file does not exist." << std::endl;
        return 1;
    }

    const auto objectDetectionResult = executeTask(
        "object-detection",
        "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
        *authToken,
        nlohmann::json{{"image_path", image_path.string()}, {"threshold", 0.7}}
    );
    processResult(objectDetectionResult, "object-detection", image_path.string());

    const auto imageClassificationResult = executeTask(
        "image-classification",
        "https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
        *authToken,
        nlohmann::json{{"image_path", image_path.string()}}
    );
    processResult(imageClassificationResult, "image-classification", image_path.string());

    return 0;
}