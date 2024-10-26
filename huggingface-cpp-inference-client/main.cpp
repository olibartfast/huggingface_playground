#include <iostream>
#include <string>
#include <string_view>
#include <optional>
#include <variant>
#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "huggingface_task.hpp"
#include "image_classification.hpp"
#include "object_detection.hpp"
#include "image_segmentation.hpp"
#include "image_text_to_text.hpp"
#include "image_processing.hpp"
#include <cxxopts.hpp>

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

void drawImageSegmentationMasks(const nlohmann::json& jsonResult, const std::string& imagePath) {
    // Load the input image
    cv::Mat image = cv::imread(imagePath);

    // Create a map to store the colors for each label
    std::map<std::string, cv::Scalar> labelColors;

    // Draw the masks on the input image
    for (const auto& segment : jsonResult) {
        std::string label = segment["label"];
        std::string base64Mask = segment["mask"];
        std::vector<unsigned char> decodedMask = ImageProcessing::decodeBase64(base64Mask);
        cv::Mat mask = cv::imdecode(decodedMask, cv::IMREAD_GRAYSCALE);

        // Resize the mask to match the input image size
        cv::resize(mask, mask, image.size());

        // Assign a color to the label if it doesn't exist
        if (labelColors.find(label) == labelColors.end()) {
            // Generate a random color for the label
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;
            labelColors[label] = cv::Scalar(b, g, r);
        }

        // Draw the mask on the input image
        cv::Mat coloredMask;
        cv::cvtColor(mask, coloredMask, cv::COLOR_GRAY2BGR);
        coloredMask *= 255; // Convert to 0-255 range
        coloredMask.setTo(labelColors[label], mask); // Set color to the label color
        cv::addWeighted(image, 1, coloredMask, 0.5, 0, image);
    }

    // Display the output image
    cv::imshow("Instance Segmentation Result", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Save the output image
    cv::imwrite("result_" + std::filesystem::path(imagePath).filename().string(), image);
}


template<typename T>
void processResult(const Result<T>& result, const std::string& taskType, const std::string& imagePath) {
    if (auto value = std::get_if<T>(&result)) {
        
        if (taskType == "object-detection") {
            drawBoundingBoxes(*value, imagePath);
            std::cout << "Success: " << value->dump(2) << std::endl;
        } else if (taskType == "image-segmentation") {
            drawImageSegmentationMasks(*value, imagePath);
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

int main(int argc, char** argv) 
{
    cxxopts::Options options("Hugging Face Serverless API C++ Inference client", "Explore the most popular huggingface models with a single API request");
    options.add_options()
        ("h,help", "Print help")
        ("i,input", "Input source", cxxopts::value<std::string>())
        ("t,task", "Task type", cxxopts::value<std::string>())
        ("m,model", "Model name", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }


    auto authToken = getEnvVar("HF_TOKEN");
    if (!authToken) {
        std::cerr << "Error: HF_TOKEN environment variable is not set." << std::endl;
        return 1;
    }

    const fs::path image_path = result["input"].as<std::string>();
    if (!fs::exists(image_path)) {
        std::cerr << "Error: Image file does not exist." << std::endl;
        return 1;
    }


    const std::string taskType = result["task"].as<std::string>();

    if (taskType == "object-detection") {
        const auto objectDetectionResult = executeTask(
            taskType,
            "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
            *authToken,
            nlohmann::json{{"image_path", image_path.string()}, {"threshold", 0.7}}
        );
        processResult(objectDetectionResult, taskType, image_path.string());
    } else if (taskType == "image-segmentation") {
        const auto imageSegmentationResult = executeTask(
            taskType,
            "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512",
            *authToken,
            nlohmann::json{
                {"image_path", image_path.string()},
                {"mask_threshold", 0.7},
                {"overlap_mask_area_threshold", 0.5},
                {"subtask", "semantic"},
                {"threshold", 0.9}
            }
        );

        processResult(imageSegmentationResult, taskType, image_path.string());
    }
    else if (taskType == "image-classification") {
        const auto imageClassificationResult = executeTask(
            taskType,
            "https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
            *authToken,
            nlohmann::json{{"image_path", image_path.string()}}
        );
        processResult(imageClassificationResult, taskType, image_path.string());
    }
    else if (taskType == "image-text-to-text") {
        ImageTextToText task(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct",
            *authToken,
            {image_path.string()},
            "Describe this image in one sentence.",
            512,
            true
        );
        std::string response = task.execute();
        std::cout << "Response: " << response << std::endl;
    }
    else {
        std::cerr << "Error: Invalid task type." << std::endl;
        return 1;
    }

    return 0;
}