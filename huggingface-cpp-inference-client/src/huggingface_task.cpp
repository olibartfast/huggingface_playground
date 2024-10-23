#include "huggingface_task.hpp"
#include "object_detection.hpp"
#include "image_classification.hpp"
#include <stdexcept>

HuggingFaceTask::HuggingFaceTask(const std::string& endpoint, const std::string& authToken)
    : apiEndpoint(endpoint), token(authToken) {}

std::unique_ptr<HuggingFaceTask> HuggingFaceTaskFactory::createTask(
    const std::string& taskType,
    const std::string& endpoint,
    const std::string& authToken,
    const nlohmann::json& params
) {
    if (taskType == "object-detection") {
        return std::make_unique<ObjectDetection>(endpoint, authToken, params);
    } else if (taskType == "image-classification") {
        return std::make_unique<ImageClassification>(endpoint, authToken, params);
    } else {
        throw std::runtime_error("Unknown task type: " + taskType);
    }
}