#include "huggingface_task.hpp"
#include <iostream>

int main() {
    // read environment variable hugging face token
    std::string authToken = std::getenv("HF_TOKEN");
    
    try {
        // Example usage for object detection
        std::string image_path = "cat.jpeg";
        
        auto objectDetectionTask = HuggingFaceTaskFactory::createTask(
            "object-detection",
            "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
            authToken,
            nlohmann::json{{"image_path", image_path}, {"threshold", 0.7}}
        );
        std::cout << "Object Detection Result:\n" << objectDetectionTask->execute() << std::endl;

        auto imageClassificationTask = HuggingFaceTaskFactory::createTask(
            "image-classification",
            "https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
            authToken,
            nlohmann::json{{"image_path", image_path}}
        );
        std::cout << "Image Classification Result:\n" << imageClassificationTask->execute() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

