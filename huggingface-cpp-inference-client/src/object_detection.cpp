#include "object_detection.hpp"
#include "image_processing.hpp"
#include <iostream>

ObjectDetection::ObjectDetection(const std::string& endpoint, const std::string& authToken, 
                                 const nlohmann::json& params)
    : HuggingFaceTask(endpoint, authToken) {
    imagePath = params["image_path"].get<std::string>();
    threshold = params.value("threshold", 0.5);
}

nlohmann::json ObjectDetection::preparePayload() const {
    // Convert image to base64
    std::string base64Image = ImageProcessing::encodeImage(imagePath, 224);  // 224 is an example size, adjust as needed
    return nlohmann::json{
        {"inputs", base64Image},
        {"parameters", {{"threshold", threshold}}}
    };
}

std::string ObjectDetection::execute() {
    std::string response = HuggingFaceTask::execute();
    nlohmann::json result = nlohmann::json::parse(response);
    
    // Process and print the results
    std::cout << "Detected objects:\n";
    for (const auto& obj : result) {
        std::cout << "Label: " << obj["label"] << ", Score: " << obj["score"] << "\n";
        std::cout << "Bounding box: "
                  << "(" << obj["box"]["xmin"] << ", " << obj["box"]["ymin"] << ") - "
                  << "(" << obj["box"]["xmax"] << ", " << obj["box"]["ymax"] << ")\n";
    }
    
    return result.dump();
}
