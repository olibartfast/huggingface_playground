
#include "image_classification.hpp"
#include "image_processing.hpp"
#include <iostream>


ImageClassification::ImageClassification(const std::string& endpoint, const std::string& authToken, 
                                         const nlohmann::json& params)
    : HuggingFaceTask(endpoint, authToken) {
    imagePath = params["image_path"].get<std::string>();
}



nlohmann::json ImageClassification::preparePayload() const {
    // Convert image to base64
    std::string base64Image = ImageProcessing::encodeImage(imagePath, 224);  // 224 is a common size for many classification models, adjust as needed
    return nlohmann::json{{"inputs", base64Image}};
}

std::string ImageClassification::execute() {
    std::string response = HuggingFaceTask::execute();
    nlohmann::json result = nlohmann::json::parse(response);
    
    // Process and print the results
    std::cout << "Image Classification Results:\n";
    for (const auto& classification : result) {
        std::cout << "Label: " << classification["label"].get<std::string>() 
                  << ", Score: " << classification["score"].get<double>() << "\n";
    }
    
    return response;
}

