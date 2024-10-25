#include "image_segmentation.hpp"
#include "image_processing.hpp"


nlohmann::json ImageSegmentation::preparePayload() const {
    std::string base64Image = ImageProcessing::encodeImage(imagePath, targetSize, resize);

    nlohmann::json payload;
    payload["inputs"] = base64Image;
    
    nlohmann::json parameters;
    if (maskThreshold > 0) parameters["mask_threshold"] = maskThreshold;
    if (overlapMaskAreaThreshold > 0) parameters["overlap_mask_area_threshold"] = overlapMaskAreaThreshold;
    if (!subtask.empty()) parameters["subtask"] = subtask;
    if (threshold > 0) parameters["threshold"] = threshold;

    if (!parameters.empty()) {
        payload["parameters"] = parameters;
    }

    return payload;
}

ImageSegmentation::ImageSegmentation(const std::string& endpoint, const std::string& authToken, 
                                     const std::string& imagePath, 
                                     double maskThreshold, 
                                     double overlapMaskAreaThreshold, 
                                     const std::string& subtask, 
                                     double threshold,
                                     int targetSize,
                                     bool resize)
    : HuggingFaceTask(endpoint, authToken), 
      imagePath(imagePath), 
      maskThreshold(maskThreshold), 
      overlapMaskAreaThreshold(overlapMaskAreaThreshold), 
      subtask(subtask), 
      threshold(threshold),
      targetSize(targetSize),
      resize(resize) {}

std::string ImageSegmentation::execute() {
    std::string response = HuggingFaceTask::execute();
    nlohmann::json result = nlohmann::json::parse(response);

    // Process and print the results
    std::cout << "Instance Segmentation Results:\n";
    for (const auto& segment : result) {
        std::cout << "Label: " << segment["label"] << ", Score: " << segment["score"] << "\n";
    }

    return result.dump();
}
