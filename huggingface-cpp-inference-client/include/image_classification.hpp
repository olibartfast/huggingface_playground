#pragma once

#include "huggingface_task.hpp"
#include <vector>

class ImageClassification : public HuggingFaceTask {
private:
    std::string imagePath;

protected:
    nlohmann::json preparePayload() const override;

public:
    ImageClassification(const std::string& endpoint, const std::string& authToken, 
                       const nlohmann::json& params);
    std::string execute() override; 
};
