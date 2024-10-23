#pragma once
#include "huggingface_task.hpp"
#include <vector>

class ObjectDetection : public HuggingFaceTask {
private:
    std::string imagePath;
    double threshold;

protected:
    nlohmann::json preparePayload() const override;

public:
    ObjectDetection(const std::string& endpoint, const std::string& authToken, 
                    const nlohmann::json& params);
    std::string execute() override;
};

