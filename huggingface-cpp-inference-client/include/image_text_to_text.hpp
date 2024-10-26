#pragma once
#include "huggingface_task.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

class ImageTextToText : public HuggingFaceTask {
public:
    ImageTextToText(const std::string& endpoint, const std::string& authToken, 
                    const std::vector<std::string>& images, 
                    const std::string& prompt,  
                    int targetSize = 1024,
                    bool resize = false);

    std::string execute() override;

protected:
    nlohmann::json preparePayload() const override;

private:
    std::vector<std::string> images;
    std::string prompt;
    int targetSize;
    bool resize;
};