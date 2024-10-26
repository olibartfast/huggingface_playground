#include "image_text_to_text.hpp"
#include "curl_wrapper.hpp"
#include "image_processing.hpp"

ImageTextToText::ImageTextToText(const std::string& endpoint, const std::string& authToken, 
                                 const std::vector<std::string>& images, 
                                 const std::string& prompt,  
                                 int targetSize, bool resize)
    : HuggingFaceTask(endpoint, authToken), 
      images(images), 
      prompt(prompt), 
      targetSize(targetSize), 
      resize(resize) 
      {
      }

std::string ImageTextToText::execute() {
    nlohmann::json payload = preparePayload();
    payload["inputs"] = payload["messages"];
    std::string payloadStr = payload.dump();

    try {
        std::string response = CurlWrapper()
            .setUrl(apiEndpoint + "/v1/chat/completions")
            .setPostFields(payloadStr)
            .addHeader("Content-Type: application/json")
            .addHeader("Authorization: Bearer " + token)
            .perform();

        return response;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("HTTP request failed: ") + e.what());
    }
}

nlohmann::json ImageTextToText::preparePayload() const {
    nlohmann::json payload;
    // Extract model name from endpoint URL
    size_t pos = apiEndpoint.find("/models/");
    std::string model_name = apiEndpoint.substr(pos + 8);
    payload["model"] = model_name;
    payload["messages"] = {
        {{"role", "user"}, {"content", {}}}
    };

    // Add images to the payload
    for (const auto& image : images) {
        std::string base64Image = ImageProcessing::encodeImage(image, targetSize, resize);
        payload["messages"][0]["content"].push_back({
            {"type", "image_url"},
            {"image_url", {{"url", "data:image/jpeg;base64," + base64Image}}}
        });
    }

    // Add prompt to the payload
    payload["messages"][0]["content"].push_back({
        {"type", "text"},
        {"text", prompt}
    });

    payload["max_tokens"] = 500;
    payload["stream"] = false;

    return payload;
}