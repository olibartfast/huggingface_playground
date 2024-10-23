#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <nlohmann/json.hpp>
#include "curl_wrapper.hpp"

class HuggingFaceTask {
protected:
    std::string apiEndpoint;
    std::string token;

    virtual nlohmann::json preparePayload() const = 0;

public:
    HuggingFaceTask(const std::string& endpoint, const std::string& authToken);
    virtual ~HuggingFaceTask() = default;

    virtual std::string execute() {
        CurlWrapper curl;
        nlohmann::json payload = preparePayload();
        std::string payloadStr = payload.dump();

        try {
            std::string response = curl.setUrl(apiEndpoint)
                .setPostFields(payloadStr)
                .addHeader("Content-Type: application/json")
                .addHeader("Authorization: Bearer " + token)
                .perform();

            return response;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("HTTP request failed: ") + e.what());
        }
    }
};

class HuggingFaceTaskFactory {
public:
    static std::unique_ptr<HuggingFaceTask> createTask(
        const std::string& taskType,
        const std::string& endpoint,
        const std::string& authToken,
        const nlohmann::json& params
    );
};
