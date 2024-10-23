#include "curl_wrapper.hpp"
#include <stdexcept>
#include <cstring>

// CurlGlobalManager implementation
CurlGlobalManager& CurlGlobalManager::getInstance() {
    static CurlGlobalManager instance;
    return instance;
}

CurlGlobalManager::CurlGlobalManager() {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        throw std::runtime_error("Failed to initialize libcurl");
    }
}

CurlGlobalManager::~CurlGlobalManager() {
    curl_global_cleanup();
}

// CurlEasyHandle implementation
CurlEasyHandle::CurlEasyHandle() : handle(curl_easy_init()) {
    if (!handle) {
        throw std::runtime_error("Failed to create CURL handle");
    }
}

CurlEasyHandle::~CurlEasyHandle() {
    if (handle) {
        curl_easy_cleanup(handle);
    }
}

CURL* CurlEasyHandle::get() {
    return handle;
}

// CurlWrapper implementation
size_t CurlWrapper::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
        return newLength;
    } catch(std::bad_alloc& e) {
        return 0;
    }
}

CurlWrapper::CurlWrapper() : headers(nullptr) {
    CurlGlobalManager::getInstance(); // Ensure global initialization
    curl_easy_setopt(easyHandle.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(easyHandle.get(), CURLOPT_WRITEDATA, &responseBuffer);
}

CurlWrapper::~CurlWrapper() {
    if (headers) {
        curl_slist_free_all(headers);
    }
}

CurlWrapper& CurlWrapper::setUrl(const std::string& url) {
    curl_easy_setopt(easyHandle.get(), CURLOPT_URL, url.c_str());
    return *this;
}

CurlWrapper& CurlWrapper::setPostFields(const std::string& data) {
    curl_easy_setopt(easyHandle.get(), CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(easyHandle.get(), CURLOPT_POSTFIELDSIZE, data.length());
    return *this;
}

CurlWrapper& CurlWrapper::addHeader(const std::string& header) {
    headers = curl_slist_append(headers, header.c_str());
    return *this;
}

std::string CurlWrapper::perform() {
    if (headers) {
        curl_easy_setopt(easyHandle.get(), CURLOPT_HTTPHEADER, headers);
    }

    responseBuffer.clear();
    CURLcode res = curl_easy_perform(easyHandle.get());
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl_easy_perform() failed: ") + curl_easy_strerror(res));
    }

    long httpCode = 0;
    curl_easy_getinfo(easyHandle.get(), CURLINFO_RESPONSE_CODE, &httpCode);
    if (httpCode >= 400) {
        throw std::runtime_error("HTTP error: " + std::to_string(httpCode) + "\nResponse: " + responseBuffer);
    }

    return responseBuffer;
}