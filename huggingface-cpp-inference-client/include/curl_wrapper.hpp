#pragma once

#include <string>
#include <curl/curl.h>

class CurlGlobalManager {
public:
    static CurlGlobalManager& getInstance();
    CurlGlobalManager(const CurlGlobalManager&) = delete;
    CurlGlobalManager& operator=(const CurlGlobalManager&) = delete;

private:
    CurlGlobalManager();
    ~CurlGlobalManager();
};

class CurlEasyHandle {
private:
    CURL* handle;

public:
    CurlEasyHandle();
    ~CurlEasyHandle();
    CURL* get();

    CurlEasyHandle(const CurlEasyHandle&) = delete;
    CurlEasyHandle& operator=(const CurlEasyHandle&) = delete;
};

class CurlWrapper {
private:
    CurlEasyHandle easyHandle;
    std::string responseBuffer;
    struct curl_slist* headers;

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s);

public:
    CurlWrapper();
    ~CurlWrapper();

    CurlWrapper& setUrl(const std::string& url);
    CurlWrapper& setPostFields(const std::string& data);
    CurlWrapper& addHeader(const std::string& header);
    std::string perform();
};