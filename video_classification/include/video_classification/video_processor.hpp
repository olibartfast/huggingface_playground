#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class VideoProcessor {
public:
    struct VideoInfo {
        int totalFrames;
        double fps;
        double duration;
    };

    struct WindowIndices {
        std::vector<int> indices;
        double startTime;
        double endTime;
    };

    VideoProcessor();
    ~VideoProcessor();

    bool openVideo(const std::string& videoPath);
    VideoInfo getVideoInfo() const;
    std::vector<WindowIndices> splitVideoIntoWindows(int windowSize, float samplingFps) const;
    std::vector<cv::Mat> extractFrames(const std::vector<int>& indices);
    std::vector<float> preprocessFrames(const std::vector<cv::Mat>& frames, int targetSize = 224);
    std::vector<cv::Mat> padVideoFrames(const std::vector<cv::Mat>& frames, int targetLength);

private:
    cv::VideoCapture cap;
    VideoInfo info;
};